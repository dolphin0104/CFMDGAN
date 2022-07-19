import logging
import time
import numpy as np
import os
import shutil
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from configs import GET_CONFIGS
from dataset.blur_300vw import get_dataloader
from model.discriminator import CFMDDiscriminator_SingleStage
from model.mimo_conada import CMD_MIMOUNet
from utils import AverageMeter, make_dir, setup_logger
from loss import ganloss
# pip install lpips
import lpips

class Solver(object):
    def __init__(self):
        ##############################################
        # Set CONFIGS
        cfg = GET_CONFIGS()
        for key in cfg:
            setattr(self, key, cfg[key])
        ##############################################               
        # set random seed       
        manual_seed = 1
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        ##############################################
        # set dirs
        self.VALID_DIR = os.path.join(self.SAVE_DIR, 'valid') 
        self.MODEL_DIR = os.path.join(self.SAVE_DIR, 'checkpoints')
        make_dir(self.SAVE_DIR)
        make_dir(self.VALID_DIR)
        make_dir(self.MODEL_DIR)
        # copy important files
        shutil.copy('./configs.py', self.SAVE_DIR)
        shutil.copy('./train.py', self.SAVE_DIR)
        shutil.copy('./model/mimo_conada.py', self.SAVE_DIR)
        shutil.copy('./model/discriminator.py', self.SAVE_DIR)
        # tensorboard         
        make_dir(os.path.join(self.SAVE_DIR, 'tensorboard'))
        self.board_writer = SummaryWriter(os.path.join(self.SAVE_DIR, 'tensorboard'))  
        ##############################################
        setup_logger('configs', self.SAVE_DIR, 'configs', level=logging.INFO, screen=True) 
        setup_logger('valid', self.SAVE_DIR, 'valid', level=logging.INFO, is_formatter=True, screen=False) 
        self.config_logger = logging.getLogger('configs')  # training logger
        self.valid_logger = logging.getLogger('valid')  # validation logger
        # save once more & log configs
        for k, v in cfg.items():
            log = '{} : {}'.format(k, v)
            self.config_logger.info(log)
        ##############################################
        # data loader 
        self.train_loader, self.valid_loader = get_dataloader(cfg)
        ##############################################
        # set training GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] =  self.CUDA_VISIBLE_DEVICES 
        torch.backends.cudnn.benckmark = True
        self.device = torch.device('cuda')
        ##############################################
        # set models          
        self.net = OrderedDict()    
        # generator
        self.net['netG'] = CMD_MIMOUNet(**self.GENERATRO_ARGS).to(self.device)                 
        # discriminator
        if self.LOSS_WEIGHT['adversarial_loss']:
            self.net['netD'] = CFMDDiscriminator_SingleStage(**self.DISCRIMINATOR_ARGS).to(self.device)                               
        # lpips
        self.lpips = lpips.LPIPS(net='alex').to(self.device)
        # if use multi gpus
        if len(self.CUDA_VISIBLE_DEVICES.split(",")) > 1:
            for net_type, _ in self.net.items():                                 
                self.net[net_type] = nn.DataParallel(self.net[net_type])
            self.lpips = nn.DataParallel(self.lpips)
        self.lpips.eval()
        ##############################################
        # optimizers & schedulers
        self.optimizers = OrderedDict()
        for net_type, _ in self.net.items():            
            self.optimizers[net_type] = torch.optim.Adam(            
                    self.net[net_type].parameters(), lr=1e-4, betas=(0.9, 0.999))
              
        self.schedulers = OrderedDict()
        for key, _ in self.optimizers.items():
            self.schedulers[key] = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[key], gamma=0.99) 
        ##############################################        
        # Loss Functions & Loss Networks
        self.loss_dict = OrderedDict()
        self.loss_dict['netG_loss'] = AverageMeter()

        self.pix_loss = nn.L1Loss().to(self.device) 
        self.loss_dict['netG_pixel_loss'] = AverageMeter() 

        self.mse_loss = nn.MSELoss().to(self.device)
        if self.LOSS_WEIGHT['adversarial_loss']:
            self.adv_loss = ganloss.GAN_Loss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0).to(self.device)                   
            
            self.loss_dict['netD_loss'] = AverageMeter()             
            self.loss_dict['netD_real_loss'] = AverageMeter()
            self.loss_dict['netD_fake_loss'] = AverageMeter() 
            self.loss_dict['netD_ar_loss'] = AverageMeter()

            self.loss_dict['netG_adv_loss'] = AverageMeter()
            self.loss_dict['netG_ar_loss'] = AverageMeter()
                 
        
        ##############################################        
        # Others
        self.start_epoch = 1        
        self.curr_epoch = 0
        self.psnr_best = 0  
        self.lpips_best = 10000
        ##############################################
        # resume training, load models
        if self.RESUME_TRAINING:
            self.load_resume()
            self.load_network(is_resume=self.RESUME_TRAINING)
        

    # Train ============================
    def train(self):
        batch_time = AverageMeter()
   
        # validation using init weight..degug        
        self.valid(is_visual=True)
        
        for epoch in range(self.start_epoch, self.MAX_EPOCH + 1):             
            for _, nets in self.net.items():
                nets.train()
            self.curr_epoch = epoch
            end = time.time()
            for iters, data in enumerate(self.train_loader):
                blur, sharp, control_factor, _ = data
                blur, sharp, control_factor = self.prepare([blur, sharp, control_factor], self.device)                
                batch_size = blur.size(0)               
                                                
                #=================================================================        
                # 1. train discriminators
                #=================================================================   
                # if epoch % 2 == 0:                    
                if self.LOSS_WEIGHT['adversarial_loss']:
                    # 1. train deblur discriminator
                    netD_loss = 0
                    self.requires_grad(self.net['netG'], False)                                
                    self.requires_grad(self.net['netD'], True)                    
                    
                    fake = self.net['netG'](blur, control_factor)[-1]

                    dis_input_fake = torch.cat((fake, blur), dim=1)
                    dis_fake_global, dis_fake_local, dis_fake_label = self.net['netD'](dis_input_fake)
                    
                    dis_input_real = torch.cat((sharp.detach(), blur), dim=1)                           
                    dis_real_global, dis_real_local, dis_real_label  = self.net['netD'](dis_input_real)
                                      
                    
                    netD_real_loss = self.adv_loss(dis_real_global, True) + self.adv_loss(dis_real_local, True)
                    netD_fake_loss = self.adv_loss(dis_fake_global, False) + self.adv_loss(dis_fake_local, False)
                    netD_loss += (netD_real_loss + netD_fake_loss) / 2
                    self.loss_dict['netD_real_loss'].update(netD_real_loss.item(), batch_size)
                    self.loss_dict['netD_real_loss'].update(netD_real_loss.item(), batch_size)

                    netD_ar_loss =  (self.mse_loss(control_factor, dis_real_label) + self.mse_loss(control_factor, dis_fake_label))/2
                    netD_loss += netD_ar_loss
                    self.loss_dict['netD_ar_loss'].update(netD_ar_loss.item(), batch_size)
                    
                    self.loss_dict['netD_loss'].update(netD_loss.item(), batch_size)                            

                    self.optimizers['netD'].zero_grad()
                    netD_loss.backward()
                    self.optimizers['netD'].step()               
                                                                 
                #=================================================================
                # 2. train generator
                #=================================================================                              
                netG_loss = 0.
                self.requires_grad(self.net['netG'], True)
                if self.LOSS_WEIGHT['adversarial_loss']:
                    self.requires_grad(self.net['netD'], False)

                out4, out2, out = self.net['netG'](blur, control_factor)

                sharp2 = F.interpolate(sharp, scale_factor=0.5, mode='bilinear')
                sharp4 = F.interpolate(sharp, scale_factor=0.25, mode='bilinear')
                
                netG_pixel_loss = self.pix_loss(out, sharp) + self.pix_loss(out2, sharp2) + self.pix_loss(out4, sharp4) 
                netG_pixel_loss = netG_pixel_loss * self.loss_weight['pixel_loss_weight']  
                self.loss_dict['netG_pixel_loss'].update(netG_pixel_loss.item(), batch_size)
                netG_loss += netG_pixel_loss

                # adversarial and regression label loss
                if self.LOSS_WEIGHT['adversarial_loss']:
                    dis_input_fake = torch.cat((out, blur), dim=1)                
                    gen_fake_global, gen_fake_local, gen_fake_label = self.net['netD'](dis_input_fake)
                    netG_adv_loss = (self.adv_loss(gen_fake_global, True) + self.adv_loss(gen_fake_local, True)) * self.LOSS_WEIGHT['adversarial_loss']                    
                    netG_ar_loss = self.mse_loss(control_factor, gen_fake_label) * self.LOSS_WEIGHT['auxilirary_regresssion_loss']                    
                    self.loss_dict['netG_adv_loss'].update(netG_adv_loss.item(), batch_size)
                    netG_loss += netG_adv_loss 
                    self.loss_dict['netG_ar_loss'].update(netG_ar_loss.item(), batch_size)
                    netG_loss += netG_ar_loss 
                                                   
                self.loss_dict['netG_loss'].update(netG_loss.item(), batch_size)
                
                self.optimizers['netG'].zero_grad()
                netG_loss.backward()                
                self.optimizers['netG'].step() 

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                print("="*30)        
                log  = '[Epoch: {}|{}] [Iter: {}|{}({:.3f}s)]'.format(
                            self.curr_epoch, self.MAX_EPOCH, iters+1, len(self.train_loader), batch_time.avg,                  
                        )
                print(log)
                print('GPU: {}'.format(self.CUDA_VISIBLE_DEVICES))
                print('Keyword: {}'.format(self.keyword))
                for loss_name, value in self.loss_dict.items():
                    self.board_writer.add_scalar('{}'.format(loss_name), value.avg, epoch+1)
                    log = '\t[{} : {:.6f}]'.format(loss_name, value.avg)
                    print(log)                    
            
            is_visual = False
            if epoch % self.VIS_EPOCH == 1:
                is_visual= True
            psnr_avg, lpips_avg = self.valid(is_visual)
            # save models by psnr
            if psnr_avg > self.psnr_best:
                is_best = 'psnr'
                self.psnr_best = max(psnr_avg, self.psnr_best)            
                self.save_network(is_best)
            if lpips_avg < self.lpips_best:
                is_best = 'lpips'
                self.lpips_best = min(lpips_avg, self.lpips_best)            
                self.save_network(is_best)
            
            self.save_state(self.curr_epoch)  
            self.adjust_learning_rate()  
   

    # Valid ============================
    def valid(self, is_visual=False):             
        test_name = 'init_001642_blur_011'
        print("="*30) 
        print('Start Valid')
        for _, nets in self.net.items():
            nets.eval()

        psnr_result = [] 
        lpips_result = []
        for _, data in enumerate(self.valid_loader):            
            blur, sharp, control_factor, filename = data     
            blur_filename = filename[0].split('_')
            # validation on center frame
            cf = control_factor[0,0,0,0]
            if cf.item()==0.5:
                blur, sharp, control_factor = self.prepare([blur, sharp, control_factor], self.device) 
                with torch.no_grad():
                    out = self.net['netG'](blur, control_factor)[-1]
                    lp_dist = self.lpips.forward(out, sharp)          
                    lp_dist = lp_dist.view(-1).data.cpu().numpy()[0]
                    
                    pred_clip1 = torch.clamp(out, 0, 1)
                    p_numpy1 = pred_clip1.squeeze(0).cpu().numpy()
                    gt_img_np1 = sharp.squeeze(0).cpu().numpy()
                    psnr = cal_psnr(p_numpy1, gt_img_np1, data_range=1)

                    log = '[File:{}] [Deblur PSNR : {:.4f} LPIPS : {:.4f}] '.format(filename[0], psnr, lp_dist)         
                    print(log)
                    psnr_result.append(psnr)  
                    lpips_result.append(lp_dist) 
                
                    # if self.curr_epoch % self.VIS_EPOCH == 1 or is_debug:
                    if is_visual:
                        save_img_filename = '{}.png'.format(filename[0])
                        save_folder = os.path.join(self.VALID_DIR, 'ep_{:04d}'.format(self.curr_epoch))
                        make_dir(save_folder)                        
                        
                        save_name = os.path.join(save_folder, save_img_filename)
                        pred_clip1 += 0.5 / 255
                        pred = torchvision.transforms.functional.to_pil_image(pred_clip1.squeeze(0).cpu(), 'RGB')
                        pred.save(save_name)
            else:
                if is_visual and (test_name in filename[0]):
                    blur, sharp, control_factor = self.prepare([blur, sharp, control_factor], self.device) 
                    with torch.no_grad():
                        out = self.net['netG'](blur, control_factor)[-1]
                        pred_clip1 = torch.clamp(out, 0, 1)
                    
                        save_img_filename = '{}.png'.format(filename[0])
                        save_folder = os.path.join(self.VALID_DIR, 'ep_{:04d}'.format(self.curr_epoch))
                        make_dir(save_folder)                        
                        
                        save_name = os.path.join(save_folder, save_img_filename)
                        pred_clip1 += 0.5 / 255
                        pred = torchvision.transforms.functional.to_pil_image(pred_clip1.squeeze(0).cpu(), 'RGB')
                        pred.save(save_name)

        psnr_avg = sum(psnr_result)/len(psnr_result)    
        lpips_avg = sum(lpips_result)/len(lpips_result) 

        # record valid logs
        self.board_writer.add_scalar('valid_psnr_avg', psnr_avg, self.curr_epoch)   
        self.board_writer.add_scalar('valid_lpips_avg', lpips_avg, self.curr_epoch) 

        log = '[Epoch:{}|{}] [Deblur PSNR : {:.4f}, LPIPS : {:.4f}]'.format(
            self.curr_epoch, self.MAX_EPOCH, psnr_avg, lpips_avg) 
        print("="*30)
        print(log)
        self.valid_logger.info(log)
        print("="*30) 
        print('End Valid')
        return psnr_avg, lpips_avg
                        

    #-------------------------------------------------------    
    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def adjust_learning_rate(self):
        for _, schedulers in self.schedulers.items():
            schedulers.step()
        
    def prepare(self, l, device, volatile=False):
        def _prepare(tensor): return tensor.to(device)           
        return [_prepare(_l) for _l in l]

    def _getmodel(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        else: return model
    
    def print_network(self, model):
        s = str(self._getmodel(model))
        n = sum(map(lambda x: x.numel(), self._getmodel(model).parameters())) 
        net_struc_str = '{} - {}'.format(
            model.__class__.__name__, 
            self._getmodel(model).__class__.__name__
        )        
        log = 'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n)
        return log, s
    
    def load_network(self, is_resume=False, last_epochs=None): 
        for key, opts in self.net.items():
            if is_resume:
                apath = os.path.join(self.model_dir, '{}_last.pth'.format(key))
            elif last_epochs:
                apath = os.path.join(self.model_dir, '{}_ep{:03d}.pth'.format(key, last_epochs))           
            else:
                raise ValueError('model weight is not found')
            assert os.path.exists(apath)               
            self._getmodel(self.net[key]).load_state_dict(torch.load(apath))
            log = 'loaded trained generator {}..!'.format(apath)
            self.config_logger.info(log)            
           
    def save_network(self, is_best=None, last_epochs=None):
        for key, opts in self.net.items(): 
            state_dict = self._getmodel(self.net[key]).state_dict()
            for sd, param in state_dict.items():
                state_dict[sd] = param.cpu()             
            if is_best:
                save_filename = os.path.join(self.model_dir, '{}_{}_best.pth'.format(key, is_best))
                torch.save(state_dict, save_filename)            
            if last_epochs:
                save_filename = os.path.join(self.model_dir, '{}_ep{:03d}.pth'.format(key, last_epochs))
                torch.save(state_dict, save_filename)
            
            save_filename = os.path.join(self.model_dir, '{}_last.pth'.format(key))
            torch.save(state_dict, save_filename)
                    
    def save_state(self, last_epochs):
        state = OrderedDict()
        state['epoch'] = last_epochs  
        state['psnr_best'] = self.psnr_best
        state['lpips_best'] = self.lpips_best  
        for key, opts in self.optimizers.items():
            state['optimizers_{}'.format(key)] = opts.state_dict() 
        for key, schs in self.schedulers.items():
            state['schedulers_{}'.format(key)] = schs.state_dict()                                   
        save_filename = 'train_state.state'
        save_path = os.path.join(self.model_dir, save_filename)
        torch.save(state, save_path) 
    
    def load_resume(self):
        resume_state_path = os.path.join(self.model_dir, 'train_state.state')
        assert os.path.exists(resume_state_path) 
        resume_state = torch.load(resume_state_path, map_location='cuda:0')
        
        self.start_epoch = resume_state['epoch']+1
        self.psnr_best = resume_state['psnr_best']
        self.lpips_best = resume_state['lpips_best']

        for key, _ in self.optimizers.items():
            print('optimizers_{}'.format(key))
            self.optimizers[key].load_state_dict(
                resume_state['optimizers_{}'.format(key)]
            )
        for key, _ in self.schedulers.items():
            self.schedulers[key].load_state_dict(
                resume_state['schedulers_{}'.format(key)]
            )
 

if __name__ == '__main__':
    trainer = Solver()
    trainer.train()
    


        



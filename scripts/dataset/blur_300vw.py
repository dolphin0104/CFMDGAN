import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import joint_transforms

# FAB: A Robust Facial Landmark Detection Framework for Motion-Blurred Videos, 
# Keqiang Sun, Wayne Wu, Tinghao Liu, Shuo Yang, Quan Wang, Qiang Zhou, Chen Qian, and Zuochang Ye
# International Conference on Computer Vision (ICCV), 2019
# https://github.com/KeqiangSun/FAB
# The results reported on the paper are trained on 29 videos and tested on 10 videos.
# TRAIN_DIRS = [4, 15, 18, 31, 47, 124, 158, 212, 213, 402, 407, 410, 411, 412, 507, 508, 514, 516, 520, 525, 528, 533, 540, 546, 547, 548, 550, 559, 562]
# TRAIN_DIRS = ['{:03d}'.format(x) for x in TRAIN_DIRS]
# TEST_DIRS = [2, 11, 22, 509, 510, 521, 531, 541, 551, 557]
# TEST_DIRS = ['{:03d}'.format(x) for x in TEST_DIRS]


# Face Video Deblurring via 3D Facial Priors,
# Wenqi Ren, Jiaolong Yang, Senyou Deng, David Wipf, Xiaochun Cao, and Xin Tong
# International Conference on Computer Vision (ICCV), 2019
# https://github.com/rwenqi/3Dfacedeblurring
# They select 83 videos as training data and 9 videos as testing data from the 114 videos in the 300VW dataset.
TRAIN_DIRS = [
    '001', '002', '003', '007', '013', '015', '016', '017', '018', '019',
    '020', '022', '025', '027', '028', '029', '031', '033', '034', '035', 
    '041', '043', '044', '046', '047', '048', '049', '057', '059', '112',
    '113', '114', '115', '119', '123', '125', '126', '138', '143', '144',
    '150', '160', '203', '208', '212', '213', '214', '218', '223', '225',
    '401', '402', '403', '404', '405', '408', '412', '505', '506', '507',
    '508', '510', '511', '514', '517', '519', '520', '521', '524', '525',
    '526', '528', '529', '530', '531', '540', '541', '546', '547', '548',
    '553', '559', '562'
    ]
TEST_DIRS = ['009', '010', '037', '039', '053', '158', '211', '406', '522']

#----------------------------------------------------------------------------------
def get_dataloader(cfg):
    trainset = Dataset300VW(**cfg.TRAINSET_ARGS)
    train_loader = DataLoader(trainset, 
                              batch_size=cfg.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=cfg.NUM_WORKERS, 
                              drop_last=True, 
                              pin_memory=True)

    validset = Dataset300VW(**cfg.VALIDSET_ARGS)
    val_loader = DataLoader(validset, 
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.NUM_WORKERS, 
                            drop_last=True, 
                            pin_memory=True)
    
    return train_loader, val_loader


#----------------------------------------------------------------------------------
class Dataset300VW(Dataset):
    def __init__(self, 
                 data_dir, 
                 image_size=256, 
                 mode='train', 
                 num_images=None, 
                 is_redirection=True):              
        assert mode in ['train', 'valid', 'test']                       
        self.mode = mode
        self.num_images = num_images        
        self.image_size = image_size
        self.is_redirection = is_redirection
        # get file list                        
        filelist_txt = os.path.join(data_dir, '{}_filelist.txt'.format(mode))    
        if mode == 'valid':
            self.blur_dir =  os.path.join(data_dir, 'test', 'blur')        
            self.sharp_dir =  os.path.join(data_dir, 'test', 'sharp') 
        else:
            self.blur_dir =  os.path.join(data_dir, '{}'.format(mode), 'blur')        
            self.sharp_dir =  os.path.join(data_dir, '{}'.format(mode), 'sharp')       

        # set transforms
        if mode == 'train':
            self.trans = joint_transforms.Compose([
                joint_transforms.RandomSizeAndCrop(crop_size=image_size, scale_min=1.0, scale_max=1.5),                
                ])
        else:
            self.trans = joint_transforms.Compose([
                joint_transforms.Resize(image_size),
            ])          
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = self._getDirs(filelist_txt)
        
    def _getDirs(self, filelist_txt):      
        data = []        
        lines = [line.rstrip() for line in open(filelist_txt, 'r')]
        lines = lines[1:]
        for _, line in enumerate(lines):            
            dirname, blur_filename, total_num_frames = line.split()[:3]            
            total_num_frames = int(total_num_frames)
            sharp_framelist = line.split()[3:]          
            assert total_num_frames == len(sharp_framelist)              
            
            #### IMPORTANT!!! REDIRECTION IS FALSE => No FMR, Temporal Ordering #####################
            if self.is_redirection is False:
                sharp_framelist = sorted(sharp_framelist)
            
            blur_path = os.path.join(self.blur_dir, dirname)
            sharp_path = os.path.join(self.sharp_dir, dirname) 
            
            # blur image
            blur_img_path = os.path.join(blur_path, blur_filename + '.png')
            
            for ith, sharp_filename in enumerate(sharp_framelist):
                control_factor = float(ith)/(total_num_frames - 1)
                sharp_img_path = os.path.join(sharp_path, sharp_filename + '.png')
                re_filename = '{}_{}_{}'.format(dirname, blur_filename, sharp_filename)
                data.append([blur_img_path, sharp_img_path, control_factor, re_filename])            
        return data       

    def __getitem__(self, idx):
        blur_img_path = sharp_img_path = control_factor = re_filename = None 
        if self.mode == 'train':
            if self.num_images:
                sel = random.randint(0, len(self.data)-1-self.num_images)
                data = self.data[sel:sel+self.num_images]
                blur_img_path, sharp_img_path, control_factor, re_filename = data[idx]
            else:
                blur_img_path, sharp_img_path, control_factor, re_filename = self.data[idx]
        else:            
            blur_img_path, sharp_img_path, control_factor, re_filename = self.data[idx]
        
        # read image & landmark
        blur = Image.open(blur_img_path)
        sharp = Image.open(sharp_img_path)                
        # augmentation
        blur, sharp = self.trans(blur, sharp)

        # control factor        
        control_factor = torch.full((1, self.image_size, self.image_size), control_factor)
                
        # to tensor
        blur = self.to_tensor(blur)
        sharp = self.to_tensor(sharp)                       
        return blur, sharp, control_factor, re_filename

    def __len__(self):
        if self.mode == 'train':
            if self.num_images:
                return self.num_images
            else: len(self.data)
        else: return len(self.data)

         



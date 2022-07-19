
from easydict import EasyDict

def GET_CONFIGS():
    cfg = EasyDict() 
    #=======================
    # 0. CPU/GPU Setting, Resume Training
    #=======================        
    cfg.CPU = False    
    cfg.CUDA_VISIBLE_DEVICES = "0"  # set gpu device ids   # default 0     
    #=======================
    # 1. Dataset & Save Path
    #=======================  
    cfg.SAVE_DIR = 'you/path/to/300vw'   
    cfg.BATCH_SIZE = 8
    cfg.NUM_WORKERS = 16
    cfg.TRAINSET_ARGS = {
        'data_dir': 'you/path/to/300vw',
        'image_size':  256,
        'mode': 'train', 
        'num_images': 12000, 
        'is_redirection': True
    }
    cfg.VALIDSET_ARGS = {
        'data_dir': 'you/path/to/300vw',
        'image_size': 256,
        'mode': 'valid',
        'num_images': None, 
        'is_redirection': True
    }   
    #=======================
    # 2. Model setting 
    #=======================   
    cfg.GENERATRO_ARGS = {   
        'use_CADC': True,
        'use_CACA': True,
        'num_res' : 8,
    }
    cfg.DISCRIMINATOR_ARGS = {
        'in_channels': 6, 
        'image_size': 256,
        'min_feat_size': 8,        
        'use_spectral_norm': True    
    }
    #=======================
    # 3. Loss Types & Weigths
    #=======================        
    cfg.GAN_TYPE = 'vanilla'
    cfg.LOSS_WEIGHT = {
        'pixel_loss': 1,
        'adversarial_loss': 0.01,
        'auxilirary_regresssion_loss': 0.05,
    }    
    
    #=======================
    # 4. Other Settings & Hyperparams 
    #=======================    
    # epochs    
    cfg.MAX_EPOCH = 200
    cfg.VIS_EPOCH = 10  
    # If start training from begining => False, 
    # Elif resume training from the last training state => True       
    cfg.RESUME_TRAINING = False  
    return cfg
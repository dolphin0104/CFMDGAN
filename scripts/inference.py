import os
import cv2
from PIL import Image
from math import log10
import torch
import torch.nn as nn
from torchvision import transforms

from model.mimo_conada import CMD_MIMOUNet
from configs import GET_CONFIGS
from utils import make_dir

##############################################
# pre-trained model path
frame_num = 51 # set any number for extract output seqence
model_path = 'path/to/cfmdgan/model_best.pth'
test_blur_img_dir = 'path/to/test'
save_result_dir = 'path/to/save/result'
make_dir(save_result_dir)


# load model
device = torch.device('cuda')
cfg = GET_CONFIGS()
model = CMD_MIMOUNet(**cfg.GENERATRO_ARGS).to(device)
if isinstance(model, nn.DataParallel):
    model = model.module
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()


rgb_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

with torch.no_grad():
    for imgs in os.listdir(test_blur_img_dir):
        img_path = os.path.join(test_blur_img_dir, imgs)
        img_filename = os.path.splitext(os.path.split(img_path)[-1])[0]

        img_pil = Image.open(img_path)
        img_tensor = rgb_to_tensor(img_pil)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.to(device)

        b, _, h, w = img_tensor.shape           
        for ith in range(frame_num):
            control_factor = float(ith)/(frame_num - 1)
            c_tensor = torch.full((b,1, h, w), control_factor).to(device)               
        
            out_img = model(img_tensor, c_tensor)[-1]  
            out_clip = torch.clamp(out_img, 0, 1) 
            out_clip += 0.5/255
            save_img_filename = '{}_fr{:03d}.png'.format(img_filename, ith)
            save_path = os.path.join(save_result_dir, save_img_filename)      
            pred = transforms.functional.to_pil_image(out_clip.squeeze(0).cpu(), 'RGB')
            pred.save(save_path)
            print(save_img_filename)


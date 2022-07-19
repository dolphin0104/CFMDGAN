import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class DoubleConv(nn.Module):
    def __init__(self, dim_in, dim_out, use_spectral_norm=True):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim_out, dim_out, 3, 1, 1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class UnetDown(nn.Module):
    def __init__(self, dim_in, dim_out, use_spectral_norm=True, use_multiscale=False):
        super(UnetDown, self).__init__()
        self.use_multiscale = use_multiscale
        self.down = nn.AvgPool2d(2)
        if self.use_multiscale:
            self.from_rgb = spectral_norm(nn.Conv2d(in_channels=3, out_channels=dim_in, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
            dim_in = dim_in + dim_in
        self.layers = DoubleConv(dim_in, dim_out, use_spectral_norm)

    def forward(self, x):
        if self.use_multiscale:
            x1, x2 = x
            x1 = self.down(x1)
            x2 = self.from_rgb(x2)
            x = torch.cat([x1, x2], dim=1)
        else:
            x = self.down(x)
        x = self.layers(x)
        return x

class UnetUp(nn.Module):
    def __init__(self, dim_in, dim_out, use_spectral_norm=True):
        super(UnetUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layers = DoubleConv(dim_in, dim_out, use_spectral_norm)

    def forward(self, x1, x2):                
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.layers(x)
        return x       

class CFMDDiscriminator_SingleStage(nn.Module):
    def __init__(self, in_channels=6, image_size=256, min_feat_size=8, use_spectral_norm=True): 
        super(CFMDDiscriminator_SingleStage, self).__init__()        
        self.image_size = image_size  
        self.repeat_num = int(np.log2(image_size//min_feat_size)) + 1
        self.encode = nn.ModuleList()       
        self.decode = nn.ModuleList()

        min_dim = 64
        max_dim = 512
        dim_in = min_dim
        dim_out = min_dim     
        for i in range(self.repeat_num):
            if i == 0:
                self.encode.append(DoubleConv(in_channels, dim_in, use_spectral_norm))
            else:
                dim_out = min(dim_in*2, max_dim)            
                self.encode.append(UnetDown(dim_in, dim_out, use_multiscale=False))                                      
            dim_in = dim_out

        min_dim = 64
        max_dim = 512
        dim_in = min_dim
        dim_out = min_dim     
        for i in range(self.repeat_num):  
            dim_out = min(dim_in*2, max_dim)     
            if i == self.repeat_num - 1:
                self.decode.insert(0, DoubleConv(dim_out, dim_in, use_spectral_norm))
            else:
                self.decode.insert(0, UnetUp(dim_in + dim_out, dim_in, use_spectral_norm))                         
            dim_in = dim_out
                
        self.global_cls = nn.Sequential(
            spectral_norm(nn.Conv2d(max_dim, max_dim, 4, 2, 1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),            
            spectral_norm(nn.Conv2d(max_dim, 1, 1, bias=not use_spectral_norm), use_spectral_norm),            
            nn.Sigmoid(),
        )

        self.label_predict = nn.Sequential(
            spectral_norm(nn.Conv2d(min_dim, min_dim, 1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),            
            spectral_norm(nn.Conv2d(min_dim, 1, 1, bias=not use_spectral_norm), use_spectral_norm),            
            nn.Sigmoid(),
        )

        self.pixel_cls = nn.Sequential(
            spectral_norm(nn.Conv2d(min_dim, min_dim, 1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),            
            spectral_norm(nn.Conv2d(min_dim, 1, 1, bias=not use_spectral_norm), use_spectral_norm),            
            nn.Sigmoid(),
        )


    def forward(self, x):
        res_feats = []
        for i in range(self.repeat_num):
            x = self.encode[i](x)
            res_feats.insert(0, x)                
        
        global_cls = self.global_cls(res_feats[0]) 
                
        for i in range(self.repeat_num):                      
            if i == 0:                            
                x = self.decode[i](res_feats[0])
            else:
                x = self.decode[i](x, res_feats[i])

        pixel_cls = self.pixel_cls(x)                
        label_predict = self.label_predict(x)
        
        return global_cls, pixel_cls, label_predict
import torch
import torchvision.ops
import torch.nn as nn
import torch.nn.functional as F


class conv1x1(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(conv1x1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
    def forward(self, x):
        return self.layer(x)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Control Adaptive Channel Attention Layer
class CACALayer(nn.Module):
    def __init__(self, img_channel, cont_channel, reduction=4, bias=False):
        super(CACALayer, self).__init__()        
        # preconv
        self.conv = nn.Conv2d(img_channel+cont_channel, img_channel, 1, padding=0, bias=bias)       
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(img_channel, img_channel // reduction, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(img_channel // reduction, img_channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: img feature map
        :param x[1]: control factor
        '''
        # f_img, f_cont = x
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Control Adaptive DeformableConv2d
# Reference https://github.com/developer0hye/Simple-PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
class CADCConv2d(nn.Module):
    def __init__(self,
                 img_channel,
                 cont_channel,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(CADCConv2d, self).__init__()
        
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        # preconv
        self.conv = nn.Conv2d(img_channel+cont_channel, img_channel, 1, padding=0, bias=bias)

        self.offset_conv = nn.Conv2d(img_channel, 
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(img_channel, 
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=img_channel,
                                      out_channels=img_channel,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x_in):
        '''
        :param x_in[0]: img feature map
        :param x_in[1]: control factor
        ''' 
        # x, f_cont = x_in
        x = torch.cat(x_in, dim=1)
        x = self.conv(x)
        
        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ConAda_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, cont_channel, use_CADC=True, use_CACA=True):
        super(ConAda_ResBlock, self).__init__()
        self.use_CADC = use_CADC
        self.use_CACA = use_CACA
              
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=False, relu=True)        
        # CADC
        if use_CADC:            
            self.conv2 = CADCConv2d(out_channel, cont_channel)
        else: self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        # CACA
        if use_CACA:
            self.ch_att = CACALayer(out_channel, cont_channel)
        else:
            self.ch_att = CALayer(out_channel)

    def forward(self, x):
        '''
        :param x[0]: img feature map
        :param x[1]: control factor
        '''
        f_img, f_cont = x

        out = self.conv1(f_img)        
        if self.use_CADC:            
            out = self.conv2((out, f_cont))
        else: out = self.conv2(out)

        if self.use_CACA:
            out = self.ch_att((out, f_cont))
        else: out = self.ch_att(out)

        out = out + f_img 
        return (out, f_cont)

##########################################################################
class MappingNetwork(nn.Module):
    """
        Mapping control factor to feature space
    """
    def __init__(self, min_dim=32, max_dim=1024, num_shared_layers=8, num_unshared_layers=3):
        super(MappingNetwork, self).__init__()
        shared_layers = []
        for i in range(num_shared_layers):
            if i == 0:
                shared_layers.append(conv1x1(1, min_dim))               
            else:
                shared_layers.append(conv1x1(min_dim, min_dim))        
        self.shared = nn.Sequential(*shared_layers)        

        self.enc_unshared = nn.ModuleList()
        self.dec_unshared = nn.ModuleList()
        cur_ch = min_dim
        for i in range(num_unshared_layers):            
            self.enc_unshared.append(conv1x1(min_dim, cur_ch))
            self.dec_unshared.insert(0, conv1x1(min_dim, cur_ch))
            cur_ch = cur_ch * 2
            # cur_ch = min(cur_ch, max_dim)

    def forward(self, f_cont):
        fc_in = self.shared(f_cont)

        map_cont_enc = []
        map_cont_dec = []
        for i in range(len(self.enc_unshared)):                        
            map_cont_enc.append(self.enc_unshared[i](fc_in))
            map_cont_dec.insert(0, self.dec_unshared[-1-i](fc_in))
            fc_in = F.interpolate(fc_in, scale_factor=0.5)

        return map_cont_enc, map_cont_dec


##########################################################################
class EBlock(nn.Module):
    def __init__(self, out_channel, cont_channel, use_CADC=True, use_CACA=True, num_res=8):
        super(EBlock, self).__init__()

        # layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        layers = [ConAda_ResBlock(out_channel, out_channel, cont_channel, use_CADC, use_CACA) for _ in range(num_res)]       
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x[0]: img feature map
        :param x[1]: control factor
        '''
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, out_channel, cont_channel, use_CADC=True, use_CACA=True, num_res=8):
        super(DBlock, self).__init__()

        # layers = [ResBlock(channel, channel) for _ in range(num_res)]
        # layers = [ConAda_ResBlock(out_channel, out_channel, cont_channel, use_CADC, use_CACA) for _ in range(num_res)]
        layers = [ConAda_ResBlock(out_channel, out_channel, cont_channel, use_CADC, use_CACA) for _ in range(num_res)]        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x[0]: img feature map
        :param x[1]: control factor
        '''
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class CMD_MIMOUNet(nn.Module):
    def __init__(self, use_CADC=True, use_CACA=True, num_res=8):
        super(CMD_MIMOUNet, self).__init__()

        base_channel = 32
        # Mapping network
        self.mapping_net = MappingNetwork(min_dim=base_channel, max_dim=base_channel*4)

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, base_channel, use_CADC, use_CACA, num_res),
            EBlock(base_channel*2, base_channel*2, use_CADC, use_CACA, num_res),
            EBlock(base_channel*4, base_channel*4, use_CADC, use_CACA, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, base_channel * 4, use_CADC, use_CACA, num_res),
            DBlock(base_channel * 2, base_channel * 2, use_CADC, use_CACA, num_res),
            DBlock(base_channel, base_channel, use_CADC, use_CACA, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x, control_factor):
        map_cont_enc, map_cont_dec = self.mapping_net(control_factor)

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        output_imgs = list()
        # encoder_feats = list()

        x_ = self.feat_extract[0](x)
        res1, _ = self.Encoder[0]((x_, map_cont_enc[0]))
        
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2, _ = self.Encoder[1]((z, map_cont_enc[1]))

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z, _ = self.Encoder[2]((z, map_cont_enc[2]))
        
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        # encoder_feats.append(z)
        z, _ = self.Decoder[0]((z, map_cont_dec[0]))
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        output_imgs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        # encoder_feats.append(z)
        z, _ = self.Decoder[1]((z, map_cont_dec[1]))
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        output_imgs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        # encoder_feats.append(z)
        z, _ = self.Decoder[2]((z, map_cont_dec[2]))
        z = self.feat_extract[5](z)
        output_imgs.append(z+x)

        return output_imgs


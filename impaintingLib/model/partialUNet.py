import torch
import torch.nn.functional as F
from torch import nn

import impaintingLib.model.layer as layer

class DoublePartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(DoublePartialConv, self).__init__()
        self.conv1 = layer.PartialConv2d(in_channels, out_channels, 3, padding=1, return_mask=True)
        self.conv2 = layer.PartialConv2d(out_channels, out_channels, 3, padding=1, return_mask=True)
        if activation == 'leakyrelu':
            self.activtion = nn.LeakyReLU(0.2)
        else: 
            self.activtion = nn.ReLU()
            
    def forward(self, x, m):
        x, m = self.conv1(x, m)
        x = self.activtion(x)
        x, m = self.conv2(x, m)
        x = self.activtion(x)
        return x, m

class DownSamplePartialBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = DoublePartialConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2) 

    def forward(self, x, m):
        x_skip, m_skip = self.conv_block(x, m)
        x = self.maxpool(x_skip)
        m = self.maxpool(m)

        return x, m , x_skip, m_skip

class UpSamplePartialBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = DoublePartialConv(in_channels, out_channels)#, activation='leakyrelu')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, m, x_skip, m_skip):
        x = self.upsample(x)
        m = self.upsample(m)
        x = torch.cat([x, x_skip], dim=1)
#        m = torch.cat([m, m_skip], dim=1)
        x, m = self.conv_block(x, m)
        
        return x, m
    

class UNetPartialConv(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.downsample_block_1 = DownSamplePartialBlock(4, 64)
        self.downsample_block_2 = DownSamplePartialBlock(64, 128)
        self.middle_conv_block = DoublePartialConv(128, 256)        

            
        self.upsample_block_2 = UpSamplePartialBlock(128 + 256, 128)
        self.upsample_block_1 = UpSamplePartialBlock(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, 3, 1)
        
        
    def forward(self, x):
        #Generate mask
        m = ((x[:,:1] != 0)*1.)
        
        x, m, x_skip1, m_skip1 = self.downsample_block_1(x, m)
        x, m, x_skip2, m_skip2 = self.downsample_block_2(x, m)
        
        x, m = self.middle_conv_block(x, m)
        
        x, m = self.upsample_block_2(x, m, x_skip2, m_skip2) 
        x, m = self.upsample_block_1(x, m, x_skip1, m_skip1)
        
        out = self.last_conv(x)
        
        return out
import torch
import torch.nn as nn
from torchvision import *


class backbone(nn.Module):
  def __init__(self):
    super(backbone, self).__init__()

    model = models.efficientnet_b6(pretrained=True)
    m1 = list(model.features.children())
    self.l1 = nn.Sequential(*list(model.features.children())[:2])
    self.l2 = nn.Sequential(*list(model.features[2][:3]))
    self.l3 = nn.Sequential(*list(model.features[3][:4]))
    self.l4 = nn.Sequential(*list(model.features[4][:4]))
    self.l5 = nn.Sequential(*list(model.features[5][:2]))
    self.l6 = nn.Sequential(*list(model.features[6][:2]))

  def forward(self, x):
    x1 = self.l1(x)
    x2 = self.l2(x1)
    x3 = self.l3(x2)
    x4 = self.l4(x3)
    x5 = self.l5(x4)
    x6 = self.l6(x5)
    return [x1, x2, x3, x5, x6]


class ConvbnGelu(nn.Module):
  def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1):
    super(ConvbnGelu, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size, stride, padding),
        nn.GELU(),
        nn.BatchNorm2d(outchannels)
    )
  def forward(self, x):
    return self.conv(x)

class SEAttention(nn.Module):   #it gives channel attention
    def __init__(self, in_channels, reduced_dim=16):  #input_shape ---> output_shape
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.c1 = ConvbnGelu(inchannels=in_c, outchannels=out_c)
        self.c2 = ConvbnGelu(inchannels=out_c, outchannels=out_c)
        self.c3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SEAttention(in_channels=out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)
        x4 = x2 + x3
        x4 = self.relu(x4)
        return x4

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.r1(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(skip_c+out_c, out_c)
        # self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        # x = self.r2(x)
        return x

class GBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(GBlock, self).__init__()

        self.c = nn.Conv2d(in_c+in_c, out_c, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2], axis=1)
        x = self.c(x)
        x = self.sig(x)
        x = x * x3
        return x

class TFPNet(nn.Module):
  def __init__(self):
    super(TFPNet, self).__init__()
    
    self.backbone = backbone()

    self.ld1 = DecoderBlock(in_c=344, skip_c=200, out_c=256)
    self.hd1 = DecoderBlock(in_c=344, skip_c=200, out_c=256)
    self.id1 = DecoderBlock(in_c=344, skip_c=200, out_c=256)
    self.g1 = GBlock(in_c=256, out_c=256)

    self.ld2 = DecoderBlock(in_c=256, skip_c=72, out_c=128)
    self.hd2 = DecoderBlock(in_c=256, skip_c=72, out_c=128)
    self.id2 = DecoderBlock(in_c=256, skip_c=72, out_c=128)
    self.g2 = GBlock(in_c=128, out_c=128)

    self.ld3 = DecoderBlock(in_c=128, skip_c=40, out_c=64)
    self.hd3 = DecoderBlock(in_c=128, skip_c=40, out_c=64)
    self.id3 = DecoderBlock(in_c=128, skip_c=40, out_c=64)
    self.g3 = GBlock(in_c=64, out_c=64)

    self.ld4 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
    self.hd4 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
    self.id4 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
    self.g4 = GBlock(in_c=32, out_c=32)

    self.ld5 =  nn.Sequential(  nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                nn.Conv2d(32, 1, 3, 1, 1),
                                nn.Sigmoid()
                            )#ResidualBlock(in_c=32, out_c=3, last=True)
    self.hd5 = nn.Sequential(   nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                nn.Conv2d(32, 1, 3, 1, 1),
                                nn.Tanh()
                            )#ResidualBlock(in_c=32, out_c=1, last=True)
    self.id5 = nn.Sequential(   nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                nn.Conv2d(32, 3, 3, 1, 1),
                                nn.Tanh()
                            )#ResidualBlock(in_c=32, out_c=3, last=True)

  def forward(self, x):

    x1, x2, x3, x4, x5 = self.backbone(x)

    """block 1"""
    l1 = self.ld1(x5,x4)  
    h1 = self.hd1(x5,x4)  
    i1 = self.id1(x5,x4) 
    i1 = self.g1(l1,h1,i1)

    """block 2"""
    l2 = self.ld2(l1,x3)  
    h2 = self.hd2(h1,x3)  
    i2 = self.id2(i1,x3) 
    i2 = self.g2(l2,h2,i2)

    """block 3"""
    l3 = self.ld3(l2,x2)  
    h3 = self.hd3(h2,x2)  
    i3 = self.id3(i2,x2) 
    i3 = self.g3(l3,h3,i3)

    """block 4"""
    l4 = self.ld4(l3,x1)  
    h4 = self.hd4(h3,x1)  
    i4 = self.id4(i3,x1) 
    i4 = self.g4(l4,h4,i4)

    """block 5 [last block]"""
    l5 = self.ld5(l4)
    h5 = self.hd5(h4)
    i5 = self.id5(i4)

    return l5,h5,i5
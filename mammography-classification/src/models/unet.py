import torch
import torch.nn as nn

class UNet(nn.Module):
    """U-Net for mammography lesion segmentation"""
    
    def __init__(self, in_channels=1, num_classes = 1):
        super().__init__()
        #encoder (downsampling)
        
        self.enc1= self.conv_block(in_channels, 64)
        self.enc2= self.conv_block(64, 128)
        self.enc3= self.conv_block(128, 256)
        self.enc4= self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        #bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        #decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        #output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        """convolutional block: Conv2d -> ReLU -> Conv2d -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )    
    
    def forward(self, x):
        #encoder
        e1= self.enc1(x)
        e2= self.enc2(self.pool(e1))
        e3= self.enc3(self.pool(e2))
        e4= self.enc4(self.pool(e3))
        
        #bottleneck
        b= self.bottleneck(self.pool(e4))
        
        #decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.out(d1))    
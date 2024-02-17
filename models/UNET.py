import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        self.up_sample_mode = up_sample_mode
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode`")
        
        self.double_conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        # Resize x to match the dimensions of skip_input
        if x.shape != skip_input.shape:
            diffY = skip_input.size()[2] - x.size()[2]
            diffX = skip_input.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNetBaseline(nn.Module):
    def __init__(self, out_classes=8, up_sample_mode='conv_transpose'):
        super(UNetBaseline, self).__init__()
        self.up_sample_mode = up_sample_mode
        
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)

        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)

        # Upsampling Path
        self.up_conv4 = UpBlock(1024, 512, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(512, 256, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(256, 128, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128, 64, 64, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)

        x = self.conv_last(x)
        return x  # Optionally add a softmax or sigmoid here, depending on your task

# Example of model instantiation
model = UNetBaseline(out_classes=8)  # Replace numberOfClasses with your actual number of classes

import torch
from torch import nn

class ConvBatchnormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBatchnormRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                  nn.BatchNorm2d(out_channels, eps=1e-05), nn.ReLU(inplace=True))
        torch.nn.init.zeros_(self.conv[0].bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,  kernel_size, in_channels):
        super(ResBlock, self).__init__()
        features = in_channels
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, in_channels, kernel_size, padding=kernel_size//2)
    def forward(self, x):
        features = self.conv1(x)
        features = self.relu(features)
        features = self.conv2(features)
        return self.relu(features + x)
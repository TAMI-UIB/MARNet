from sympy import factorint
from torch import nn
import torch
from src.models.operators.basic_blocks import ConvBatchnormRelu

class EdgeProtector(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size, features=3):
        super(EdgeProtector, self).__init__()
        init_channels = x_channels + y_channels
        mid_channels = x_channels + y_channels + features
        final_channels = x_channels
        self.convbatchnorm1 = ConvBatchnormRelu(in_channels=init_channels, out_channels=mid_channels,
                                                kernel_size=kernel_size)
        self.convbatchnorm2 = ConvBatchnormRelu(in_channels=mid_channels, out_channels=mid_channels,
                                                kernel_size=kernel_size)
        self.convbatchnorm3 = ConvBatchnormRelu(in_channels=mid_channels, out_channels=final_channels,
                                                kernel_size=kernel_size)

    def forward(self, x, y):
        features_1 = self.convbatchnorm1(torch.cat((x, y), dim=1))
        features_2 = self.convbatchnorm2(features_1)
        features_3 = self.convbatchnorm3(features_2)
        features_3 = features_3 + x
        return features_3

class Up(nn.Module):
    def __init__(self,  in_channels, support_channels, sampling, kernel_size=3, depthwise_coef=1, features=3):
        super(Up, self).__init__()
        self.sampling = sampling

        self.pan_list = []
        self.steps = 0
        conv_trans = []
        edge_protector = []
        for p, exp in sorted(factorint(sampling).items(), reverse=True):
            kernel = p + 1 if p % 2 == 0 else p + 2
            for _ in range(0, exp):
                conv_trans.append(nn.ConvTranspose2d(in_channels=in_channels,
                                                     out_channels=in_channels,
                                                     kernel_size=kernel,
                                                     stride=p,
                                                     padding=kernel//2,
                                                     bias=False,
                                                     groups=in_channels,
                                                     output_padding=p-1))
                edge_protector.append(EdgeProtector(in_channels, support_channels, kernel_size=kernel_size, features=features))
                self.steps = self.steps + 1
        self.last_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depthwise_coef,
            kernel_size=3,
            padding=1,
            groups=in_channels,)
        self.last_conv.weight.data = (1 / 16) * torch.ones(self.last_conv.weight.data.size())
        self.conv_trans = nn.ModuleList(conv_trans)
        self.edge_protector = nn.ModuleList(edge_protector)

    def forward(self, input):
        for i in range(self.steps):
            input = self.conv_trans[i](input)
            input = self.edge_protector[i](input, self.pan_list[self.steps-1-i]/10)
        input = self.last_conv(input)
        return input

    def set_pan(self, pan_list):
        self.pan_list = pan_list

import torch
from sympy import factorint
from torch import nn

class PatchAvg(nn.Module):
    def __init__(self, sampling):
        super(PatchAvg, self).__init__()
        self.sampling = sampling

    def forward(self, input):
        B, C, H, W = input.size()
        downsamp = torch.zeros(B, C, H//self.sampling, W//self.sampling, device=input.device)
        for i in range(self.sampling):
            for j in range(self.sampling):
                downsamp[:,:,:,:] += input[:,:,i::self.sampling,j::self.sampling] / ( self.sampling ** 2 )
        return downsamp

class Down(nn.Module):
    def __init__(self, channels, sampling):
        super(Down, self).__init__()
        self.sampling = sampling
        conv_layers = []
        decimation_layers = []

        for p, exp in factorint(sampling).items():
            kernel = p+1 if p %2 == 0 else p+2
            for _ in range(0, exp):
                conv = nn.Conv2d(in_channels=channels,out_channels=channels,
                                             kernel_size=kernel,
                                             padding=kernel // 2,
                                             bias=False,
                                             groups=channels
                                            # OJO AMB EL GROUPS = channels
                                             )
                with torch.no_grad():
                    conv.weight.zero_()
                    center = conv.kernel_size[0] // 2  # Asumimos kernel cuadrado.
                    for i in range(channels):
                        conv.weight[i, 0, center, center] = 1.0
                conv_layers.append(conv)
                decimation_layers.append(PatchAvg(p))

        self.conv_k = nn.ModuleList(conv_layers)
        self.decimation = nn.ModuleList(decimation_layers)

    def forward(self, input):
        list = [input]
        for i, conv in enumerate(self.conv_k):
            input = conv(input)
            input = self.decimation[i](input)
            list.append(input)
        return list[:-1], input

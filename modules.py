import torch
from sympy import factorint
from torch import nn
from torch.nn.functional import unfold


class p_Decimation(nn.Module):
    def __init__(self, sampling):
        super(p_Decimation, self).__init__()
        self.sampling = sampling

    def forward(self, input):
        height, width = input.size(2), input.size(3)
        return input[:, :, 0:height:self.sampling, 0:width:self.sampling]


class general_Downsamp(nn.Module):
    def __init__(self, channels, sampling):
        super(general_Downsamp, self).__init__()
        self.sampling = sampling
        conv_layers = []
        decimation_layers = []

        for p, exp in factorint(sampling).items():
            kernel = 2*p+1
            for _ in range(0, exp):
                conv_layers.append(nn.Conv2d(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=kernel,
                                             padding=kernel // 2,
                                             bias=False))
                decimation_layers.append(p_Decimation(p))

        self.conv_k = nn.ModuleList(conv_layers)
        self.decimation = nn.ModuleList(decimation_layers)

    def forward(self, input):
        list = [input]
        for i, conv in enumerate(self.conv_k):
            input = conv(input)
            input = self.decimation[i](input)
            list.append(input)
        return list[:-1], input


class ConvBatchnormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBatchnormRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                  nn.BatchNorm2d(out_channels, eps=1e-05), nn.ReLU(inplace=True))
        torch.nn.init.zeros_(self.conv[0].bias)

    def forward(self, x):
        x = self.conv(x)
        return x


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


class general_Upsampling(nn.Module):
    def __init__(self,  in_channels, support_channels, sampling, kernel_size=3, depthwise_coef=1, features=3):
        super(general_Upsampling, self).__init__()
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

class SpatialWeightsEsDot(torch.nn.Module):
    def __init__(self, channels,  window_size, patch_size):
        super(SpatialWeightsEsDot, self).__init__()
        self.channels = channels
        self.phi = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Conv2d(channels, channels, 1, bias=False)
        self.window_size = window_size
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-6

    def forward(self, u):
        b, c, h, w = u.size()
        phi = self.phi(u)
        #self.phi(u)
        theta = self.phi(u)
        #self.theta(u)
        theta = unfold(theta, self.patch_size, padding=self.patch_size // 2)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, -1)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, h, w)
        theta = theta.permute(0, 3, 4, 1, 2)


        phi = unfold(phi, self.patch_size, padding=self.patch_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, h, w)
        phi = unfold(phi, self.window_size, padding=self.window_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, self.window_size * self.window_size, h, w)
        phi = phi.permute(0, 3, 4, 1, 2)

        att = torch.matmul(theta, phi)

        return self.softmax(att)


class SelfAttentionEsDot(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(SelfAttentionEsDot, self).__init__()
        self.pan_channels = pan_channels
        self.u_channels = u_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.spatial_weights = SpatialWeightsEsDot(channels=pan_channels, window_size=window_size, patch_size=patch_size)
        self.g = nn.Conv2d(u_channels, u_channels, 1, bias=False)
    def forward(self, u, pan):
        b, c, h, w = u.size()
        weights = self.spatial_weights(pan)
        g = self.g(u)  # [b, 3, h, w]
        g = unfold(g, self.window_size, padding=self.window_size // 2)
        g = g.view(b, self.u_channels, self.window_size * self.window_size, -1)
        g = g.view(b, self.u_channels, self.window_size * self.window_size, h, w)
        g = g.permute(0, 3, 4, 2, 1)
        return torch.matmul(weights, g).permute(0, 4, 1, 2, 3)


class MultiHeadAttentionEsDot(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(MultiHeadAttentionEsDot, self).__init__()
        self.geometric_head = SelfAttentionEsDot(u_channels=u_channels, pan_channels=pan_channels, patch_size=patch_size, window_size=window_size)
        self.spectral_head = SelfAttentionEsDot(u_channels=u_channels, pan_channels=u_channels, patch_size=1, window_size=window_size)
        self.mix_head = SelfAttentionEsDot(u_channels=u_channels, pan_channels=pan_channels+u_channels, patch_size=patch_size, window_size=window_size)
        self.mlp = nn.Linear(3, 1)

    def forward(self, u, pan):
        head1 = self.geometric_head(u, pan)
        head2 = self.spectral_head(u, u)

        head3 = self.mix_head(u, torch.concat([u,pan],dim=1))

        return self.mlp(torch.concat([head1, head2, head3], dim=4)).squeeze(4)


class MultiAttentionResNet6EsDot_2(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features_channels, patch_size, window_size, kernel_size=3):
        super(MultiAttentionResNet6EsDot_2, self).__init__()

        self.features_channels = features_channels
        self.aux_channels = 5
        self.ResNet_features = nn.Conv2d(in_channels=u_channels, out_channels=features_channels - self.aux_channels,
                                         kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.res1 = ResBlock(kernel_size=kernel_size, in_channels=features_channels)
        self.res2 = ResBlock(kernel_size=kernel_size, in_channels=features_channels)
        self.res3 = ResBlock(kernel_size=kernel_size, in_channels=features_channels)

        self.MultiAtt_features_u = nn.Conv2d(in_channels=u_channels, out_channels=self.aux_channels, kernel_size=kernel_size,
                                             stride=1,
                                             bias=False, padding=kernel_size // 2)
        self.MultiAtt_features_pan = nn.Conv2d(in_channels=pan_channels, out_channels=3, kernel_size=kernel_size,
                                               stride=1,
                                               bias=False, padding=kernel_size // 2)


        self.multi_head = MultiHeadAttentionEsDot(u_channels=self.aux_channels, pan_channels=3, patch_size=patch_size,
                                                  window_size=window_size)
        self.recon= nn.Sequential(*[
            nn.Conv2d(in_channels=features_channels+self.aux_channels, out_channels=features_channels, kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2),
            nn.BatchNorm2d(features_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_channels, out_channels=u_channels, kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2),
            nn.BatchNorm2d(u_channels),
            nn.ReLU()])

    def forward(self, u, pan):
        # Multi Attention Component
        u_features = self.MultiAtt_features_u(u)
        pan_features = self.MultiAtt_features_pan(pan)
        u_multi_att = self.multi_head(u_features, pan_features)
        # Residual Component
        u_features = self.ResNet_features(u)
        res1 = self.res1(torch.concat([u_features, u_multi_att], dim=1))
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res = torch.cat([res3, u_multi_att], dim=1)
        return self.recon(res) + u


def Prox_Squared_L2_with_inner_product(input, inner_product, hyper_parameter, step_size):
    alpha = hyper_parameter
    sigma = step_size
    coef = 1./(1.+sigma/alpha)
    return coef * (input - sigma * inner_product)


def Prox_Dual_L1(input, inner_product, hyper_parameter, step_size):
    argument = input-step_size*inner_product
    return torch.where(argument>hyper_parameter, hyper_parameter, argument)

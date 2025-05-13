import torch
from torch import nn
from torch.nn.functional import unfold
from src.models.operators.basic_blocks import ResBlock

class AttentionWeights(torch.nn.Module):
    def __init__(self, channels,  window_size, patch_size):
        super(AttentionWeights, self).__init__()
        self.channels = channels
        self.phi = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Conv2d(channels, channels, 1, bias=False)
        self.window_size = window_size
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.InstanceNorm1d(channels * patch_size * patch_size)
        self.eps = 1e-6

    def forward(self, u):
        b, c, h, w = u.size()
        phi = self.phi(u)
        #self.phi(u)
        theta = self.theta(u)
        #self.theta(u)
        theta = unfold(theta, self.patch_size, padding=self.patch_size // 2)
        theta = self.norm(theta)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, -1)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, h, w)

        theta = theta.permute(0, 3, 4, 1, 2)


        phi = unfold(phi, self.patch_size, padding=self.patch_size // 2)
        phi = self.norm(phi)
        phi = phi.view(b, c * self.patch_size * self.patch_size, h, w)
        phi = unfold(phi, self.window_size, padding=self.window_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, self.window_size * self.window_size, h, w)
        phi = phi.permute(0, 3, 4, 1, 2)

        att = torch.matmul(theta, phi)

        return self.softmax(att)
class HeadAttention(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(HeadAttention, self).__init__()
        self.pan_channels = pan_channels
        self.u_channels = u_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.spatial_weights = AttentionWeights(channels=pan_channels, window_size=window_size, patch_size=patch_size)
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
class MHA(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(MHA, self).__init__()
        self.geometric_head = HeadAttention(u_channels=u_channels, pan_channels=pan_channels, patch_size=patch_size, window_size=window_size)
        self.spectral_head = HeadAttention(u_channels=u_channels, pan_channels=u_channels, patch_size=1, window_size=window_size)
        self.mix_head = HeadAttention(u_channels=u_channels, pan_channels=pan_channels+u_channels, patch_size=patch_size, window_size=window_size)
        self.mlp = nn.Linear(3, 1)

    def forward(self, u, pan):
        head1 = self.geometric_head(u, pan)
        head2 = self.spectral_head(u, u)

        head3 = self.mix_head(u, torch.concat([u,pan],dim=1))

        return self.mlp(torch.concat([head1, head2, head3], dim=4)).squeeze(4)

class MHAResNet(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features, patch_size, window_size, kernel_size=3):
        super(MHAResNet, self).__init__()

        self.features_channels = features
        self.aux_channels = 5
        self.ResNet_features = nn.Conv2d(in_channels=u_channels, out_channels=features - self.aux_channels,
                                         kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.res1 = ResBlock(kernel_size=kernel_size, in_channels=features)
        self.res2 = ResBlock(kernel_size=kernel_size, in_channels=features)
        self.res3 = ResBlock(kernel_size=kernel_size, in_channels=features)

        self.MultiAtt_features_u = nn.Conv2d(in_channels=u_channels, out_channels=self.aux_channels, kernel_size=kernel_size,
                                             stride=1,
                                             bias=False, padding=kernel_size // 2)
        self.MultiAtt_features_pan = nn.Conv2d(in_channels=pan_channels, out_channels=3, kernel_size=kernel_size,
                                               stride=1,
                                               bias=False, padding=kernel_size // 2)


        self.multi_head = MHA(u_channels=self.aux_channels, pan_channels=3, patch_size=patch_size,
                                                  window_size=window_size)
        self.recon= nn.Sequential(*[
            nn.Conv2d(in_channels=features + self.aux_channels, out_channels=features, kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(in_channels=features, out_channels=u_channels, kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2),
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
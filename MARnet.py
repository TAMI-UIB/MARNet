import torch
import torch.nn as nn
import torch.nn.functional as func

from modules import (general_Downsamp, general_Upsampling, MultiAttentionResNet6EsDot_2,
                                           Prox_Squared_L2_with_inner_product, Prox_Dual_L1)


# Whole network
class MARNet(nn.Module):

    def __init__(
            self,
            channels,
            multi_channels=1,
            iter_stages=3,
            device='cpu',
            sampling=6
    ):
        super(MARNet, self).__init__()
        self.out_channels = multi_channels
        self.sampling = sampling
        self.downsamp_ms = general_Downsamp(channels=multi_channels, sampling=sampling)
        self.downsamp = general_Downsamp(channels=channels, sampling=sampling)
        self.upsamp = general_Upsampling(channels, multi_channels, sampling, features=32)
        self.upsamp_ms = general_Upsampling(multi_channels, multi_channels, sampling, features=3)
        self.device = device

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.theta = torch.tensor(1.0)
        self.memory_features = 14
        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.tau = nn.Parameter(torch.tensor(0.1))

        self.middlestages = \
            nn.ModuleList([
                PrimalDualStage(hyper_channels=channels, multi_channels=multi_channels, sampling=sampling,
                                sigma=self.sigma, tau=self.tau, alpha=self.alpha, beta=self.beta, theta=self.theta
                                )
                for i in range(iter_stages)
            ])
        self.laststage = LastStage(channels, multi_channels, sampling=sampling)

    def forward(self, pan, hs):
        uk, hs, hs_lf, pan_lf, _ = self.initialization(hs, pan)

        list_uk = [uk]
        u_hat = uk
        _, q = self.downsamp(uk)
        r = pan_lf * uk
        for stage in self.middlestages:
            u_hat, uk, q, r = stage(u_hat, uk, q, r, hs=hs, hs_lf=hs_lf, pan=pan, pan_lf=pan_lf)
            list_uk.append(uk)

        u_out, F_data, F_rm = self.laststage(uk, hs, hs_lf, pan, pan_lf)

        list_uk.append(u_out)
        return {"pred": u_out}

    def initialization(self, hs, pan):
        # u = torch.zeros((hs.size(0), hs.size(1), pan.size(2), pan.size(3))).to(self.device)
        u = func.interpolate(hs, scale_factor=self.sampling)

        pan_list, pan_low = self.downsamp_ms(pan)
        self.upsamp_ms.set_pan(pan_list)
        self.upsamp.set_pan(pan_list)
        for stage in self.middlestages:
            stage.upsamp.set_pan(pan_list)
        pan_lf = self.upsamp_ms(pan_low)
        hs_lf = self.upsamp(hs)

        return u, hs, hs_lf, pan_lf, None


class PrimalDualStage(nn.Module):
    def __init__(
            self,
            hyper_channels,
            multi_channels,
            sigma,
            tau,
            alpha,
            beta,
            theta,
            sampling
    ):
        super(PrimalDualStage, self).__init__()
        # Hyper parameteres
        self.alpha = alpha
        self.beta = beta
        # Step-size
        self.tau = tau
        self.sigma = sigma
        # Relaxation parameters
        self.theta = theta
        # Operators
        self.downsamp = general_Downsamp(channels=hyper_channels, sampling=sampling)
        self.upsamp = general_Upsampling(hyper_channels, multi_channels, sampling, features=32)
        # Proximities
        self.prox_q = Prox_Squared_L2_with_inner_product
        self.prox_r = Prox_Dual_L1
        self.prox_u = MultiAttentionResNet6EsDot_2(u_channels=hyper_channels, pan_channels=multi_channels,
                                                   features_channels=68, patch_size=3, window_size=9, kernel_size=3)

    def forward(self, u_hat, uk, q, r, hs, hs_lf, pan, pan_lf):
        sigma = self.sigma
        alpha = self.alpha
        beta = self.beta
        u_prev = uk
        # Compute the ascending step of dual variables
        q = self.prox_q(input=self._q_argument(q=q, u_hat=u_hat), inner_product=hs, step_size=sigma, hyper_parameter=alpha)
        r = self.prox_r(input=self._r_argument(r=r, u_hat=u_hat, pan_lf=pan_lf), inner_product=pan*hs_lf, step_size=sigma, hyper_parameter=beta)
        # Compute the descending step of primal variable
        uk = self.prox_u(self._u_argument(u=uk, q=q, r=r, pan_lf=pan_lf), pan)
        # Overrelaxation
        u_hat = self._update_u_hat(u=u_prev, u_upd=uk)
        return u_hat, uk, q, r

    def _q_argument(self, q, u_hat):
        _, dbu = self.downsamp(u_hat)
        return q + self.sigma * dbu

    def _r_argument(self, r, u_hat, pan_lf):
        return r + self.sigma * pan_lf * u_hat

    def _u_argument(self, u, q, r, pan_lf):
        tau = self.tau
        up_q = self.upsamp(q)
        return u + tau * up_q + tau * pan_lf * r

    def _update_u_hat(self, u, u_upd):
        theta = self.theta
        return u + theta * (u - u_upd)


class LastStage(nn.Module):
    def __init__(self, hyper_channels, multi_channels, sampling):
        super(LastStage, self).__init__()

        self.resnet = MultiAttentionResNet6EsDot_2(u_channels=hyper_channels, pan_channels=multi_channels,
                                                   features_channels=68, patch_size=3, window_size=9, kernel_size=3)
        self.downsamp = general_Downsamp(channels=hyper_channels, sampling=sampling)

    def forward(self, u, hs, hs_lf, pan, pan_lf):
        u_out = self.resnet(u, pan)
        F = self.F_data(u, hs)
        L = self.F_rm(u, pan, hs_lf, pan_lf)

        return u_out, F, L

    def F_data(self, u, hs):
        # DBU-H
        _, dbu = self.downsamp(u)
        return dbu - hs

    def F_rm(self, u, pan, hs_lf, pan_lf):
        # P̄U-PH
        return pan_lf * u - pan * hs_lf

if __name__ == '__main__':
    C, H, W = 8, 128, 128
    sampling = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ms = torch.rand(1, C, H//sampling, W//sampling).to(device)
    pan = torch.rand(1, 1, H, W).to(device)
    model = MARNet(channels=C, multi_channels=1, iter_stages=3, device = device, sampling=sampling).to(device)
    output = model(pan, ms)
    fused = output['pred']
    print(fused.size())
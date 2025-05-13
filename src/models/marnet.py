import torch
import torch.nn as nn
import torch.nn.functional as func

from src.models.operators.closed_form_prox import Prox_Dual_L1
from src.models.operators.closed_form_prox import Prox_Squared_L2_with_inner_product
from src.models.operators.downsampling import Down
from src.models.operators.nonlocal_prox import MHAResNet

from src.models.operators.upsampling import Up

prox_dict = {"MARNet": MHAResNet}

# Whole network
class MARNet(nn.Module):

    def __init__(
            self,
            hs_channels,
            ms_channels,
            iter_stages,
            sampling,
            features=64,
            patch_size=3,
            window_size=9,
    ):
        super(MARNet, self).__init__()
        self.out_channels = ms_channels
        self.sampling = sampling
        self.downsamp_ms = Down(channels=ms_channels, sampling=sampling)
        self.downsamp_hs = Down(channels=hs_channels, sampling=sampling)
        self.upsamp_hs = Up(hs_channels, ms_channels, sampling, features=32)
        self.upsamp_lf_ms = Up(hs_channels, ms_channels, sampling, features=32)

        self.pd_stages = \
            nn.ModuleList([
                PrimalDualStage(hyper_channels=hs_channels, multi_channels=ms_channels, sampling=sampling,
                                features=features, patch_size=patch_size, window_size=window_size,
                               )
                for i in range(iter_stages)
            ])
        self.post = MHAResNet(u_channels=hs_channels, pan_channels=ms_channels,
                              features=features, patch_size=patch_size, window_size=window_size,
                              kernel_size=3)


    def fine_tune(self,):
        for param in self.pd_stages.parameters():
            param.requires_grad = False

        for param in self.downsamp_hs.parameters():
            param.requires_grad = False

        for param in self.downsamp_ms.parameters():
            param.requires_grad = False

        for param in self.upsamp_hs.parameters():
            param.requires_grad = False


    def forward(self, pan, hs):
        uk, hs, hs_lf, pan_lf = self.initialization(hs, pan)

        list_uk = [uk]
        u_hat = uk
        _, q = self.downsamp_hs(uk)
        r = pan_lf * uk
        for stage in self.pd_stages:
            u_hat, uk, q, r = stage(u_hat, uk, q, r, hs=hs, hs_lf=hs_lf, pan=pan, pan_lf=pan_lf)
            list_uk.append(uk)

        u_out = self.post(uk, pan)

        return {"pred": u_out,  "u_list": list_uk}

    def initialization(self, hs, pan):

        hs_lf = func.interpolate(hs, scale_factor=self.sampling)
        pan_list, pan_low = self.downsamp_ms(pan)

        self.upsamp_hs.set_pan(pan_list)
        for stage in self.pd_stages:
            stage.upsamp.set_pan(pan_list)
        pan_lf = func.interpolate(pan_low, scale_factor=self.sampling)
        u = self.upsamp_hs(hs)
        return u, hs, hs_lf, pan_lf


class PrimalDualStage(nn.Module):
    def __init__(
            self,
            hyper_channels,
            multi_channels,
            sampling,
            features,
            patch_size,
            window_size,

    ):
        super(PrimalDualStage, self).__init__()
        # Hyper parameteres
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.theta = torch.tensor(1.0)
        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.tau = nn.Parameter(torch.tensor(0.1))

        # Operators
        self.downsamp = Down(channels=hyper_channels, sampling=sampling)
        self.upsamp = Up(hyper_channels, multi_channels, sampling, features=32)
        # Proximities
        self.prox_q = Prox_Squared_L2_with_inner_product
        self.prox_r = Prox_Dual_L1
        self.prox_u = MHAResNet(u_channels=hyper_channels, pan_channels=multi_channels,
                                features=features, patch_size=patch_size, window_size=window_size,
                                kernel_size=3)

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
        return u - tau * up_q - tau * pan_lf * r

    def _update_u_hat(self, u, u_upd):
        theta = self.theta
        return u_upd + theta * (u_upd - u)


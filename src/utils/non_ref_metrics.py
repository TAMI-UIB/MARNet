import math
import numbers

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import unfold as create_block_map

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()

        self.padding = kernel_size // 2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 2:
            self.conv = nn.Conv2d(in_channels=channels,
                                  out_channels=channels,
                                  kernel_size=kernel_size,
                                  groups=channels,
                                  padding='same',
                                  bias=False,
                                  padding_mode='replicate')

            self.conv.weight.data = kernel
            self.conv.weight.requires_grad = False


        else:
            raise RuntimeError(
                'Only 2 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input)


def decimation(x, scale):
    down = torch.zeros(x.size(0), x.size(1), x.size(2)//scale, x.size(3)//scale).to(x.device)
    for i in range(x.size(2)//scale):
        for j in range(x.size(3)//scale):
            down[:, :, i, j] = x[:, :, i*scale, j*scale]
    return down


def Q_index(x, y, block_size=8, stride=1):
    kernel = block_size
    stride = stride
    N, C, H, W = x.size()

    block_map_x = create_block_map(x, kernel_size=kernel, stride=stride)
    block_num = block_map_x.size(-1)
    block_map_x = block_map_x.view(N,C,kernel*kernel,block_num)
    block_map_y = create_block_map(y, kernel_size=kernel, stride=stride)
    block_map_y = block_map_y.view(N,C,kernel*kernel,block_num)
    mean_x = torch.mean(block_map_x, dim=2, keepdim=True)
    std_x = torch.std(block_map_x, dim=2, keepdim=True)
    mean_y = torch.mean(block_map_y, dim=2, keepdim=True)
    std_y = torch.std(block_map_y, dim=2, keepdim=True)
    cov_xy = (1./(kernel*kernel-1))*torch.sum((block_map_x-mean_x)*(block_map_y-mean_y),dim=2, keepdim=True)
    return torch.mean(4*cov_xy * mean_x * mean_y / (torch.multiply(mean_x**2+mean_y**2, std_x**2+std_y**2)+1.e-8), dim=3, keepdim=True)

def D_lambda(pred, ms):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    blur = 1.2
    channels = ms.size(1)
    low_pass = GaussianSmoothing(kernel_size=math.ceil(4 * blur) + 1, sigma=blur, channels=channels).to(ms.device)
    scale = pred.shape[2] // ms.shape[2]
    pred_low = low_pass(pred)
    pred_low_dec = decimation(pred_low, scale)
    C = ms.shape[1]
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C):
        for j in range(C):
            if i != j:
                band1 = pred_low_dec[:, [i], ...]  # abans estava així: band1 = img_fake[..., i]
                band2 = pred_low_dec[:, [j], ...]
                Q_fake.append(Q_index(band1, band2).item())
                # for real
                band1 = ms[:, [i], ...]
                band2 = ms[:, [j], ...]
                Q_lm.append(Q_index(band1, band2).item())
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm)).mean()
    return D_lambda_index


def D_s(pred, ms, pan):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    scale = pred.shape[2] // ms.shape[2]
    C_f = ms.shape[1]
    pan_lr = torch.nn.functional.interpolate(pan, scale_factor=1/scale, mode='bicubic') # Esto deberia ser P_tilde (falta convolucionar antes)
    # D_s
    Q_hr = []
    Q_lr = []
    q_hr=Q_index(pred,pan.repeat(1, C_f, 1, 1))
    q_lr=Q_index(ms,pan_lr.repeat(1, C_f, 1, 1))

    D_s_index = torch.abs(q_hr - q_lr).mean()
    if torch.isnan(D_s_index):
        print("problemos")
    return torch.clamp(D_s_index,0,1)

def D_f_lambda(pred, ms):
    blur = 1.2
    channels = ms.size(1)
    low_pass = GaussianSmoothing(kernel_size=math.ceil(4 * blur) + 1, sigma=blur, channels=channels).to(ms.device)
    scale = pred.shape[2] // ms.shape[2]
    pred_low = low_pass(pred)
    pred_low_dec = decimation(pred_low, scale)
    return 1 - q2n_index_own(pred_low_dec, ms)
def Q2N(pred, gt):

    return q2n_index_own(pred, gt)

def QNR(D_lmb, D_s, alpha=1, beta=1):
    """QNR - No reference IQA"""
    QNR_idx = (1 - D_lmb) ** alpha * (1 - D_s) ** beta
    return QNR_idx

def HQNR(D_F_lmb, Ds, alpha=1, beta=1):
    return (1-D_F_lmb)**alpha * (1-Ds)**beta

def q2n(I_GT, I_F, Q_blocks_size, Q_shift):
    H, W, C = I_GT.size()
    size2 = Q_blocks_size

    stepx = -(-H // Q_shift)
    stepy = -(-W // Q_shift)

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * Q_shift + Q_blocks_size - H
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - W

    if est1 != 0 or est2 != 0:
        refref = torch.zeros(H + est1, W + est2, C)
        fusfus = torch.zeros(H + est1, W + est2, C)

        refref[:H, :W, :] = I_GT
        refref[:, W:W + est2, :] = torch.flip(refref[:, W-est2:W,:], dims=[1])
        refref[H:H + est1, :] = torch.flip(refref[H-est1:H, :], dims=[0])

        fusfus[:H, :W, :] = I_F
        fusfus[:, W:W + est2, :] = torch.flip(fusfus[:, W-est2:W, :], dims=[1])
        fusfus[H:H + est1, :, :] = torch.flip(fusfus[H - est1:H, :, :], dims=[0])

        I_GT = refref
        I_F = fusfus

    if ((math.ceil(math.log2(C))) - math.log2(C)) != 0:
        Ndif = (2 ** (math.ceil(math.log2(C))) - C)

        dif = torch.zeros(I_GT.size(0), I_GT.size(1), Ndif)
        I_GT = torch.cat((I_GT, dif), dim=2)
        I_F = torch.cat((I_F, dif), dim=2)

    valori = torch.zeros(stepx, stepy, I_GT.size(2))

    for j in range(stepx):
        for i in range(stepy):
            start_x = (j * Q_shift)
            end_x = start_x + Q_blocks_size
            start_y = (i * Q_shift)
            end_y = start_y + size2
            o = onions_quality(I_GT[start_x:end_x, start_y:end_y, :], I_F[start_x:end_x, start_y:end_y, :], Q_blocks_size)
            valori[j, i, :] = o

    Q2n_index_map = torch.sqrt(torch.sum(valori ** 2, dim=2))
    Q2n_index = torch.mean(Q2n_index_map).item()
    print(Q2n_index_map)
    return Q2n_index, Q2n_index_map


def onions_quality(dat1, dat2, size1):
    dat1 = dat1.double()
    dat2 = dat2.double()
    dat2 = torch.cat((dat2[:, :, 0].unsqueeze(2), -dat2[:, :, 1:]), dim=2)
    _, _, C = dat1.shape
    size2 = size1

    # Block normalization
    for i in range(C):
        a1, s, t = norm_blocco(dat1[:, :, i].squeeze())
        dat1[:, :, i] = a1
        if s == 0:
            if i == 0:
                dat2[:, :, i] = dat2[:, :, i] - s + 1
            else:
                dat2[:, :, i] = -(-dat2[:, :, i] - s + 1)
        else:
            if i == 0:
                dat2[:, :, i] = ((dat2[:, :, i] - s) / t) + 1
            else:
                dat2[:, :, i] = -(((-dat2[:, :, i] - s) / t) + 1)

    m1 = torch.zeros(C)
    m2 = torch.zeros(C)
    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = torch.zeros(size1, size2)
    mod_q2 = torch.zeros(size1, size2)

    for i in range(C):
        m1[i] = torch.mean(dat1[:, :, i])
        m2[i] = torch.mean(dat2[:, :, i])
        mod_q1m += m1[i] ** 2
        mod_q2m += m2[i] ** 2
        mod_q1 += (dat1[:, :, i]) ** 2
        mod_q2 += (dat2[:, :, i]) ** 2

    mod_q1m = torch.sqrt(mod_q1m)
    mod_q2m = torch.sqrt(mod_q2m)
    mod_q1 = torch.sqrt(mod_q1)
    mod_q2 = torch.sqrt(mod_q2)

    termine2 = mod_q1m * mod_q2m
    termine4 = (mod_q1m ** 2) + (mod_q2m ** 2)
    int1 = (size1 * size2) / ((size1 * size2) - 1) * torch.mean(mod_q1 ** 2)
    int2 = (size1 * size2) / ((size1 * size2) - 1) * torch.mean(mod_q2 ** 2)
    termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2) - 1) * ((mod_q1m ** 2) + (mod_q2m ** 2))
    mean_bias = 2 * termine2 / termine4

    if termine3 == 0:
        q = torch.zeros(1, 1, C)
        q[:, :, C - 1] = mean_bias
    else:
        cbm = 2 / termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = torch.zeros(C)
        for i in range(C):
            qv[i] = (size1 * size2) / ((size1 * size2) - 1) * torch.mean(qu[:, :, i])
        q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
        q = q * mean_bias * cbm

    return q


def onion_mult2D(onion1, onion2):
    _, _, C = onion1.shape
    if C > 1:
        L = C // 2
        a = onion1[:, :, :L]
        b = onion1[:, :, L:]
        b = torch.cat((b[:, :, 0].unsqueeze(2), -b[:, :, 1:]), dim=2)
        c = onion2[:, :, :L]
        d = onion2[:, :, L:]
        d = torch.cat((d[:, :, 0].unsqueeze(2), -d[:, :, 1:]), dim=2)
        if C == 2:
            ris = torch.cat((a * c - d * b, a * d + c * b), dim=2)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, torch.cat((b[:, :, 0].unsqueeze(2), -b[:, :, 1:]), dim=2))
            ris3 = onion_mult2D(torch.cat((a[:, :, 0].unsqueeze(2), -a[:, :, 1:]), dim=2), d)
            ris4 = onion_mult2D(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat((aux1, aux2), dim=2)
    else:
        ris = onion1 * onion2
    return ris


def onion_mult(onion1, onion2):
    N = len(onion1)
    if N > 1:
        L = N // 2
        a = onion1[:L]
        b = onion1[L:]
        b = torch.cat((b[0].unsqueeze(0), -b[1:]))
        c = onion2[:L]
        d = onion2[L:]
        d = torch.cat((d[0].unsqueeze(0), -d[1:]))
        if N == 2:
            ris = torch.tensor([a * c - d * b, a * d + c * b])
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, torch.cat((b[0].unsqueeze(0), -b[1:])))
            ris3 = onion_mult(torch.cat((a[0].unsqueeze(0), -a[1:])), d)
            ris4 = onion_mult(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat((aux1, aux2))
    else:
        ris = onion1 * onion2
    return ris


def norm_blocco(x):
    a = torch.mean(x)
    c = torch.std(x)
    if c == 0:
        c = torch.finfo(x.dtype).eps  # Machine epsilon for the data type
    y = ((x - a) / c) + 1
    return y, a, c


def q2n_index(ref, pred, Q_blocks_size=10, Q_shift=5):
    ref = ref.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)
    N = ref.size(0)
    value = 0
    for i in range(N):
        Q2n_index, Q2n_index_map = q2n(ref[i], pred[i], Q_blocks_size, Q_shift)
        value += Q2n_index
    return value
def onion_mult2D_own(onion1, onion2):

    C = onion1.size(-1)

    if C > 1:
        L = C // 2
        a = onion1[:, :, :, :L]
        b = onion1[:, :, :,L:]
        # if (L ==1):
        #     print('in')
        #     print(a.size())
        #     print(b.size())
        #     print('out')
        b = torch.cat((b[:, :, :, [0]], -b[:, :, :,1:]), dim=3)
        c = onion2[:, :, :, :L]
        d = onion2[:, :, :,L:]
        d = torch.cat((d[:, :, :, [0]], -d[:, :, :,1:]), dim=3)
        # print("C: ", C)
        # print("L: ", L)
        if C == 2:
            # print(a.size())
            # print(b.size())
            # print(c.size())
            # print(d.size())
            ris = torch.cat((a * c - d * b, a * d + c * b), dim=3)
        else:
            ris1 = onion_mult2D_own(a, c)
            ris2 = onion_mult2D_own(d, torch.cat((b[:, :, :,  [0]], -b[:, :, :,  1:]), dim=3))
            ris3 = onion_mult2D_own(torch.cat((a[:, :, :,  [0]], -a[:, :, :,  1:]), dim=3), d)
            ris4 = onion_mult2D_own(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat((aux1, aux2), dim=3)
    else:
        ris = onion1 * onion2
    return ris
def onion_mult_own(onion1, onion2):
    C = onion1.size(-1)

    if C > 1:
        L = C // 2
        a = onion1[:, :, :L]
        b = onion1[:, :, L:]
        b = torch.cat((b[:, :, [0]], -b[:, :, 1:]), dim=2)
        c = onion2[:, :, :L]
        d = onion2[:, :, L:]
        d = torch.cat((d[:, :, [0]], -d[:, :, 1:]), dim=2)
        if C == 2:
            # print("uep")
            # print(a.size())
            ris = torch.cat((a * c - d * b, a * d + c * b),dim=2)
        else:
            ris1 = onion_mult_own(a, c)
            ris2 = onion_mult_own(d, torch.cat((b[:, :, [0]], -b[:, :, 1:]), dim=2))
            ris3 = onion_mult_own(torch.cat((a[ :, :, [0]], -a[:, :, 1:]), dim=2), d)
            ris4 = onion_mult_own(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat((aux1, aux2), dim=2)
    else:
        ris = onion1 * onion2
    return ris
def norm_blocco_own(x):
    a = torch.mean(x, dim=2, keepdim=True)
    c = torch.std(x, dim=2, keepdim=True)
    y = ((x - a) / c) + 1
    return y, a, c
def onions_quality_own(dat1, dat2, size1):
    dat1 = dat1.double()
    dat2 = dat2.double()
    dat2 = torch.cat((dat2[:, :, :, [0]], -dat2[:, : , :, 1:]), dim=3)
    L, N, K2, C = dat1.size()
    size2 = size1
    # Block normalization
    a1, s, t = norm_blocco_own(dat1)



    dat1 = a1
    dat2[:, :, :, 0] = torch.where(s[:, :, :, 0] == 0, dat2[:, :, :, 0] - s[:, :, :, 0] + 1, ((dat2[:, :, :, 0] - s[:, :, :, 0]) / t[:, :, :, 0]) + 1)
    dat2[:, :, :, 1:] = torch.where(s[:,:,:,1:] == 0, -(-dat2[:, :, :, 1:] - s[:, :, :, 1:] + 1), -(((-dat2[:, :, :, 1:] - s[:, :, :, 1:]) / t[:, :, :, 1:]) + 1))
    dat2 = torch.where(torch.isnan(dat2), 0.,dat2)
    dat1 = torch.where(torch.isnan(dat1), 0.,dat1)



    m1 = torch.mean(dat1, dim=2)
    m2 = torch.mean(dat2, dim=2)
    mod_q1m = torch.sum(m1 ** 2, dim=2)
    mod_q2m = torch.sum(m2 ** 2, dim=2)
    mod_q1 = torch.sum(dat1 ** 2, dim=3)
    mod_q2 = torch.sum(dat2 ** 2, dim=3)

    mod_q1m = torch.sqrt(mod_q1m)
    mod_q2m = torch.sqrt(mod_q2m)
    mod_q1 = torch.sqrt(mod_q1)
    mod_q2 = torch.sqrt(mod_q2)

    termine2 = mod_q1m * mod_q2m
    termine4 = (mod_q1m ** 2) + (mod_q2m ** 2)
    int1 = (size1 * size2) / ((size1 * size2) - 1) * torch.mean(mod_q1 ** 2, dim=2)
    int2 = (size1 * size2) / ((size1 * size2) - 1) * torch.mean(mod_q2 ** 2, dim=2)
    termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2) - 1) * ((mod_q1m ** 2) + (mod_q2m ** 2))
    mean_bias = 2 * termine2 / termine4
    mean_bias = mean_bias.view(L, 1, 1)
    # if termine3 == 0:
    #     q = torch.zeros(1, 1, C)
    #     q[:, :, C - 1] = mean_bias
    cbm = 2 / termine3
    cbm = cbm.view(L, 1, 1)
    qu = onion_mult2D_own(dat1, dat2)

    qm = onion_mult_own(m1, m2)
    qv = (size1 * size2) / ((size1 * size2) - 1) * torch.mean(qu, dim=2)
    q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
    q = q * mean_bias
    q = torch.where(termine3.view(L,1,1) == 0, mean_bias, q * cbm)
    return q
def q2n_own(I_GT, I_F, Q_blocks_size, Q_shift):
    N, C, H, W = I_GT.size()

    if ((math.ceil(math.log2(C))) - math.log2(C)) != 0:
        C_hyper = (2 ** (math.ceil(math.log2(C))))
    else:
        C_hyper = C

    I_GT_aux = torch.zeros(N, C_hyper, H, W).to(I_GT.device)
    I_F_aux = torch.zeros(N, C_hyper, H, W).to(I_GT.device)
    I_GT_aux[:, :C, :, :] = I_GT
    I_F_aux[:, :C, :, :] = I_F
    map_gt = create_block_map(I_GT_aux, kernel_size=Q_blocks_size, stride=Q_shift)
    num_blocks = map_gt.size(-1)
    map_gt.size()
    map_gt = map_gt.view(N,C_hyper, Q_blocks_size*Q_blocks_size, num_blocks).permute(3, 0, 2, 1)
    map_f = create_block_map(I_F_aux, kernel_size=Q_blocks_size, stride=Q_shift)
    map_f = map_f.view(N, C_hyper, Q_blocks_size * Q_blocks_size, num_blocks).permute(3, 0, 2, 1) # (num_blocks, 1, Q_blocks_size*Q_blocks_size, C)

    valori = onions_quality_own(map_gt, map_f, Q_blocks_size)


    Q2n_index_map = torch.sqrt(torch.sum(valori ** 2, dim=2))
    Q2n_index = torch.mean(Q2n_index_map, dim=0).item()
    return Q2n_index, Q2n_index_map

def q2n_index_own(ref, pred, Q_blocks_size=10, Q_shift=5):
    Q2n_index, Q2n_index_map = q2n_own(ref, pred, Q_blocks_size, Q_shift)
    return Q2n_index

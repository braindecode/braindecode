# Authors: Martin Wimpff <martin.wimpff@iss.uni-stuttgart.de>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import math

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype: torch.dtype,
                           device: torch.device) -> torch.Tensor:
    """
    Generates a 1-dimensional Gaussian kernel based on the specified kernel
    size and standard deviation (sigma).

    This kernel is useful for Gaussian smoothing or filtering operations in
    image processing. The function calculates a range limit to ensure the kernel
    effectively covers the Gaussian distribution. It generates a tensor of
    specified size and type, filled with values distributed according to a
    Gaussian curve, normalized using a softmax function
    to ensure all weights sum to 1.

    Parameters
    ----------
    kernel_size: int
    sigma: float
    dtype: torch.dtype
    device: torch.device

    Returns
    -------
    kernel1d: torch.Tensor

    Notes
    -----
    Code copied and modified from TorchVision:
    https://github.com/pytorch/vision/blob/f21c6bd8ca5cd057fc3d1e0dc5ea8f6403dc144f/torchvision/transforms/v2/functional/_misc.py#L84-L88
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.
        * Neither the name of the NumPy Developers nor the names of any
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0) * sigma)
    x = torch.linspace(-lim, lim, steps=kernel_size, dtype=dtype, device=device)
    kernel1d = torch.softmax(x.pow_(2).neg_(), dim=0)
    return kernel1d


class SqueezeAndExcitation(nn.Module):
    """
    Parameters
    ----------
    in_channels : int, number of input feature channels
    reduction_rate : int, reduction ratio of the fully-connected layers
    bias: bool, default=False
    
    References
    ----------
    .. [Hu2018] Hu, J., Albanie, S., Sun, G., Wu, E., 2018.
    Squeeze-and-Excitation Networks. CVPR 2018.
    """

    def __init__(self, in_channels: int, reduction_rate: int, bias: int = False):
        super(SqueezeAndExcitation, self).__init__()
        sq_channels = int(in_channels // reduction_rate)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=in_channels,
                             out_channels=sq_channels,
                             kernel_size=1,
                             bias=bias)
        self.nonlinearity = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=reduction_rate,
                             out_channels=sq_channels,
                             kernel_size=1,
                             bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.gap(x)
        scale = self.fc1(scale)
        scale = self.nonlinearity(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return scale * x


class GSoP(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, reduction_rate: int, bias: bool = True):
        super(GSoP, self).__init__()
        sq_channels = int(in_channels // reduction_rate)
        self.pw_conv1 = nn.Conv2d(in_channels, sq_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(sq_channels)
        self.rw_conv = nn.Conv2d(sq_channels, sq_channels * 4, (sq_channels, 1),
                                 groups=sq_channels, bias=bias)
        self.pw_conv2 = nn.Conv2d(sq_channels * 4, in_channels, 1, bias=bias)

    def forward(self, x):
        scale = self.pw_conv1(x).squeeze(-2)  # b x c x t
        scale_zero_mean = scale - scale.mean(-1, keepdim=True)
        t = scale_zero_mean.shape[-1]
        cov = torch.bmm(scale_zero_mean, scale_zero_mean.transpose(1, 2)) / (t - 1)
        cov = cov.unsqueeze(-1)  # b x c x c x 1
        cov = self.bn(cov)
        scale = self.rw_conv(cov)  # b x c x 1 x 1
        scale = self.pw_conv2(scale)
        return scale * x


class FCA(torch.nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels, seq_len: int = 62, reduction_rate: int = 4,
                 freq_idx: int = 0):
        super(FCA, self).__init__()
        mapper_y = [freq_idx]
        assert in_channels % len(mapper_y) == 0

        self.weight = nn.Parameter(self.get_dct_filter(seq_len, mapper_y, in_channels),
                                   requires_grad=False)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_rate, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_rate, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = x.squeeze(-2) * self.weight
        scale = torch.sum(scale, dim=-1)
        scale = rearrange(self.fc(scale), "b c -> b c 1 1")
        return x * scale.expand_as(x)

    @staticmethod
    def get_dct_filter(seq_len: int, mapper_y: list, in_channels: int):
        dct_filter = torch.zeros(in_channels, seq_len)

        c_part = in_channels // len(mapper_y)

        for i, v_y in enumerate(mapper_y):
            for t_y in range(seq_len):
                filter = math.cos(math.pi * v_y * (t_y + 0.5) / seq_len) / math.sqrt(
                    seq_len)
                filter = filter * math.sqrt(2) if v_y != 0 else filter
                dct_filter[i * c_part: (i + 1) * c_part, t_y] = filter
        return dct_filter


class EncNet(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, n_codewords: int):
        super(EncNet, self).__init__()
        self.n_codewords = n_codewords
        self.codewords = nn.Parameter(torch.empty(n_codewords, in_channels))
        self.smoothing = nn.Parameter(torch.empty(n_codewords))
        std = 1 / ((n_codewords * in_channels) ** (1 / 2))
        nn.init.uniform_(self.codewords.data, -std, std)
        nn.init.uniform_(self.smoothing, -1, 0)
        self.bn = nn.BatchNorm1d(n_codewords)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        b, c, _, seq = x.shape
        # b x c x 1 x t -> b x t x k x c
        x_ = rearrange(x, "b c 1 seq -> b seq 1 c").expand(b, seq, self.n_codewords, c)
        cw_ = self.codewords.unsqueeze(0).unsqueeze(0)  # 1 x 1 x k x c
        a = self.smoothing.unsqueeze(0).unsqueeze(0) * (x_ - cw_).pow(2).sum(3)
        a = torch.softmax(a, dim=2)  # b x t x k

        # aggregate
        e = (a.unsqueeze(3) * (x_ - cw_)).sum(1)  # b x k x c
        e_norm = torch.relu(self.bn(e)).mean(1)  # b x c

        scale = torch.sigmoid(self.fc(e_norm))
        return x * scale.unsqueeze(2).unsqueeze(3)


class ECA(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, kernel_size: int):
        super(ECA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        assert kernel_size % 2 == 1, "kernel size must be odd for same padding"
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2,
                              bias=False)

    def forward(self, x):
        scale = self.gap(x)
        scale = rearrange(scale, "b c 1 1 -> b 1 c")
        scale = self.conv(scale)
        scale = torch.sigmoid(rearrange(scale, "b 1 c -> b c 1 1"))
        return x * scale


class GatherExcite(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, seq_len: int = 62, extra_params: bool = False,
                 use_mlp: bool = False, reduction_rate: int = 4):
        super(GatherExcite, self).__init__()
        if extra_params:
            self.gather = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, seq_len), groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.gather = nn.AdaptiveAvgPool2d(1)


        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channels, int(in_channels // reduction_rate), 1,
                          bias=False),
                nn.ReLU(),
                nn.Conv2d(int(in_channels // reduction_rate), in_channels, 1,
                          bias=False),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, x):
        scale = self.gather(x)
        scale = torch.sigmoid(self.mlp(scale))
        return scale * x


class GCT(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x, eps: float = 1e-5):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + eps).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + eps).pow(0.5)
        gate = 1.0 + torch.tanh(embedding * norm + self.beta)
        return x * gate


class SRM(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, use_mlp: bool = False, reduction_rate: int = 4,
                 bias: bool = False):
        super(SRM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        if use_mlp:
            self.style_integration = nn.Sequential(
                Rearrange("b c n_metrics -> b (c n_metrics)"),
                nn.Linear(in_channels * 2, in_channels * 2 // reduction_rate,
                          bias=bias),
                nn.ReLU(),
                nn.Linear(in_channels * 2 // reduction_rate, in_channels, bias=bias),
                Rearrange("b c -> b c 1")
            )
        else:
            self.style_integration = nn.Conv1d(in_channels, in_channels, 2,
                                               groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        mu = self.gap(x).squeeze(-1)  # b x c x 1
        std = x.std(dim=(-2, -1), keepdim=True).squeeze(-1)  # b x c x 1
        t = torch.cat([mu, std], dim=2)  # b x c x 2
        z = self.style_integration(t)  # b x c x 1
        z = self.bn(z)
        scale = nn.functional.sigmoid(z).unsqueeze(-1)
        return scale * x


class CBAM(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, reduction_rate: int, kernel_size: int):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, bias=False)
        )
        assert kernel_size % 2 == 1, "kernel size must be odd for same padding"
        self.conv = nn.Conv2d(2, 1,
                              (1, kernel_size), padding=(0, kernel_size // 2))

    def forward(self, x):
        channel_attention = torch.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * channel_attention
        spat_input = torch.cat(
            [torch.mean(x, dim=1, keepdim=True),
             torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_attention = torch.sigmoid(self.conv(spat_input))
        return x * spatial_attention


class CAT(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, reduction_rate: int, kernel_size: int,
                 bias=False):
        super(CAT, self).__init__()
        self.gauss_filter = nn.Conv2d(1, 1,
                                      (1, 5), padding=(0, 2), bias=False)
        self.gauss_filter.weight = nn.Parameter(
            _get_gaussian_kernel1d(5, 1.0)[None, None, None, :],
            requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, bias=bias)
        )
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size),
                              padding=(0, kernel_size // 2), bias=bias)

        self.c_alpha = nn.Parameter(torch.zeros(1))
        self.c_beta = nn.Parameter(torch.zeros(1))
        self.c_gamma = nn.Parameter(torch.zeros(1))
        self.s_alpha = nn.Parameter(torch.zeros(1))
        self.s_beta = nn.Parameter(torch.zeros(1))
        self.s_gamma = nn.Parameter(torch.zeros(1))
        self.c_w = nn.Parameter(torch.zeros(1))
        self.s_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_blurred = self.gauss_filter(x.transpose(1, 2)).transpose(1, 2)

        c_gap = self.mlp(x.mean(dim=(-2, -1), keepdim=True))
        c_gmp = self.mlp(torch.amax(x_blurred, dim=(-2, -1), keepdim=True))
        pi = torch.softmax(x, dim=-1)
        c_gep = -1 * (pi * torch.log(pi)).sum(dim=(-2, -1), keepdim=True)
        c_gep_min = torch.amin(c_gep, dim=(-3, -2, -1), keepdim=True)
        c_gep_max = torch.amax(c_gep, dim=(-3, -2, -1), keepdim=True)
        c_gep = self.mlp((c_gep - c_gep_min) / (c_gep_max - c_gep_min))
        channel_score = torch.sigmoid(
            c_gap * self.c_alpha + c_gmp * self.c_beta + c_gep * self.c_gamma)
        channel_score = channel_score.expand(b, c, h, w)

        s_gap = x.mean(dim=1, keepdim=True)
        s_gmp = torch.amax(x_blurred, dim=(-2, -1), keepdim=True)
        pi = torch.softmax(x, dim=1)
        s_gep = -1 * (pi * torch.log(pi)).sum(dim=1, keepdim=True)
        s_gep_min = torch.amin(s_gep, dim=(-2, -1), keepdim=True)
        s_gep_max = torch.amax(s_gep, dim=(-2, -1), keepdim=True)
        s_gep = (s_gep - s_gep_min) / (s_gep_max - s_gep_min)
        spatial_score = -s_gap * self.s_alpha + s_gmp * self.s_beta + s_gep * self.s_gamma
        spatial_score = torch.sigmoid(self.conv(spatial_score)).expand(b, c, h, w)

        c_w = torch.exp(self.c_w) / (torch.exp(self.c_w) + torch.exp(self.s_w))
        s_w = torch.exp(self.s_w) / (torch.exp(self.c_w) + torch.exp(self.s_w))

        scale = channel_score * c_w + spatial_score * s_w
        return scale * x


class CATLite(nn.Module):
    """
    XXX. TODO: Add description.
    """
    def __init__(self, in_channels: int, reduction_rate: int, bias: bool = True):
        super(CATLite, self).__init__()
        self.gauss_filter = nn.Conv2d(1, 1,
                                      (1, 5), padding=(0, 2), bias=False)
        self.gauss_filter.weight = nn.Parameter(
            _get_gaussian_kernel1d(5, 1.0)[None, None, None, :],
            requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // reduction_rate),
                      1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(int(in_channels // reduction_rate), in_channels,
                      1, bias=bias)
        )

        self.c_alpha = nn.Parameter(torch.zeros(1))
        self.c_beta = nn.Parameter(torch.zeros(1))
        self.c_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_blurred = self.gauss_filter(x.transpose(1, 2)).transpose(1, 2)

        c_gap = self.mlp(x.mean(dim=(-2, -1), keepdim=True))
        c_gmp = self.mlp(torch.amax(x_blurred, dim=(-2, -1), keepdim=True))
        pi = torch.softmax(x, dim=-1)
        c_gep = -1 * (pi * torch.log(pi)).sum(dim=(-2, -1), keepdim=True)
        c_gep_min = torch.amin(c_gep, dim=(-3, -2, -1), keepdim=True)
        c_gep_max = torch.amax(c_gep, dim=(-3, -2, -1), keepdim=True)
        c_gep = self.mlp((c_gep - c_gep_min) / (c_gep_max - c_gep_min))
        channel_score = torch.sigmoid(
            c_gap * self.c_alpha + c_gmp * self.c_beta + c_gep * self.c_gamma)
        channel_score = channel_score.expand(b, c, h, w)

        return channel_score * x


def get_attention_block(attention_mode: str, ch_dim: int = 16,
                        reduction_rate: int = 4,
                        use_mlp: bool = False, seq_len: int = None,
                        freq_idx: int = 0,
                        n_codewords: int = 4, kernel_size: int = 9,
                        extra_params: bool = False):
    if attention_mode == "se":
        return SqueezeAndExcitation(ch_dim, reduction_rate)
    # improving the squeeze module
    elif attention_mode == "gsop":
        return GSoP(ch_dim, reduction_rate)
    elif attention_mode == "fca":
        assert seq_len is not None
        return FCA(ch_dim, seq_len, reduction_rate, freq_idx=freq_idx)
    elif attention_mode == "encnet":
        return EncNet(ch_dim, n_codewords=n_codewords)
    # improving the excitation module
    elif attention_mode == "eca":
        return ECA(ch_dim, kernel_size=kernel_size)
    # improving the squeeze and the excitation module
    elif attention_mode == "ge":
        return GatherExcite(ch_dim, seq_len=seq_len, extra_params=extra_params,
                            use_mlp=use_mlp, reduction_rate=reduction_rate)
    elif attention_mode == "gct":
        return GCT(ch_dim)
    elif attention_mode == "srm":
        return SRM(ch_dim, use_mlp=use_mlp, reduction_rate=reduction_rate)
    # temporal and channel attention
    elif attention_mode == "cbam":
        return CBAM(ch_dim, reduction_rate, kernel_size=kernel_size)
    elif attention_mode == "cat":
        return CAT(ch_dim, reduction_rate, kernel_size)
    elif attention_mode == "catlite":
        return CATLite(ch_dim, reduction_rate=reduction_rate)
    else:
        raise NotImplementedError

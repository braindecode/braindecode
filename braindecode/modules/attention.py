"""
Attention modules used in the AttentionBaseNet from Martin Wimpff (2023).

Here, we implement some popular attention modules that can be used in the
AttentionBaseNet class.

"""

# Authors: Martin Wimpff <martin.wimpff@iss.uni-stuttgart.de>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.functional import _get_gaussian_kernel1d


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation Networks from [Hu2018]_.

    Parameters
    ----------
    in_channels : int,
        number of input feature channels.
    reduction_rate : int,
        reduction ratio of the fully-connected layers.
    bias: bool, default=False
        if True, adds a learnable bias will be used in the convolution.

    References
    ----------
    .. [Hu2018] Hu, J., Albanie, S., Sun, G., Wu, E., 2018.
    Squeeze-and-Excitation Networks. CVPR 2018.
    """

    def __init__(self, in_channels: int, reduction_rate: int, bias: bool = False):
        super(SqueezeAndExcitation, self).__init__()
        sq_channels = int(in_channels // reduction_rate)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels=in_channels, out_channels=sq_channels, kernel_size=1, bias=bias
        )
        self.nonlinearity = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels=reduction_rate,
            out_channels=in_channels,
            kernel_size=1,
            bias=bias,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply the Squeeze-and-Excitation block to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        scale*x: Pytorch.Tensor
        """
        scale = self.gap(x)
        scale = self.fc1(scale)
        scale = self.nonlinearity(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return scale * x


class GSoP(nn.Module):
    """
    Global Second-order Pooling Convolutional Networks from [Gao2018]_.

    Parameters
    ----------
    in_channels : int,
        number of input feature channels
    reduction_rate : int,
        reduction ratio of the fully-connected layers
    bias: bool, default=False
        if True, adds a learnable bias will be used in the convolution.

    References
    ----------
    .. [Gao2018] Gao, Z., Jiangtao, X., Wang, Q., Li, P., 2018.
    Global Second-order Pooling Convolutional Networks. CVPR 2018.
    """

    def __init__(self, in_channels: int, reduction_rate: int, bias: bool = True):
        super(GSoP, self).__init__()
        sq_channels = int(in_channels // reduction_rate)
        self.pw_conv1 = nn.Conv2d(in_channels, sq_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(sq_channels)
        self.rw_conv = nn.Conv2d(
            sq_channels,
            sq_channels * 4,
            (sq_channels, 1),
            groups=sq_channels,
            bias=bias,
        )
        self.pw_conv2 = nn.Conv2d(sq_channels * 4, in_channels, 1, bias=bias)

    def forward(self, x):
        """
        Apply the Global Second-order Pooling Convolutional Networks block.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        scale = self.pw_conv1(x).squeeze(-2)  # b x c x t
        scale_zero_mean = scale - scale.mean(-1, keepdim=True)
        t = scale_zero_mean.shape[-1]
        cov = torch.bmm(scale_zero_mean, scale_zero_mean.transpose(1, 2)) / (t - 1)
        cov = cov.unsqueeze(-1)  # b x c x c x 1
        cov = self.bn(cov)
        scale = self.rw_conv(cov)  # b x c x 1 x 1
        scale = self.pw_conv2(scale)
        return scale * x


class FCA(nn.Module):
    """
    Frequency Channel Attention Networks from [Qin2021]_.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels
    seq_len : int
        Sequence length along temporal dimension, default=62
    reduction_rate : int, default=4
        Reduction ratio of the fully-connected layers.

    References
    ----------
    .. [Qin2021] Qin, Z., Zhang, P., Wu, F., Li, X., 2021.
    FcaNet: Frequency Channel Attention Networks. ICCV 2021.
    """

    def __init__(
        self, in_channels, seq_len: int = 62, reduction_rate: int = 4, freq_idx: int = 0
    ):
        super(FCA, self).__init__()
        mapper_y = [freq_idx]
        if in_channels % len(mapper_y) != 0:
            raise ValueError("in_channels must be divisible by number of DCT filters")

        self.weight = nn.Parameter(
            self.get_dct_filter(seq_len, mapper_y, in_channels), requires_grad=False
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_rate, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_rate, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Apply the Frequency Channel Attention Networks block to the input.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        scale = x.squeeze(-2) * self.weight
        scale = torch.sum(scale, dim=-1)
        scale = rearrange(self.fc(scale), "b c -> b c 1 1")
        return x * scale.expand_as(x)

    @staticmethod
    def get_dct_filter(seq_len: int, mapper_y: list, in_channels: int):
        """
        Util function to get the DCT filter.

        Parameters
        ----------
        seq_len: int
            Size of the sequence
        mapper_y:
            List of frequencies
        in_channels:
            Number of input channels.

        Returns
        -------
        torch.Tensor
        """
        dct_filter = torch.zeros(in_channels, seq_len)

        c_part = in_channels // len(mapper_y)

        for i, v_y in enumerate(mapper_y):
            for t_y in range(seq_len):
                filter = math.cos(math.pi * v_y * (t_y + 0.5) / seq_len) / math.sqrt(
                    seq_len
                )
                filter = filter * math.sqrt(2) if v_y != 0 else filter
                dct_filter[i * c_part : (i + 1) * c_part, t_y] = filter
        return dct_filter


class EncNet(nn.Module):
    """
    Context Encoding for Semantic Segmentation from [Zhang2018]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    n_codewords : int
        number of codewords

    References
    ----------
    .. [Zhang2018] Zhang, H. et al. 2018.
    Context Encoding for Semantic Segmentation. CVPR 2018.
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
        """
        Apply attention from the Context Encoding for Semantic Segmentation.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        b, c, _, seq = x.shape
        # b x c x 1 x t -> b x t x k x c
        x_ = rearrange(x, pattern="b c 1 seq -> b seq 1 c")
        x_ = x_.expand(b, seq, self.n_codewords, c)
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
    Efficient Channel Attention [Wang2021]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    kernel_size : int
        kernel size of convolutional layer, determines degree of channel
        interaction, must be odd.

    References
    ----------
    .. [Wang2021] Wang, Q. et al., 2021. ECA-Net: Efficient Channel Attention
    for Deep Convolutional Neural Networks. CVPR 2021.
    """

    def __init__(self, in_channels: int, kernel_size: int):
        super(ECA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        if kernel_size % 2 != 1:
            raise ValueError("kernel size must be odd for same padding")
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x):
        """
        Apply the Efficient Channel Attention block to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        scale = self.gap(x)
        scale = rearrange(scale, "b c 1 1 -> b 1 c")
        scale = self.conv(scale)
        scale = torch.sigmoid(rearrange(scale, "b 1 c -> b c 1 1"))
        return x * scale


class GatherExcite(nn.Module):
    """
    Gather-Excite Networks from [Hu2018b]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    seq_len : int, default=62
        sequence length along temporal dimension
    extra_params : bool, default=False
        whether to use a convolutional layer as a gather module
    use_mlp : bool, default=False
        whether to use an excite block with fully-connected layers
    reduction_rate : int, default=4
        reduction ratio of the excite block (if used)

    References
    ----------
    .. [Hu2018b] Hu, J., Albanie, S., Sun, G., Vedaldi, A., 2018.
    Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks.
    NeurIPS 2018.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int = 62,
        extra_params: bool = False,
        use_mlp: bool = False,
        reduction_rate: int = 4,
    ):
        super(GatherExcite, self).__init__()
        if extra_params:
            self.gather = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    (1, seq_len),
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
            )
        else:
            self.gather = nn.AdaptiveAvgPool2d(1)

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Conv2d(
                    in_channels, int(in_channels // reduction_rate), 1, bias=False
                ),
                nn.ReLU(),
                nn.Conv2d(
                    int(in_channels // reduction_rate), in_channels, 1, bias=False
                ),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, x):
        """
        Apply the Gather-Excite Networks block to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        scale = self.gather(x)
        scale = torch.sigmoid(self.mlp(scale))
        return scale * x


class GCT(nn.Module):
    """
    Gated Channel Transformation from [Yang2020]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels

    References
    ----------
    .. [Yang2020] Yang, Z. Linchao, Z., Wu, Y., Yang, Y., 2020.
    Gated Channel Transformation for Visual Recognition. CVPR 2020.
    """

    def __init__(self, in_channels: int):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x, eps: float = 1e-5):
        """
        Apply the Gated Channel Transformation block to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor
        eps: float, default=1e-5

        Returns
        -------
        Pytorch.Tensor
            the original tensor x multiplied by the gate.
        """
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + eps).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + eps).pow(0.5)
        gate = 1.0 + torch.tanh(embedding * norm + self.beta)
        return x * gate


class SRM(nn.Module):
    """
    Attention module from [Lee2019]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    use_mlp : bool, default=False
        whether to use fully-connected layers instead of a convolutional layer,
    reduction_rate : int, default=4
        reduction ratio of the fully-connected layers (if used),

    References
    ----------
    .. [Lee2019] Lee, H., Kim, H., Nam, H., 2019. SRM: A Style-based
    Recalibration Module for Convolutional Neural Networks. ICCV 2019.
    """

    def __init__(
        self,
        in_channels: int,
        use_mlp: bool = False,
        reduction_rate: int = 4,
        bias: bool = False,
    ):
        super(SRM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        if use_mlp:
            self.style_integration = nn.Sequential(
                Rearrange("b c n_metrics -> b (c n_metrics)"),
                nn.Linear(
                    in_channels * 2, in_channels * 2 // reduction_rate, bias=bias
                ),
                nn.ReLU(),
                nn.Linear(in_channels * 2 // reduction_rate, in_channels, bias=bias),
                Rearrange("b c -> b c 1"),
            )
        else:
            self.style_integration = nn.Conv1d(
                in_channels, in_channels, 2, groups=in_channels, bias=bias
            )
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        """
        Apply the Style-based Recalibration Module to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        mu = self.gap(x).squeeze(-1)  # b x c x 1
        std = x.std(dim=(-2, -1), keepdim=True).squeeze(-1)  # b x c x 1
        t = torch.cat([mu, std], dim=2)  # b x c x 2
        z = self.style_integration(t)  # b x c x 1
        z = self.bn(z)
        scale = nn.functional.sigmoid(z).unsqueeze(-1)
        return scale * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module from [Woo2018]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    reduction_rate : int
        reduction ratio of the fully-connected layers
    kernel_size : int
        kernel size of the convolutional layer

    References
    ----------
    .. [Woo2018] Woo, S., Park, J., Lee, J., Kweon, I., 2018.
    CBAM: Convolutional Block Attention Module. ECCV 2018.
    """

    def __init__(self, in_channels: int, reduction_rate: int, kernel_size: int):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, bias=False),
        )
        if kernel_size % 2 != 1:
            raise ValueError("kernel size must be odd for same padding")
        self.conv = nn.Conv2d(2, 1, (1, kernel_size), padding=(0, kernel_size // 2))

    def forward(self, x):
        """
        Apply the Convolutional Block Attention Module to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
        channel_attention = torch.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )
        x = x * channel_attention
        spat_input = torch.cat(
            [torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]],
            dim=1,
        )
        spatial_attention = torch.sigmoid(self.conv(spat_input))
        return x * spatial_attention


class CAT(nn.Module):
    """
    Attention Mechanism from [Wu2023]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    reduction_rate : int
        reduction ratio of the fully-connected layers
    kernel_size : int
        kernel size of the convolutional layer
    bias : bool, default=False
        if True, adds a learnable bias will be used in the convolution,

    References
    ----------
    .. [Wu2023] Wu, Z. et al., 2023
        CAT: Learning to Collaborate Channel and Spatial Attention from
        Multi-Information Fusion. IET Computer Vision 2023.
    """

    def __init__(
        self, in_channels: int, reduction_rate: int, kernel_size: int, bias=False
    ):
        super(CAT, self).__init__()
        self.gauss_filter = nn.Conv2d(1, 1, (1, 5), padding=(0, 2), bias=False)
        self.gauss_filter.weight = nn.Parameter(
            _get_gaussian_kernel1d(5, 1.0)[None, None, None, :], requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, bias=bias),
        )
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=bias,
        )

        self.c_alpha = nn.Parameter(torch.zeros(1))
        self.c_beta = nn.Parameter(torch.zeros(1))
        self.c_gamma = nn.Parameter(torch.zeros(1))
        self.s_alpha = nn.Parameter(torch.zeros(1))
        self.s_beta = nn.Parameter(torch.zeros(1))
        self.s_gamma = nn.Parameter(torch.zeros(1))
        self.c_w = nn.Parameter(torch.zeros(1))
        self.s_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Apply the CAT block to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
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
            c_gap * self.c_alpha + c_gmp * self.c_beta + c_gep * self.c_gamma
        )
        channel_score = channel_score.expand(b, c, h, w)

        s_gap = x.mean(dim=1, keepdim=True)
        s_gmp = torch.amax(x_blurred, dim=(-2, -1), keepdim=True)
        pi = torch.softmax(x, dim=1)
        s_gep = -1 * (pi * torch.log(pi)).sum(dim=1, keepdim=True)
        s_gep_min = torch.amin(s_gep, dim=(-2, -1), keepdim=True)
        s_gep_max = torch.amax(s_gep, dim=(-2, -1), keepdim=True)
        s_gep = (s_gep - s_gep_min) / (s_gep_max - s_gep_min)
        spatial_score = (
            -s_gap * self.s_alpha + s_gmp * self.s_beta + s_gep * self.s_gamma
        )
        spatial_score = torch.sigmoid(self.conv(spatial_score)).expand(b, c, h, w)

        c_w = torch.exp(self.c_w) / (torch.exp(self.c_w) + torch.exp(self.s_w))
        s_w = torch.exp(self.s_w) / (torch.exp(self.c_w) + torch.exp(self.s_w))

        scale = channel_score * c_w + spatial_score * s_w
        return scale * x


class CATLite(nn.Module):
    """
    Modification of CAT without the convolutional layer from [Wu2023]_.

    Parameters
    ----------
    in_channels : int
        number of input feature channels
    reduction_rate : int
        reduction ratio of the fully-connected layers
    bias : bool, default=True
        if True, adds a learnable bias will be used in the convolution,

    References
    ----------
    .. [Wu2023] Wu, Z. et al., 2023 CAT: Learning to Collaborate Channel and
        Spatial Attention from Multi-Information Fusion. IET Computer Vision 2023.
    """

    def __init__(self, in_channels: int, reduction_rate: int, bias: bool = True):
        super(CATLite, self).__init__()
        self.gauss_filter = nn.Conv2d(1, 1, (1, 5), padding=(0, 2), bias=False)
        self.gauss_filter.weight = nn.Parameter(
            _get_gaussian_kernel1d(5, 1.0)[None, None, None, :], requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // reduction_rate), 1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(int(in_channels // reduction_rate), in_channels, 1, bias=bias),
        )

        self.c_alpha = nn.Parameter(torch.zeros(1))
        self.c_beta = nn.Parameter(torch.zeros(1))
        self.c_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Apply the CATLite block to the input tensor.

        Parameters
        ----------
        x: Pytorch.Tensor

        Returns
        -------
        Pytorch.Tensor
        """
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
            c_gap * self.c_alpha + c_gmp * self.c_beta + c_gep * self.c_gamma
        )
        channel_score = channel_score.expand(b, c, h, w)

        return channel_score * x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.rearrange_stack = Rearrange(
            "b n (h d) -> b h n d",
            h=num_heads,
        )
        self.rearrange_unstack = Rearrange(
            "b h n d -> b n (h d)",
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        queries = self.rearrange_stack(self.queries(x))
        keys = self.rearrange_stack(self.keys(x))
        values = self.rearrange_stack(self.values(x))
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = float("-inf")
            energy = energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = self.rearrange_unstack(out)
        out = self.projection(out)
        return out

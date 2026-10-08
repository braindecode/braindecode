"""TMSA-Net for motor-imagery EEG classification."""

# Authors: Qian Zhao <zhaoqian0120@qq.com>
#          Weina Zhu
#          (braindecode adaptation)
# License: MIT
# Adapted from https://github.com/Whit3Zhao/TMSA-Net (MIT).

from __future__ import annotations

import math

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock


class TMSANet(EEGModuleMixin, nn.Module, license="mit"):
    r"""TMSA-Net from Zhao and Zhu (2025) [tmsanet]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    TMSA-Net combines multi-scale temporal convolutions, a spatial
    convolution across EEG channels, and a Transformer whose attention sums a
    global-key branch and a local-key branch (keys from multi-scale 1D
    convolutions) for motor-imagery classification.

    `License <https://github.com/Whit3Zhao/TMSA-Net/blob/main/LICENSE>`_.

    As in the released code, the head width is ``embed_dim // num_heads``, so
    the default ``embed_dim=19`` with four heads projects queries, keys and
    values through 16 dimensions (19 -> 16 -> 19).

    Parameters
    ----------
    embed_dim : int, default=19
        Embedding width after the temporal/spatial feature extractor.
        The released source uses 19 for BCI Competition IV 2a,
        6 for BCI Competition IV 2b, and 10 for HGD.
    pool_size : int, default=50
        Kernel size of the temporal average pooling layer.
    pool_stride : int, default=15
        Stride of the temporal average pooling layer.
    num_heads : int, default=4
        Number of attention heads.
    fc_ratio : int, default=2
        Expansion ratio of the Transformer feed-forward block.
    depth : int, default=1
        Number of Transformer encoder blocks.
    drop_prob : float, default=0.5
        Dropout probability before the Transformer and after the local-key
        convolutions.
    att_drop_prob : float, default=0.5
        Dropout probability on the attention weights (0.7 for HGD in the
        released source).
    fc_drop_prob : float, default=0.5
        Dropout probability in the feed-forward block.
    activation : type[nn.Module], default=nn.GELU
        Activation after the spatial convolution and in the feed-forward block.

    Notes
    -----
    Ported from ``Whit3Zhao/TMSA-Net@c60882db35eeff860a5014df7b0f54dda6601c65``;
    the reference ``radix`` argument only multiplies the channel count and is
    not exposed.

    References
    ----------
    .. [tmsanet] Zhao, Q., Zhu, W. TMSA-Net: A novel attention mechanism
        for improved motor imagery EEG signal processing. Biomedical Signal
        Processing and Control 102, 107189 (2025).
        https://doi.org/10.1016/j.bspc.2024.107189
    """

    def __init__(
        self,
        # Braindecode signal arguments
        n_outputs=None,
        n_chans=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model-specific arguments
        embed_dim: int = 19,
        pool_size: int = 50,
        pool_stride: int = 15,
        num_heads: int = 4,
        fc_ratio: int = 2,
        depth: int = 1,
        drop_prob: float = 0.5,
        att_drop_prob: float = 0.5,
        fc_drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if embed_dim < num_heads:
            # embed_dim // num_heads would be 0: an attention without inputs.
            raise ValueError(
                f"embed_dim ({embed_dim}) must be at least num_heads ({num_heads})."
            )

        self.add_channel_dim = Rearrange("b c t -> b 1 c t")
        # Two temporal kernels (31 and 15 samples) summed, then one spatial
        # convolution over all channels.
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.n_chans, 1))
        self.activation = activation()
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.to_sequence = Rearrange("b d 1 t -> b d t")
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        self.dropout = nn.Dropout(drop_prob)
        self.transformer = nn.Sequential(
            Rearrange("b d n -> b n d"),
            *[
                _TMSABlock(
                    embed_dim,
                    num_heads,
                    fc_ratio,
                    drop_prob,
                    att_drop_prob,
                    fc_drop_prob,
                    activation,
                )
                for _ in range(depth)
            ],
        )
        # The reference flattens (embed_dim, tokens), embedding-major.
        self.flatten = Rearrange("b n d -> b (d n)")
        n_tokens = (self.n_times - pool_size) // pool_stride + 1
        self.final_layer = nn.Linear(embed_dim * n_tokens, self.n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_channel_dim(x)
        x = self.temp_conv1(x) + self.temp_conv2(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.to_sequence(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.flatten(x)
        return self.final_layer(x)


class _TMSAAttention(nn.Module):
    """Attention summing a local-key and a global-key branch over shared values."""

    def __init__(self, embed_dim, num_heads, drop_prob, att_drop_prob):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        inner_dim = self.head_dim * num_heads
        # Local keys: kernels 3 and 5 over tokens, concatenated on channels.
        self.to_channels = Rearrange("b n d -> b d n")
        self.local_convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, k, padding=k // 2) for k in (3, 5)]
        )
        self.local_bn = nn.BatchNorm1d(2 * embed_dim)
        self.local_drop = nn.Dropout(drop_prob)
        self.to_tokens = Rearrange("b d n -> b n d")
        self.w_q = nn.Linear(embed_dim, inner_dim)
        self.w_k_local = nn.Linear(2 * embed_dim, inner_dim)
        self.w_k_global = nn.Linear(embed_dim, inner_dim)
        self.w_v = nn.Linear(embed_dim, inner_dim)
        self.w_o = nn.Linear(inner_dim, embed_dim)
        self.split_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.att_drop = nn.Dropout(att_drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        local = self.to_channels(x)
        local = torch.cat([conv(local) for conv in self.local_convs], dim=1)
        local = self.local_bn(local)
        local = self.local_drop(local)
        local = self.to_tokens(local)

        q = self.split_heads(self.w_q(x))
        k_local = self.split_heads(self.w_k_local(local))
        k_global = self.split_heads(self.w_k_global(x))
        v = self.split_heads(self.w_v(x))
        # Explicit softmax attention keeps the reference's dropout RNG order.
        scale = math.sqrt(self.head_dim)
        attn_local = F.softmax(q @ k_local.transpose(-2, -1) / scale, dim=-1)
        x_local = self.att_drop(attn_local) @ v
        attn_global = F.softmax(q @ k_global.transpose(-2, -1) / scale, dim=-1)
        x_global = self.att_drop(attn_global) @ v
        return self.w_o(self.merge_heads(x_local + x_global))


class _TMSABlock(nn.Module):
    """Pre-norm residual block: TMSA attention, then feed-forward."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        fc_ratio,
        drop_prob,
        att_drop_prob,
        fc_drop_prob,
        activation,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = _TMSAAttention(embed_dim, num_heads, drop_prob, att_drop_prob)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForwardBlock(
            embed_dim,
            fc_ratio,
            fc_drop_prob,
            activation,
            output_drop_p=fc_drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))

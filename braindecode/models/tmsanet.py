"""TMSA-Net for motor-imagery EEG classification."""

# Authors: Qian Zhao <zhaoqian0120@qq.com>
#          Weina Zhu
#          (braindecode adaptation)
# License: MIT

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin


class TMSANet(EEGModuleMixin, nn.Module, license="mit"):
    r"""TMSA-Net from Zhao and Zhu (2025) [tmsanet]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    TMSA-Net combines multi-scale temporal convolutions, a spatial
    convolution across EEG channels, and a custom local/global attention
    block for motor-imagery classification.

    A notable property of the released attention implementation is that the
    per-head width is computed with floor division,
    ``head_dim = embed_dim // num_heads``. Therefore the default
    ``embed_dim=19``, ``num_heads=4`` configuration projects
    queries, keys and values through 16 dimensions (19 -> 16 -> 19), rather
    than requiring ``embed_dim`` to be divisible by the number of heads.

    Parameters
    ----------
    embed_dim : int, default=19
        Embedding width after the temporal/spatial feature extractor.
        The released source recommends 19 for BCI Competition IV 2a,
        6 for BCI Competition IV 2b, and 10 for HGD.
    pool_size : int, default=50
        Kernel size of the temporal average pooling layer.
    pool_stride : int, default=15
        Stride of the temporal average pooling layer.
    num_heads : int, default=4
        Number of custom attention heads.
    fc_ratio : int, default=2
        Expansion ratio of the Transformer feed-forward block.
    depth : int, default=1
        Number of Transformer encoder blocks.
    drop_prob : float, default=0.5
        Dropout probability used in the local-key multi-scale convolution
        and before the Transformer, matching the released source.
    att_drop_prob : float, default=0.5
        Dropout probability applied to local/global attention weights.
    fc_drop_prob : float, default=0.5
        Dropout probability in the feed-forward block.
    activation : nn.Module, default=nn.GELU
        Activation used after the spatial convolution and in the feed-forward
        block.

    References
    ----------
    .. [tmsanet] Zhao, Q., Zhu, W. TMSA-Net: A novel attention mechanism
        for improved motor imagery EEG signal processing. Biomedical Signal
        Processing and Control 102, 107189 (2025).
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

        if embed_dim < 1:
            raise ValueError("embed_dim must be at least 1.")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if embed_dim // num_heads < 1:
            raise ValueError("embed_dim must be at least num_heads.")
        if pool_size < 1 or pool_stride < 1:
            raise ValueError("pool_size and pool_stride must be positive.")
        if self.n_times < pool_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be at least pool_size ({pool_size})."
            )
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if fc_ratio < 1:
            raise ValueError("fc_ratio must be at least 1.")

        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.num_heads = num_heads
        self.fc_ratio = fc_ratio
        self.depth = depth
        self.drop_prob = drop_prob
        self.att_drop_prob = att_drop_prob
        self.fc_drop_prob = fc_drop_prob
        self.activation = activation

        self.feature_extractor = _TMSAFeatureExtractor(
            n_chans=self.n_chans,
            embed_dim=embed_dim,
            pool_size=pool_size,
            pool_stride=pool_stride,
            activation=activation,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.transformer = _TMSATransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fc_ratio=fc_ratio,
            depth=depth,
            local_drop_prob=drop_prob,
            att_drop_prob=att_drop_prob,
            fc_drop_prob=fc_drop_prob,
            activation=activation,
        )

        n_tokens = (self.n_times - pool_size) // pool_stride + 1
        self.final_layer = nn.Linear(embed_dim * n_tokens, self.n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        """Return classification logits for EEG input ``(batch, channels, time)``."""
        x = self.feature_extractor(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.final_layer(x)


class _TMSAFeatureExtractor(nn.Module):
    """Temporal multi-scale plus full-channel spatial feature extractor."""

    def __init__(
        self,
        n_chans: int,
        embed_dim: int,
        pool_size: int,
        pool_stride: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (n_chans, 1))
        self.activation = activation()
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.temp_conv1(x) + self.temp_conv2(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = x.squeeze(2)
        return self.avg_pool(x)


class _TMSAMultiScaleConv1d(nn.Module):
    """Local-key extractor used by the released TMSA attention block."""

    def __init__(
        self,
        embed_dim: int,
        drop_prob: float,
        kernel_sizes: tuple[int, int] = (3, 5),
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    embed_dim,
                    embed_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.bn = nn.BatchNorm1d(embed_dim * len(kernel_sizes))
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.bn(x)
        return self.dropout(x)


class _TMSAAttention(nn.Module):
    """Custom attention summing local-key and global-key branches."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        local_drop_prob: float,
        att_drop_prob: float,
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.inner_dim = num_heads * self.head_dim

        self.local_key = _TMSAMultiScaleConv1d(embed_dim, local_drop_prob)
        self.w_q = nn.Linear(embed_dim, self.inner_dim)
        self.w_k_local = nn.Linear(embed_dim * 2, self.inner_dim)
        self.w_k_global = nn.Linear(embed_dim, self.inner_dim)
        self.w_v = nn.Linear(embed_dim, self.inner_dim)
        self.w_o = nn.Linear(self.inner_dim, embed_dim)
        self.dropout = nn.Dropout(att_drop_prob)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        return x.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        local_key = self.local_key(x.transpose(1, 2)).transpose(1, 2)

        q = self._reshape_heads(self.w_q(x))
        k_local = self._reshape_heads(self.w_k_local(local_key))
        k_global = self._reshape_heads(self.w_k_global(x))
        v = self._reshape_heads(self.w_v(x))

        scale = math.sqrt(self.head_dim)
        local_scores = torch.matmul(q, k_local.transpose(-2, -1)) / scale
        global_scores = torch.matmul(q, k_global.transpose(-2, -1)) / scale

        local_attn = self.dropout(F.softmax(local_scores, dim=-1))
        global_attn = self.dropout(F.softmax(global_scores, dim=-1))

        attended = torch.matmul(local_attn, v) + torch.matmul(global_attn, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, -1, self.inner_dim
        )
        return self.w_o(attended)


class _TMSAFeedForward(nn.Module):
    """Reference two-layer feed-forward block."""

    def __init__(
        self,
        embed_dim: int,
        fc_ratio: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * fc_ratio)
        self.activation = activation()
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(embed_dim * fc_ratio, embed_dim)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)


class _TMSATransformerBlock(nn.Module):
    """Pre-norm TMSA attention and feed-forward residual block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        fc_ratio: int,
        local_drop_prob: float,
        att_drop_prob: float,
        fc_drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = _TMSAAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            local_drop_prob=local_drop_prob,
            att_drop_prob=att_drop_prob,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = _TMSAFeedForward(
            embed_dim=embed_dim,
            fc_ratio=fc_ratio,
            drop_prob=fc_drop_prob,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class _TMSATransformer(nn.Module):
    """Stack of TMSA Transformer blocks."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        fc_ratio: int,
        depth: int,
        local_drop_prob: float,
        att_drop_prob: float,
        fc_drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TMSATransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    fc_ratio=fc_ratio,
                    local_drop_prob=local_drop_prob,
                    att_drop_prob=att_drop_prob,
                    fc_drop_prob=fc_drop_prob,
                    activation=activation,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)

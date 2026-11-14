# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
# Adapted from https://github.com/wzhlearning/fNIRS-Transformer (MIT).
"""fNIRS-T and fNIRS-PreT: Transformer Models for fNIRS Classification."""

from __future__ import annotations

from typing import Literal

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order
from braindecode.modules import (
    FeedForwardBlock,
    MultiHeadAttention,
    PreNormTransformerBlock,
)


class FNIRST(EEGModuleMixin, nn.Module, license="mit"):
    r"""fNIRS-T from Wang et al. (2022).

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. versionadded:: 2.1

    Two transformer branches on paired HbO/HbR fNIRS windows
    :footcite:`wang2022fnirst`: a spatial branch with one token per group of
    ``spatial_kernel_size`` neighbouring channels and a channel branch with one
    token per channel, each from a strided temporal convolution. The two pooled
    outputs are concatenated, normalized and classified.

    `License <https://github.com/wzhlearning/fNIRS-Transformer/blob/main/LICENSE>`_

    Parameters
    ----------
    dim : int, default=64
        Token width of both branches.
    depth : int, default=6
        Transformer blocks per branch.
    heads : int, default=8
        Attention heads per block.
    mlp_dim : int, default=64
        Hidden width of the feed-forward sub-layers.
    pool : {"cls", "mean"}, default="cls"
        Token pooling of each branch.
    dim_head : int, default=64
        Width of each attention head.
    conv_channels : int, default=8
        Feature maps of both convolutions.
    spatial_kernel_size : int, default=5
        Channels per spatial token, clamped to the number of channels.
    temporal_kernel_size : int, default=30
        Temporal kernel width of both convolutions, in samples.
    temporal_stride : int, default=4
        Temporal stride of both convolutions, in samples.
    n_chromophores : {1, 2}, default=2
        Chromophores per channel; an odd ``n_chans`` uses 1.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info``.
    drop_prob : float, default=0.0
        Dropout in the attention and feed-forward sub-layers.
    emb_drop_prob : float, default=0.0
        Dropout on the token + positional embeddings.
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward sub-layers.

    References
    ----------
    .. footbibliography::
    """

    _preblock = False  # FNIRSPreT prepends the moving-average frontend

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 64,
        pool: Literal["cls", "mean"] = "cls",
        dim_head: int = 64,
        conv_channels: int = 8,
        spatial_kernel_size: int = 5,
        temporal_kernel_size: int = 30,
        temporal_stride: int = 4,
        n_chromophores: int = 2,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.0,
        emb_drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if self.n_times < temporal_kernel_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be >= temporal_kernel_size."
            )
        n_chromophores = 1 if self.n_chans % 2 else n_chromophores
        pairs_info = self._chs_info if n_chromophores == 2 else None
        order = fnirs_channel_order(pairs_info, channel_order)
        n_phys_chans = self.n_chans // n_chromophores
        self.pool = pool
        # (batch, n_chans, n_times) -> (batch, n_chromophores, n_phys_chans, n_times)
        if order == "interleaved" and n_chromophores == 2:
            self.to_planes = Rearrange("b (c g) t -> b g c t", g=2)
        else:
            self.to_planes = Rearrange("b (g c) t -> b g c t", g=n_chromophores)
        n_bins = (self.n_times - temporal_kernel_size) // temporal_stride + 1
        spatial_kernel = min(spatial_kernel_size, n_phys_chans)
        self.pre = (
            _FNIRSPreFilter(self.n_times, n_chromophores) if self._preblock else None
        )

        def branch() -> nn.Sequential:
            return nn.Sequential(
                *[
                    PreNormTransformerBlock(
                        dim,
                        MultiHeadAttention(
                            dim, heads, head_dim=dim_head, qkv_bias=False
                        ),
                        FeedForwardBlock(
                            dim, 1, drop_prob, activation, hidden_features=mlp_dim
                        ),
                        drop_prob,
                    )
                    for _ in range(depth)
                ]
            )

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(
                n_chromophores,
                conv_channels,
                (spatial_kernel, temporal_kernel_size),
                (1, temporal_stride),
            ),
            Rearrange("b c h w -> b h (c w)"),
            nn.Linear(n_bins * conv_channels, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_patch = nn.Parameter(
            torch.randn(1, n_phys_chans - spatial_kernel + 2, dim)
        )
        self.dropout_patch = nn.Dropout(emb_drop_prob)
        self.transformer_patch = branch()

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(
                n_chromophores,
                conv_channels,
                (1, temporal_kernel_size),
                (1, temporal_stride),
            ),
            Rearrange("b c h w -> b h (c w)"),
            nn.Linear(n_bins * conv_channels, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_channel = nn.Parameter(torch.randn(1, n_phys_chans + 1, dim))
        self.dropout_channel = nn.Dropout(emb_drop_prob)
        self.transformer_channel = branch()

        self.final_norm = nn.LayerNorm(dim * 2)
        self.final_layer = nn.Linear(dim * 2, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Redrawn after the layers, so reset_parameters() resets them too.
        with torch.no_grad():
            self.cls_token_patch.normal_()
            self.pos_embedding_patch.normal_()
            self.cls_token_channel.normal_()
            self.pos_embedding_channel.normal_()

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``x``: ``(batch, n_chans, n_times)``. ``mask`` (bool, same shape):
        True samples are zeroed first. ``ch_mask`` is refused: the spatial
        kernel and positional tables span every channel."""
        if ch_mask is not None:
            raise ValueError("FNIRST does not accept ch_mask (fixed channel layout).")
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        # (batch, n_chans, n_times) -> (batch, n_chromophores, n_phys_chans, n_times)
        x = self.to_planes(x).contiguous()
        if self.pre is not None:
            x = self.pre(x)
        batch_size = x.shape[0]
        # -> (batch, n_groups, dim) -> (batch, n_groups + 1, dim)
        patch = self.to_patch_embedding(x)
        patch = torch.cat(
            (self.cls_token_patch.expand(batch_size, -1, -1), patch), dim=1
        )
        patch = self.dropout_patch(patch + self.pos_embedding_patch)
        patch = self.transformer_patch(patch)
        # -> (batch, n_phys_chans, dim) -> (batch, n_phys_chans + 1, dim)
        channel = self.to_channel_embedding(x)
        channel = torch.cat(
            (self.cls_token_channel.expand(batch_size, -1, -1), channel), dim=1
        )
        channel = self.dropout_channel(channel + self.pos_embedding_channel)
        channel = self.transformer_channel(channel)
        # 2 x (batch, n_tokens + 1, dim) -> (batch, 2 * dim)
        if self.pool == "mean":
            features = torch.cat((patch.mean(dim=1), channel.mean(dim=1)), dim=1)
        else:
            features = torch.cat((patch[:, 0], channel[:, 0]), dim=1)
        features = self.final_norm(features)
        if return_features:
            cls_token = features if self.pool == "cls" else None
            return {"features": features, "cls_token": cls_token}
        return self.final_layer(features)


class FNIRSPreT(FNIRST, license="mit"):  # the kwarg is not inherited
    r"""fNIRS-PreT from Wang et al. (2022).

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. versionadded:: 2.1

    :class:`~braindecode.models.FNIRST` for raw signals
    :footcite:`wang2022fnirst`: three moving averages (kernels 5, 13, 7) and a
    LayerNorm over time per chromophore come first, in place of offline
    band-pass filtering and baseline correction. The parameters are those of
    :class:`~braindecode.models.FNIRST`.

    `License <https://github.com/wzhlearning/fNIRS-Transformer/blob/main/LICENSE>`_

    References
    ----------
    .. footbibliography::
    """

    _preblock = True


class _FNIRSPreFilter(nn.Module):
    def __init__(self, n_times: int, n_chromophores: int) -> None:
        super().__init__()
        self.smooth = nn.Sequential(  # "same" padding
            nn.AvgPool1d(5, stride=1, padding=2),
            nn.AvgPool1d(13, stride=1, padding=6),
            nn.AvgPool1d(7, stride=1, padding=3),
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(n_times) for _ in range(n_chromophores)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_chromophores, n_phys_chans, n_times) -> same shape
        return torch.stack(
            [norm(self.smooth(x[:, i])) for i, norm in enumerate(self.norms)], dim=1
        )

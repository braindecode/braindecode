# Authors: Hamdi Altaheri <haltaheri@uwaterloo.ca> (original implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT

from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d, Conv1dWithConstraint, DropPath


def _glorot_weight_zero_bias(module: nn.Module) -> None:
    """Xavier-uniform init on non-norm weights, zero bias (reference init).

    Replicates ``utils.weight_initialization.glorot_weight_zero_bias`` from the
    original TCFormer code.
    """
    for mod in module.modules():
        weight = getattr(mod, "weight", None)
        if weight is not None and "norm" not in mod.__class__.__name__.lower():
            nn.init.xavier_uniform_(weight)
        bias = getattr(mod, "bias", None)
        if bias is not None:
            nn.init.constant_(bias, 0)


# ----------------------------------------------------------------------------- #
# Rotary positional embedding (verbatim port of the reference; see plan note on
# the deliberate NeoX-cache / interleaved-rotate mismatch).
def _build_rotary_cache(head_dim: int, seq_len: int) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` each of shape ``(seq_len, head_dim)``."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    seq_idx = torch.arange(seq_len).float()
    freqs = torch.outer(seq_idx, theta)          # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)      # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _apply_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply rotary embedding to ``q`` and ``k`` (shape ``(B, heads, T, d)``)."""

    def _rotate(x: Tensor) -> Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    q_out = (q * cos) + (_rotate(q) * sin)
    k_out = (k * cos) + (_rotate(k) * sin)
    return q_out, k_out


# ----------------------------------------------------------------------------- #
class _GroupedQueryAttention(nn.Module):
    """Grouped-query self-attention (``q_heads >= kv_heads``) with RoPE."""

    def __init__(
        self, d_model: int, q_heads: int, kv_heads: int, drop_prob: float = 0.4
    ):
        super().__init__()
        assert d_model % q_heads == 0, "d_model must be divisible by q_heads"
        assert q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads"
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // q_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(drop_prob)
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.q_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(b, t, self.kv_heads, 2, self.head_dim)
        k = kv[..., 0, :].transpose(1, 2)
        v = kv[..., 1, :].transpose(1, 2)
        repeat = self.q_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        q, k = _apply_rope(q, k, cos[:t], sin[:t])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(out)


# ----------------------------------------------------------------------------- #
class _TransformerBlock(nn.Module):
    """Pre-norm encoder block: GQA + position-wise MLP, both with DropPath."""

    def __init__(
        self,
        d_model: int,
        q_heads: int,
        kv_heads: int,
        mlp_ratio: int = 2,
        drop_prob: float = 0.4,
        drop_path_rate: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _GroupedQueryAttention(d_model, q_heads, kv_heads, drop_prob)
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            activation(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------------------------------------------------------- #
class _ChannelGroupAttention(nn.Module):
    """Grouped Squeeze-and-Excitation: one sigmoid gate per channel group."""

    def __init__(self, in_channels: int, num_groups: int, reduction: int = 4):
        super().__init__()
        assert in_channels % num_groups == 0
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_size = in_channels // num_groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att_fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, groups=num_groups, bias=False
        )
        self.att_fc2 = nn.Conv2d(
            in_channels // reduction, num_groups, 1, groups=num_groups, bias=False
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        gate = self.sigmoid(self.att_fc2(self.relu(self.att_fc1(self.pool(x)))))
        x = x.view(b, self.num_groups, self.group_size, h, w)
        gate = gate.view(b, self.num_groups, 1, 1, 1)
        return (x * gate).view(b, c, h, w)


# ----------------------------------------------------------------------------- #
class _MultiKernelConvBlock(nn.Module):
    """Multi-kernel temporal + depthwise spatial CNN producing feature tokens."""

    def __init__(
        self,
        n_chans: int,
        temp_kernel_lengths: tuple[int, ...],
        n_filters_time: int,
        depth_multiplier: int,
        pool_length_1: int,
        pool_length_2: int,
        temp_kernel_length_2: int,
        drop_prob: float,
        group_dim: int,
        se_reduction: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        n_groups = len(temp_kernel_lengths)
        self.d_model = group_dim * n_groups
        self.rearrange = Rearrange("b c t -> b 1 c t")

        # one temporal conv per kernel ('same' time padding via ConstantPad2d),
        # BN only -- NO activation here (matches reference).
        self.temporal_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConstantPad2d(
                        (k // 2 - 1, k // 2, 0, 0)
                        if k % 2 == 0
                        else (k // 2, k // 2, 0, 0),
                        0,
                    ),
                    nn.Conv2d(1, n_filters_time, (1, k), bias=False),
                    nn.BatchNorm2d(n_filters_time),
                )
                for k in temp_kernel_lengths
            ]
        )

        in_dw = n_filters_time * n_groups
        f2 = in_dw * depth_multiplier
        self.channel_dw_conv = nn.Sequential(
            nn.Conv2d(in_dw, f2, (n_chans, 1), bias=False, groups=in_dw),
            nn.BatchNorm2d(f2),
            activation(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(drop_prob)

        # grouped 1x1 channel reduction f2 -> d_model (one group per kernel)
        self.use_channel_reduction = self.d_model != f2
        if self.use_channel_reduction:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(f2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )

        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(
                self.d_model,
                self.d_model,
                (1, temp_kernel_length_2),
                padding="same",
                bias=False,
                groups=n_groups,
            ),
            nn.BatchNorm2d(self.d_model),
            activation(),
        )

        self.use_group_attn = n_groups > 1
        if self.use_group_attn:
            self.group_attn = _ChannelGroupAttention(
                self.d_model, n_groups, se_reduction
            )

        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(drop_prob)

        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> Tensor:
        x = self.rearrange(x)
        x = torch.cat([conv(x) for conv in self.temporal_convs], dim=1)
        x = self.channel_dw_conv(x)
        x = self.drop1(self.pool1(x))
        if self.use_channel_reduction:
            x = self.channel_reduction(x)
        x = self.temporal_conv_2(x)
        if self.use_group_attn:
            x = x + self.group_attn(x)
        x = self.drop2(self.pool2(x))
        return x.squeeze(2)  # (B, d_model, Tc)


# ----------------------------------------------------------------------------- #
class _TCNResidualBlock(nn.Module):
    """Two grouped dilated-causal convs (BN + activation + dropout) + residual."""

    def __init__(
        self,
        n_filters: int,
        kernel_length: int,
        dilation: int,
        n_groups: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            n_filters, n_filters, kernel_length, dilation=dilation, groups=n_groups
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.act1 = activation()
        self.drop1 = nn.Dropout(drop_prob)
        self.conv2 = CausalConv1d(
            n_filters, n_filters, kernel_length, dilation=dilation, groups=n_groups
        )
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.act2 = activation()
        self.drop2 = nn.Dropout(drop_prob)
        self.act3 = activation()
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return self.act3(x + out)


class _TCN(nn.Module):
    """Stack of ``depth`` residual blocks with exponentially growing dilation."""

    def __init__(
        self,
        depth: int,
        kernel_length: int,
        n_filters: int,
        n_groups: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _TCNResidualBlock(
                    n_filters, kernel_length, 2**i, n_groups, drop_prob, activation
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

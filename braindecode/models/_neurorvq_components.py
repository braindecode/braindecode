# Authors: Konstantinos Barmpas et al. (original implementation)
#          Braindecode contributors (adaptation)
#
# License: CC BY-NC 4.0
"""Private NeuroRVQ layers required by the standalone tokenizer port."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NEURORVQ_CHANNELS = (
    "a1",
    "a2",
    "af3",
    "af4",
    "af7",
    "af8",
    "afz",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "ccp1",
    "ccp2",
    "ccp3",
    "ccp4",
    "ccp5",
    "ccp6",
    "ccp7",
    "ccp8",
    "cfc1",
    "cfc2",
    "cfc3",
    "cfc4",
    "cfc5",
    "cfc6",
    "cfc7",
    "cfc8",
    "cp1",
    "cp2",
    "cp3",
    "cp4",
    "cp5",
    "cp6",
    "cpz",
    "cz",
    "eog",
    "f1",
    "f10",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "fc1",
    "fc2",
    "fc3",
    "fc4",
    "fc5",
    "fc6",
    "fcz",
    "fp1",
    "fp2",
    "fpz",
    "ft7",
    "ft8",
    "fz",
    "iz",
    "loc",
    "o1",
    "o2",
    "oz",
    "p08",
    "p1",
    "p10",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "po1",
    "po10",
    "po2",
    "po3",
    "po4",
    "po7",
    "po8",
    "po9",
    "poz",
    "pz",
    "roc",
    "sp1",
    "sp2",
    "t1",
    "t10",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9",
    "tp10",
    "tp7",
    "tp8",
    "tp9",
)


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: Callable[[int], nn.Module] | None,
        attn_drop: float,
        proj_drop: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        inner_dim = head_dim * num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)
            )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).type_as(v)
            k = self.k_norm(k).type_as(v)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: Callable[[int], nn.Module] | None,
        drop: float,
        attn_drop: float,
        drop_path: float,
        init_values: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias, qk_norm, attn_drop, drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop)
        self.gamma_1 = (
            nn.Parameter(init_values * torch.ones(dim)) if init_values > 0 else None
        )
        self.gamma_2 = (
            nn.Parameter(init_values * torch.ones(dim)) if init_values > 0 else None
        )

    def forward(self, x: Tensor) -> Tensor:
        attn = self.attn(self.norm1(x))
        if self.gamma_1 is not None:
            attn = self.gamma_1 * attn
        x = x + self.drop_path(attn)
        mlp = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            mlp = self.gamma_2 * mlp
        return x + self.drop_path(mlp)


class _MultiScaleTemporalConv(nn.Module):
    """Released four-branch EEG patch embedding."""

    def __init__(self, out_chans: int = 8, activation: type[nn.Module] = nn.GELU):
        super().__init__()
        for i, kernel in enumerate((21, 15, 9, 5), start=1):
            setattr(
                self,
                f"conv1_{i}",
                nn.Conv2d(1, out_chans, (1, kernel), padding=(0, kernel // 2)),
            )
            setattr(self, f"norm1_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool1_{i}", nn.AvgPool2d((1, 2)))
        for i, kernel in enumerate((9, 7, 5, 3), start=1):
            setattr(
                self,
                f"conv2_{i}",
                nn.Conv2d(out_chans, out_chans, (1, kernel), padding=(0, kernel // 2)),
            )
            setattr(self, f"norm2_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool2_{i}", nn.AvgPool2d((1, 4)))
        self.gelu1 = activation()
        self.gelu2 = activation()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # x: batch x channels x patches x patch_size, in channel-major order.
        batch, channels, patches, patch_size = x.shape
        x = x.reshape(batch, channels * patches, patch_size).unsqueeze(1)
        outputs = []
        for i in range(1, 5):
            branch = getattr(self, f"pool1_{i}")(
                self.gelu1(getattr(self, f"norm1_{i}")(getattr(self, f"conv1_{i}")(x)))
            )
            branch = getattr(self, f"pool2_{i}")(
                self.gelu2(
                    getattr(self, f"norm2_{i}")(getattr(self, f"conv2_{i}")(branch))
                )
            )
            # Match the reference reshape B,C,channel*patch,time -> B,token,time*C.
            branch = branch.permute(0, 2, 3, 1).reshape(batch, channels * patches, -1)
            outputs.append(branch)
        return tuple(outputs)

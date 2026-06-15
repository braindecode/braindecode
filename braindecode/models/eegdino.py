# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.modules import DropPath


class _Mlp(nn.Module):
    """Feed-forward block (fc1 -> act -> fc2 -> drop); names match released keys."""

    def __init__(self, in_features, hidden_features, activation=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    """BEiT-style attention: fused qkv (bias=False) + separate q/v bias."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, n_tokens, dim = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch_size, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer block (LayerScale disabled, matching released config)."""

    def __init__(
        self, d_model, num_heads, dim_feedforward, activation=nn.GELU, drop_prob=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _Attention(
            d_model, num_heads=num_heads, attn_drop=drop_prob, proj_drop=drop_prob
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _Mlp(d_model, dim_feedforward, activation=activation, drop=drop_prob)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

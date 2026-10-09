# Authors: Phil Wang (lucidrains), linear-attention-transformer
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/lucidrains/linear-attention-transformer
# (version 0.19.1, MIT License, Copyright (c) 2020 Phil Wang).
"""Linear-attention transformer encoder of BIOT and TFMTokenizer.

The part of ``linear_attention_transformer.LinearAttentionTransformer`` these
models run: non-causal, global linear attention in every head, sequential
layers. The original also builds a local attention that this configuration
never calls (it has no parameters). Parameter names and order match the
original, so its checkpoints load unchanged, and ``forward`` takes no
``**kwargs``, so it compiles with :func:`torch.jit.script`.
"""

import torch
from torch import nn


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn  # registered before ``norm``, as in the original key order
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class _Chunk(nn.Module):
    """One chunk (``ff_chunks=1``): keeps the original ``fn.fn`` key level."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.0)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.act(self.w1(x))))


class _SelfAttention(nn.Module):
    """Global linear attention: softmax over features for queries, over the
    sequence for keys (Shen et al., 2021)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        self.d_heads = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        # (b, t, heads * d) -> (b, heads, t, d)
        q = self.to_q(x).reshape(b, t, -1, self.d_heads).transpose(1, 2)
        k = self.to_k(x).reshape(b, t, -1, self.d_heads).transpose(1, 2)
        v = self.to_v(x).reshape(b, t, -1, self.d_heads).transpose(1, 2)
        q = q.softmax(dim=-1) * self.d_heads**-0.5
        k = k.softmax(dim=-2)
        context = torch.einsum("bhnd,bhne->bhde", [k, v])
        attn = torch.einsum("bhnd,bhde->bhne", [q, context])
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))


class _Layers(nn.Module):
    """``SequentialSequence`` without argument routing or layer dropout."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:  # (attention, feed-forward)
            x = x + layer[0](x)
            x = x + layer[1](x)
        return x


class LinearAttentionTransformer(nn.Module):
    """Stack of pre-norm global linear attention and feed-forward layers.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    depth : int
        Number of layers.
    heads : int
        Number of attention heads; ``dim`` must be divisible by it.
    attn_layer_dropout : float
        Dropout on the attention output.
    """

    def __init__(
        self, dim: int, depth: int, heads: int = 8, attn_layer_dropout: float = 0.0
    ):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            # Feed-forward first: the original draws its initialization first.
            feed_forward = _Chunk(_FeedForward(dim))
            attention = _SelfAttention(dim, heads, attn_layer_dropout)
            layers.append(
                nn.ModuleList([_PreNorm(dim, attention), _PreNorm(dim, feed_forward)])
            )
        self.layers = _Layers(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

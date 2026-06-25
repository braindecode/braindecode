# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original DANCE)
#
# License: MIT
# Re-implemented from facebookresearch/dance (MIT); the upstream modules live
# in the `neuraltrain` package (neuraltrain.models.common / .simpleconv).
from __future__ import annotations

import math

import torch
from einops import rearrange, repeat
from torch import nn

from braindecode.modules.layers import ChannelMerger


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, activation, last):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=1, padding=pad, dilation=dilation
        )
        self.act = None if last else activation()

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SimpleConv(nn.Module):
    """Dilated conv front-end (Defossez lineage), parity-gated.

    No BatchNorm, no input/conv dropout, no skip in the DANCE config
    (verified from upstream ``_default_encoder_config``). ``activation``
    defaults to ``nn.ReLU`` to match upstream. Owns an OPTIONAL nested
    ``ChannelMerger`` (``self.merger``) applied first, mirroring
    ``neuraltrain.models.simpleconv.SimpleConvModel`` so the call path and
    state_dict keys (``merger.*``, ``initial_linear.*``, ``sequence.{k}.*``)
    align with upstream for parity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        hidden: int = 512,
        depth: int = 10,
        kernel_size: int = 9,
        dilation_growth: float = 2.5,
        initial_linear: int = 256,
        initial_depth: int = 1,
        drop_prob: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
        merger: "ChannelMerger | None" = None,
    ):
        super().__init__()
        if dilation_growth > 1 and kernel_size % 2 == 0:
            raise ValueError("Odd kernel required with dilation.")
        # Optional nested merger; when present the conv blocks consume its
        # out_channels (upstream sets in_channels = merger.n_virtual_channels).
        self.merger = merger
        conv_in = merger.heads.shape[0] if merger is not None else in_channels
        layers: list[nn.Module] = []
        for _ in range(initial_depth):
            layers.append(nn.Conv1d(conv_in, initial_linear, 1))
            conv_in = initial_linear
        self.initial_linear = nn.Sequential(*layers)
        sizes = [initial_linear] + [hidden] * (depth - 1) + [out_channels]
        blocks = []
        # Match ConvSequence: dilation starts at 1.0, int() per block, then
        # multiply by dilation_growth AFTER each block (no dilation_period reset).
        dilation = 1.0
        for i in range(depth):
            d = int(dilation)
            last = i == depth - 1
            blocks.append(
                _ConvBlock(sizes[i], sizes[i + 1], kernel_size, d, activation, last)
            )
            dilation = dilation * dilation_growth
        self.sequence = nn.ModuleList(blocks)
        # Dilation of the LAST block, read from the built conv stack so it
        # reflects the int-cumulative ConvSequence schedule (Global Constraint
        # line 22) rather than the power-form ``int(growth ** (depth - 1))``.
        self.max_dilation = self.sequence[-1].conv.dilation[0]

    def forward(
        self,
        x: torch.Tensor,
        positions: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        # x: (B, in_channels, T); positions: (B, in_channels, 2) when merger set.
        length = x.shape[-1]
        if self.merger is not None:
            if positions is None:
                raise ValueError("SimpleConv with a merger requires positions.")
            x = self.merger(x, positions)  # (B, merger.out_channels, T)
        x = self.initial_linear(x)
        for block in self.sequence:
            x = block(x)
        if x.shape[-1] < length:
            raise ValueError(f"Expected output time dim >= {length}, got {x.shape[-1]}")
        return x[..., :length]


def _fourier_encode(x, max_freq, num_bands):
    # Transcribed verbatim from dance/dance/models/perceiver.py:14-21
    # (== perceiver_pytorch.fourier_encode). x: (...); returns
    # (..., 2*num_bands + 1): [sin(num_bands), cos(num_bands), original_position].
    x = x.unsqueeze(-1)
    orig = x  # keep the raw position; appended as the final channel
    scales = torch.linspace(
        1.0, max_freq / 2, num_bands, device=x.device, dtype=x.dtype
    )
    scales = scales[(*((None,) * (x.ndim - 1)), Ellipsis)]
    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    return torch.cat((x, orig), dim=-1)


def _sinusoidal_latents(num_latents, dim):
    pe = torch.zeros(num_latents, dim)
    position = torch.arange(0, num_latents).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gates)


class _FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), _GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class _PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None and "context" in kwargs:
            kwargs["context"] = self.norm_context(kwargs["context"])
        return self.fn(x, **kwargs)


class _Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=2, dim_head=64):
        super().__init__()
        inner = dim_head * heads
        context_dim = context_dim or query_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context=None):
        h = self.heads
        context = context if context is not None else x
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Perceiver(nn.Module):
    """Perceiver cross-attention bottleneck to a fixed latent grid.

    Re-implemented from ``perceiver_pytorch`` 0.9.0 (parity-gated). Latents are
    sinusoidally initialized; inputs are fourier-encoded along the time axis.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_latents: int = 256,
        latent_dim: int = 128,
        depth: int = 6,
        cross_heads: int = 2,
        latent_heads: int = 2,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        max_freq: float = 10.0,
        num_freq_bands: int = 6,
        self_per_cross_attn: int = 1,
    ):
        super().__init__()
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        fourier_dim = num_freq_bands * 2 + 1  # = 13
        context_dim = input_dim + fourier_dim  # = 141
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        with torch.no_grad():
            self.latents.copy_(_sinusoidal_latents(num_latents, latent_dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        _PreNorm(
                            latent_dim,
                            _Attention(
                                latent_dim, context_dim, cross_heads, cross_dim_head
                            ),
                            context_dim=context_dim,
                        ),
                        _PreNorm(latent_dim, _FeedForward(latent_dim)),
                        nn.ModuleList(
                            [
                                nn.ModuleList(
                                    [
                                        _PreNorm(
                                            latent_dim,
                                            _Attention(
                                                latent_dim,
                                                None,
                                                latent_heads,
                                                latent_dim_head,
                                            ),
                                        ),
                                        _PreNorm(latent_dim, _FeedForward(latent_dim)),
                                    ]
                                )
                                for _ in range(self_per_cross_attn)
                            ]
                        ),
                    ]
                )
            )
        self.to_logits = nn.Identity()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: (B, T, input_dim). input_axis=1 -> single time axis.
        # Transcribed from dance _Perceiver.forward (perceiver.py:50-79):
        # meshgrid over the 1 axis, fourier-encode, rearrange "... n d -> ... (n d)".
        b, t, _ = data.shape
        axis_pos = [
            torch.linspace(-1.0, 1.0, steps=t, device=data.device, dtype=data.dtype)
        ]
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing="ij"), dim=-1)  # (T, 1)
        enc_pos = _fourier_encode(pos, self.max_freq, self.num_freq_bands)  # (T,1,13)
        enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")  # (T, 13)
        enc_pos = repeat(enc_pos, "... -> b ...", b=b)  # (B, T, 13)
        data = torch.cat((data, enc_pos), dim=-1)  # (B, T, 141)
        data = rearrange(data, "b ... d -> b (...) d")  # (B, T, 141)
        x = repeat(self.latents, "n d -> b n d", b=b)
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data) + x
            x = cross_ff(x) + x
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
        return self.to_logits(x)


def _sinusoidal_embeddings(length, dim):
    # (duplicate of _sinusoidal_latents; kept separate intentionally)
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _DecoderLayer(nn.Module):
    def __init__(self, dim, heads, drop_prob, activation):
        super().__init__()
        self.self_attn = _Attention(dim, None, heads, dim // heads)
        self.cross_attn = _Attention(dim, dim, heads, dim // heads)
        self.ff = _FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, queries, memory):
        q = self.norm1(queries)
        queries = queries + self.drop(self.self_attn(q))
        q = self.norm2(queries)
        queries = queries + self.drop(self.cross_attn(q, context=memory))
        q = self.norm3(queries)
        queries = queries + self.drop(self.ff(q))
        return queries


class DanceDetrDecoder(nn.Module):
    """DETR cross-attention decoder emitting per-query event spans.

    Re-implemented with inline pre-norm self-/cross-attention; bit-exact parity
    with x_transformers' ScaleNorm/rotary/scale_residual block is NOT a goal
    (see plan Global Constraints). ``center``/``duration`` heads are dropped.

    Note: ``activation`` is accepted for interface symmetry but the feed-forward
    hardwires GEGLU (see :class:`_FeedForward`); changing it has no effect.
    """

    def __init__(
        self,
        input_dim: int = 128,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        n_queries: int = 100,
        n_outputs: int = 4,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, dim))
        self.layers = nn.ModuleList(
            [_DecoderLayer(dim, heads, drop_prob, activation) for _ in range(depth)]
        )
        self.class_head = nn.Linear(dim, n_outputs)
        self.start_head = nn.Linear(dim, 1)
        self.end_head = nn.Linear(dim, 1)

    def forward(self, memory: torch.Tensor) -> dict:
        b, t, _ = memory.shape
        memory = self.input_proj(memory)  # (B, T, dim)
        pe = _sinusoidal_embeddings(t, memory.shape[-1]).to(memory)
        memory = memory + pe.unsqueeze(0).expand(b, -1, -1)
        x = self.query_embed.expand(b, -1, -1)
        for layer in self.layers:
            x = layer(x, memory)
        return {
            "class": self.class_head(x),
            "start": torch.sigmoid(self.start_head(x)).squeeze(-1),
            "end": torch.sigmoid(self.end_head(x)).squeeze(-1),
        }

# Authors: Qian Xiao and OpenTSLab BrainOmni contributors
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/OpenTSLab/BrainOmni (MIT). Shared by the
# models that run its spatial-temporal block (BrainOmni, MEG-XL).
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from braindecode.functional import apply_rotary, rotary_frequencies
from braindecode.modules.blocks import FeedForwardBlock


class _SELU(nn.SELU):
    """``nn.SELU`` as ``ELU(x, alpha * scale)`` times ``scale`` where ``x > 0``.

    Same values and gradients, through the ``scale == 1`` ELU kernel that every
    backend implements (see ``test_registered_model_avoids_accelerator_gaps``).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha, scale = (
            1.6732632423543772848170429916717,
            1.0507009873554804934193349852946,
        )
        # ELU is x on the positive side, so this equals the where(x > 0, x * scale,
        # elu) form bit for bit, without its zero-filled where backward. The
        # factor is at least float32, like the scalar multiply's compute type.
        one = x.new_ones((), dtype=torch.promote_types(x.dtype, torch.float32))
        factor = torch.where(x > 0, one * scale, one)
        return (F.elu(x, alpha * scale) * factor).to(x.dtype)


def _attend(q, k, v, n_head: int, dropout_p: float) -> torch.Tensor:
    """SDPA over ``(batch, seq, n_head * head_dim)`` inputs."""
    q = q.unflatten(-1, (n_head, -1))
    k = k.unflatten(-1, (n_head, -1))
    v = v.unflatten(-1, (n_head, -1))
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p
    )
    return out.transpose(1, 2).flatten(2)


class SpatialTemporalBlock(nn.Module):
    """Temporal attention (RoPE) on one feature half, spatial on the other.

    The block of BrainOmni and MEG-XL. Steps: RMS-normalize the tokens; the first
    half of the features attends over time (with RoPE) and the second half over
    the channels; the two outputs are concatenated, spatial first (the halves
    swap, as released), and added to the input; then a RMS-normalized
    feed-forward layer with its residual. ``(batch, n_chans, n_times, n_dim)``
    in and out.

    Parameters
    ----------
    n_dim : int
        Token width. Each attention works on ``n_dim // 2`` features.
    n_head : int
        Attention heads; each attention has ``n_head // 2``.
    dropout : float
        Dropout of the attention and of the feed-forward output.
    rope_positions : int, default=240
        Positions cached by the rotary embedding of the temporal attention; a
        longer sequence rebuilds it.
    """

    def __init__(self, n_dim, n_head, dropout, rope_positions=240):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim, eps=1e-6)
        self.time_attn = _MultiHeadAttentionRoPE(
            n_dim // 2, n_head // 2, dropout, True, rope_positions
        )
        self.spatial_attn = _MultiHeadAttentionRoPE(
            n_dim // 2, n_head // 2, dropout, False
        )
        self.pre_ff_norm = RMSNorm(n_dim, eps=1e-6)
        # Released FeedForward: Linear -> SELU -> Linear -> Dropout.
        self.ff = FeedForwardBlock(n_dim, 4, 0.0, _SELU, output_drop_p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, chans, times, dim = x.shape
        h = self.pre_attn_norm(x)
        # (b, c, t, d) -> (b * t, c, d) and (b * c, t, d), and back
        xs = h[..., dim // 2 :].permute(0, 2, 1, 3).reshape(batch * times, chans, -1)
        xt = h[..., : dim // 2].reshape(batch * chans, times, -1)
        xs = self.spatial_attn(xs).reshape(batch, times, chans, -1).permute(0, 2, 1, 3)
        xt = self.time_attn(xt).reshape(batch, chans, times, -1)
        # Spatial half first: the halves swap, as released.
        x = x + torch.cat([xs, xt], dim=-1)
        return x + self.ff(self.pre_ff_norm(x))


class _RotaryPositionalEmbedding(nn.Module):
    """RoPE of the released BrainOmni and MEG-XL.

    ``rotate`` caches ``n_positions`` positions as ``(cos, sin)`` pairs (240
    for BrainOmni, whose released checkpoints hold cosines only, zero sine;
    625 for MEG-XL). A longer sequence rebuilds the cache from ``freqs`` and
    later calls keep it (not saved). The caches stay float32 under a dtype
    cast, ``freqs`` follows it.
    """

    # TorchScript types the ``rebuilt`` buffer from this; a postponed string
    # annotation would not resolve.
    __annotations__ = {"rebuilt": Optional[torch.Tensor]}

    def __init__(self, n_dim, base=10000, n_positions=240):
        super().__init__()
        self.n_dim = n_dim
        self.register_buffer("freqs", rotary_frequencies(n_dim, base))
        self.register_buffer("rotate", self._polar(n_positions))
        self.register_buffer("rebuilt", None, persistent=False)

    def _polar(self, seq: int) -> torch.Tensor:
        positions = torch.arange(seq, device=self.freqs.device).type_as(self.freqs)
        angles = torch.outer(positions, self.freqs).float()
        return torch.stack((angles.cos(), angles.sin()), dim=-1)

    def _apply(self, fn, recurse=True):
        before = {name: self._buffers[name] for name in ("rotate", "rebuilt")}
        super()._apply(fn, recurse)
        for name, buffer in before.items():
            moved = self._buffers[name]
            if buffer is not None and moved is not None and moved.dtype != buffer.dtype:
                self._buffers[name] = buffer.to(device=moved.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self.rebuilt = None
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, q, k):
        """Rotate ``q`` and ``k`` of shape ``(batch, seq, n_heads, head_dim)``."""
        _, seq, heads, _ = q.shape
        rebuilt = self.rebuilt
        rotate = self.rotate if rebuilt is None else rebuilt
        if seq > rotate.shape[0]:
            rotate = self._polar(seq)
            self.rebuilt = rotate
        # In at least float32 (the cache's dtype), float64 for float64 inputs.
        work = torch.promote_types(q.dtype, torch.float32)
        rotate = rotate.to(work)[:seq].repeat_interleave(2, dim=1)
        rotate = rotate.unflatten(1, (heads, -1))  # (seq, heads, head_dim, 2)
        cos, sin = rotate[..., 0], rotate[..., 1]
        q_float, k_float = q.to(work), k.to(work)
        q_out = apply_rotary(q_float, cos, sin)
        k_out = apply_rotary(k_float, cos, sin)
        return q_out.type_as(q), k_out.type_as(k)


class _MultiHeadAttentionRoPE(nn.Module):
    """Self-attention with a fused ``qkv`` projection and optional RoPE."""

    def __init__(self, n_dim, n_head, dropout, rope, rope_positions=240):
        super().__init__()
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope_embedding_layer = (
            _RotaryPositionalEmbedding(n_dim, n_positions=rope_positions)
            if rope
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.split(self.qkv(x), self.n_dim, dim=-1)
        dropout_p = self.dropout if self.training else 0.0
        if self.rope_embedding_layer is not None:  # on (batch, seq, heads, head_dim)
            q, k = self.rope_embedding_layer(
                q.unflatten(-1, (self.n_head, -1)), k.unflatten(-1, (self.n_head, -1))
            )
            q, k = q.flatten(2), k.flatten(2)
        out = _attend(q, k, v, self.n_head, dropout_p)
        return self.proj(out)

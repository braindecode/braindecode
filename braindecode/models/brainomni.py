# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
#
# Ported from https://github.com/OpenTSLab/BrainOmni (MIT License, 2025 OpenTSLab).
# SEANet/conv/LSTM submodules derive from Meta's EnCodec (MIT License).
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# MNE FIFF constants (avoid importing mne at module load for a tiny lookup)
_FIFF_MEG_CH = 1
_FIFF_EEG_CH = 2
# sensor_type integer codes consumed by _BrainSensorModule
_SENSOR_EEG, _SENSOR_MAG, _SENSOR_GRAD = 0, 1, 2


def _resolve_kind(kind) -> int:
    """Return FIFF int kind from an MNE ``info['chs']`` ``kind`` field.

    Accepts the harness/string form (``"eeg"``, ``"mag"``, ``"grad"``) and the
    real MNE integer constants.
    """
    if isinstance(kind, str):
        k = kind.lower()
        if k == "eeg":
            return _FIFF_EEG_CH
        if k in ("mag", "grad", "meg"):
            return _FIFF_MEG_CH
        raise ValueError(f"Unsupported channel kind string: {kind!r}")
    return int(kind)


def _sensor_type_of(ch: dict) -> int:
    kind = _resolve_kind(ch.get("kind", "eeg"))
    if kind == _FIFF_EEG_CH:
        return _SENSOR_EEG
    if kind not in (_FIFF_EEG_CH, _FIFF_MEG_CH):
        raise ValueError(
            f"Unsupported channel kind {kind!r}; pass only EEG/MEG channels "
            "(e.g. raw.pick(['eeg', 'meg']))."
        )
    # MEG: distinguish MAG vs GRAD. Prefer the string kind when present,
    # else use coil_type (planar/grad coils contain "GRAD"/"PLANAR").
    raw_kind = ch.get("kind")
    if isinstance(raw_kind, str) and raw_kind.lower() == "grad":
        return _SENSOR_GRAD
    coil = str(ch.get("coil_type", ""))
    if "PLANAR" in coil or "GRAD" in coil:
        return _SENSOR_GRAD
    return _SENSOR_MAG


def _orientation_of(ch: dict, sensor_type: int) -> np.ndarray:
    """3-D orientation vector extracted from ``ch['loc']``.

    EEG: zeros (3,).
    GRAD: ``loc[3:6]`` (first direction triplet).
    MAG: ``loc[9:12]`` (third direction triplet).
    """
    if sensor_type == _SENSOR_EEG:
        return np.zeros(3, dtype=np.float64)
    loc = np.asarray(ch["loc"], dtype=np.float64)
    dir_idx = 1 if sensor_type == _SENSOR_GRAD else 3
    return loc[3 * dir_idx : 3 * (dir_idx + 1)]


def _normalize_pos(pos: np.ndarray, sensor_type: np.ndarray) -> np.ndarray:
    """Per-modality position normalization (upstream ``normalize_pos``).

    EEG positions and MEG positions are each mean-centered then divided by
    ``sqrt(3 * mean(squared_norm))``.
    """
    pos = pos.copy()
    eeg = sensor_type == _SENSOR_EEG
    meg = (sensor_type == _SENSOR_MAG) | (sensor_type == _SENSOR_GRAD)
    for mask in (eeg, meg):
        if not mask.any():
            continue
        xyz = pos[mask, :3]
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz**2, axis=1)))
        scale = scale if scale > 0 else 1.0
        pos[mask, :3] = xyz / scale
    return pos


def _geometry_from_chs_info(chs_info):
    """Derive ``(pos (C, 6) float32, sensor_type (C,) int64)`` from ``chs_info``.

    Raises ``ValueError`` if any channel has a missing/non-finite ``loc``.
    """
    if not chs_info:
        raise ValueError("chs_info is empty; at least one channel is required.")
    pos, stype = [], []
    for ch in chs_info:
        if "loc" not in ch:
            raise ValueError(
                f"Channel {ch.get('ch_name', '?')!r} has no 'loc'. "
                "BrainOmni needs sensor positions; call raw.set_montage(...)."
            )
        loc = np.asarray(ch["loc"], dtype=np.float64)
        if loc.shape != (12,):
            raise ValueError(
                f"Channel {ch.get('ch_name', '?')!r} has loc of shape "
                f"{loc.shape}; expected (12,)."
            )
        s = _sensor_type_of(ch)
        ori = _orientation_of(ch, s)
        vec = np.concatenate([loc[:3], ori])
        if not np.all(np.isfinite(vec)):
            raise ValueError(
                f"Channel {ch.get('ch_name', '?')!r} has non-finite position. "
                "BrainOmni needs sensor positions; call raw.set_montage(...)."
            )
        pos.append(vec)
        stype.append(s)
    pos = np.stack(pos).astype(np.float32)
    sensor_type = np.asarray(stype, dtype=np.int64)
    pos = _normalize_pos(pos, sensor_type).astype(np.float32)
    return pos, sensor_type


# ---------------------------------------------------------------------------
# Attention / norm primitives (ported from BrainOmni/model_utils/attn.py)
# ---------------------------------------------------------------------------


class _RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, n_dim, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim)) if elementwise_affine else 1.0
        self.eps = eps

    def forward(self, x: torch.Tensor):
        weight = self.weight
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (weight * x).to(input_dtype)


class _FeedForward(nn.Module):
    """Two-layer feed-forward block with SELU activation."""

    def __init__(self, n_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_dim, int(4 * n_dim)),
            nn.SELU(),
            nn.Linear(int(4 * n_dim), n_dim),
            nn.Dropout(dropout) if dropout != 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)


class _RotaryEmbedding(nn.Module):
    """Complex-valued Rotary Position Embedding (RoPE)."""

    def __init__(self, n_dim, init_seq_len, base=10000):
        super().__init__()
        self.register_buffer(
            "freqs",
            1.0 / (base ** (torch.arange(0, n_dim, 2)[: (n_dim // 2)].float() / n_dim)),
        )
        self._set_rotate_cache(init_seq_len)

    def _set_rotate_cache(self, seq_len):
        self.max_seq_len_cache = seq_len
        t = torch.arange(seq_len, device=self.freqs.device).type_as(self.freqs)
        rotate = torch.outer(t, self.freqs).float()
        self.register_buffer("rotate", torch.polar(torch.ones_like(rotate), rotate))

    def reshape_for_broadcast(self, x: torch.Tensor):
        """x: Batch seq n_head d_head  /  rotate: seq dim."""
        B, T, H, D = x.shape
        if T > self.max_seq_len_cache:
            self._set_rotate_cache(T)
        rotate = self.rotate[:T, :]
        assert H * D == rotate.shape[1]
        return rearrange(rotate, "T (H D)-> T H D", H=H).unsqueeze(0)

    def forward(self, q, k):
        assert len(q.shape) == len(k.shape) == 4
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        rotate = self.reshape_for_broadcast(q_)
        q_out = torch.view_as_real(q_ * rotate).flatten(3)
        k_out = torch.view_as_real(k_ * rotate).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


class _SelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE and causal masking."""

    def __init__(
        self,
        n_dim,
        n_head,
        dropout,
        causal: bool = False,
        rope: bool = False,
    ):
        super().__init__()
        assert n_dim % n_head == 0
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.causal = causal
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope = rope
        self.rope_embedding_layer = (
            _RotaryEmbedding(n_dim=n_dim, init_seq_len=240)
            if self.rope
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, mask=None):
        B, T, C = x.shape
        x = self.qkv(x)
        q, k, v = torch.split(x, split_size_or_sections=self.n_dim, dim=-1)

        if self.rope:
            q = q.view(B, T, self.n_head, -1)
            k = k.view(B, T, self.n_head, -1)
            q, k = self.rope_embedding_layer(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = rearrange(q, "B T (H D) -> B H T D", H=self.n_head)
            k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)

        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, T, -1)
        return self.proj(output)


class _SpatialTemporalAttentionBlock(nn.Module):
    """Spatial-temporal factored attention block from BrainOmni.

    Splits the feature dimension in half: one half is attended over the
    temporal axis (per channel), the other over the spatial axis (per
    time-step).  Both halves are concatenated and passed through a
    feed-forward layer with pre-normalisation.

    Parameters
    ----------
    n_dim : int
        Total feature dimension (must be even).
    n_head : int
        Total number of attention heads (must be even).
    dropout : float
        Dropout probability for feed-forward and attention.
    causal : bool
        Whether to apply causal masking to the temporal attention.
    """

    def __init__(self, n_dim, n_head, dropout, causal):
        super().__init__()
        assert n_dim % 2 == 0 and n_head % 2 == 0, (
            "n_dim and n_head must be even (split into spatial/temporal halves)"
        )
        self.pre_attn_norm = _RMSNorm(n_dim)
        self.time_attn = _SelfAttention(
            n_dim // 2, n_head // 2, dropout, causal=causal, rope=True
        )
        self.spatial_attn = _SelfAttention(
            n_dim // 2, n_head // 2, dropout, causal=False, rope=False
        )
        self.pre_ff_norm = _RMSNorm(n_dim)
        self.ff = _FeedForward(n_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self._attn_operator(self.pre_attn_norm(x))
        x = x + self.ff(self.pre_ff_norm(x))
        return x

    def _attn_operator(self, x):
        B, C, W, D = x.shape
        xs = rearrange(x[:, :, :, D // 2 :], "B C W D -> (B W) C D")
        xt = rearrange(x[:, :, :, : D // 2], "B C W D->(B C) W D")
        xs = self.spatial_attn(xs, None)
        xt = self.time_attn(xt, None)
        xs = rearrange(xs, "(B W) C D -> B C W D", B=B)
        xt = rearrange(xt, "(B C) W D->B C W D", B=B)
        return torch.cat([xt, xs], dim=-1)

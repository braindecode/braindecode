# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
#
# Ported from https://github.com/OpenTSLab/BrainOmni (MIT License, 2025 OpenTSLab).
# SEANet/conv/LSTM submodules derive from Meta's EnCodec (MIT License).
from __future__ import annotations

import math
import warnings

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Classic weight_norm keeps ``conv.weight_g``/``weight_v`` keys (checkpoint parity).
from torch.nn.utils import weight_norm  # noqa: F401

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info

_SENSOR_CODE = {"eeg": 0, "mag": 1, "grad": 2}


def _normalize_pos(pos: np.ndarray, sensor_type: np.ndarray) -> np.ndarray:
    """Per-modality position normalization (upstream ``normalize_pos``).

    EEG (code 0) and MEG (codes 1, 2) positions are each mean-centered then
    divided by ``sqrt(3 * mean(squared_norm))``.
    """
    pos = pos.copy()
    eeg = sensor_type == 0
    meg = (sensor_type == 1) | (sensor_type == 2)
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

    Positions come from :func:`extract_channel_locations_from_chs_info`; MEG coil
    orientation (``loc[3:6]`` GRAD / ``loc[9:12]`` MAG) and the EEG/MAG/GRAD type
    split (via :meth:`mne.Info.get_channel_types`) are model-specific. Raises if
    any channel lacks a finite position.
    """
    if not chs_info:
        raise ValueError("chs_info is empty; at least one channel is required.")
    try:
        info = mne.Info(
            chs=chs_info,
            ch_names=[ch["ch_name"] for ch in chs_info],
            sfreq=1.0,
            nchan=len(chs_info),
            bads=[],
        )
        types = info.get_channel_types()
    except Exception:
        types = [
            str(ch.get("ch_type", ch.get("kind", "eeg"))).lower() for ch in chs_info
        ]

    xyz = extract_channel_locations_from_chs_info(chs_info)
    if xyz is None or len(xyz) != len(chs_info) or not np.isfinite(xyz).all():
        raise ValueError(
            "chs_info lacks finite sensor positions; call raw.set_montage(...)."
        )

    pos, sensor_type = [], []
    for i, ch_type in enumerate(types):
        if ch_type not in _SENSOR_CODE:
            raise ValueError(
                f"Unsupported channel type {ch_type!r}; pass only EEG/MEG channels."
            )
        if ch_type == "eeg":
            ori = np.zeros(3)
        else:
            loc = np.asarray(chs_info[i]["loc"], dtype=np.float64)
            ori = loc[3:6] if ch_type == "grad" else loc[9:12]
        pos.append(np.concatenate([xyz[i].astype(np.float64), ori]))
        sensor_type.append(_SENSOR_CODE[ch_type])

    pos = _normalize_pos(
        np.stack(pos).astype(np.float32), np.asarray(sensor_type, dtype=np.int64)
    )
    return pos.astype(np.float32), np.asarray(sensor_type, dtype=np.int64)


# Attention / norm primitives


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def _apply(self, fn, recurse=True):
        # Rotate cache is complex64; a real-dtype cast would drop imaginary part, so regenerate.
        prev_dtype = self.rotate.dtype
        result = super()._apply(fn, recurse=recurse)
        if self.rotate.dtype != prev_dtype:
            self._set_rotate_cache(self.max_seq_len_cache)
        return result

    def reshape_for_broadcast(self, x: torch.Tensor):
        """x: Batch seq n_head d_head  /  rotate: seq dim."""
        batch, seq, heads, head_dim = x.shape
        if seq > self.max_seq_len_cache:
            self._set_rotate_cache(seq)
        rotate = self.rotate[:seq, :]
        assert heads * head_dim == rotate.shape[1], (
            f"RoPE cache shape mismatch: heads={heads}, head_dim={head_dim}, rotate.shape[1]={rotate.shape[1]}"
        )
        return rearrange(
            rotate, "seq (heads head_dim) -> seq heads head_dim", heads=heads
        ).unsqueeze(0)

    def forward(self, q, k):
        assert len(q.shape) == len(k.shape) == 4, (
            f"Expected 4-D q and k, got q.ndim={len(q.shape)}, k.ndim={len(k.shape)}"
        )
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        rotate = self.reshape_for_broadcast(q_)
        q_out = torch.view_as_real(q_ * rotate).flatten(3)
        k_out = torch.view_as_real(k_ * rotate).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


class _Attention(nn.Module):
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
        assert n_dim % n_head == 0, (
            f"n_dim ({n_dim}) must be divisible by n_head ({n_head})"
        )
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
        batch, seq, dim = x.shape
        x = self.qkv(x)
        q, k, v = torch.split(x, split_size_or_sections=self.n_dim, dim=-1)

        if self.rope:
            q = q.view(batch, seq, self.n_head, -1)
            k = k.view(batch, seq, self.n_head, -1)
            q, k = self.rope_embedding_layer(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = rearrange(
                q,
                "batch seq (heads head_dim) -> batch heads seq head_dim",
                heads=self.n_head,
            )
            k = rearrange(
                k,
                "batch seq (heads head_dim) -> batch heads seq head_dim",
                heads=self.n_head,
            )

        v = rearrange(
            v,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )

        if mask is not None:
            mask = mask.unsqueeze(1)

        # SDPA applies dropout regardless of train mode; gate it for deterministic eval.
        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(batch, seq, -1)
        return self.proj(output)


class _STBlock(nn.Module):
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
        self.pre_attn_norm = nn.RMSNorm(n_dim, eps=1e-6)
        self.time_attn = _Attention(
            n_dim // 2, n_head // 2, dropout, causal=causal, rope=True
        )
        self.spatial_attn = _Attention(
            n_dim // 2, n_head // 2, dropout, causal=False, rope=False
        )
        self.pre_ff_norm = nn.RMSNorm(n_dim, eps=1e-6)
        self.ff = _FeedForward(n_dim, dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self._attn_operator(self.pre_attn_norm(x))
        x = x + self.ff(self.pre_ff_norm(x))
        return x

    def _attn_operator(self, x):
        batch, chans, tokens, dim = x.shape
        # Upper half of feature dim attends over channels (spatial); lower half over windows (temporal).
        xs = rearrange(
            x[:, :, :, dim // 2 :], "batch chans tokens dim -> (batch tokens) chans dim"
        )
        xt = rearrange(
            x[:, :, :, : dim // 2], "batch chans tokens dim -> (batch chans) tokens dim"
        )
        xs = self.spatial_attn(xs, None)
        xt = self.time_attn(xt, None)
        xs = rearrange(
            xs, "(batch tokens) chans dim -> batch chans tokens dim", batch=batch
        )
        xt = rearrange(
            xt, "(batch chans) tokens dim -> batch chans tokens dim", batch=batch
        )
        # Spatial first, temporal second: halves are swapped vs. input split (upstream parity).
        return torch.cat([xs, xt], dim=-1)


# SEANet conv helpers


class _SConv1d(nn.Module):
    """Conv1d with built-in asymmetric / causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",  # kept for call-site compat; always weight_norm
        norm_kwargs: dict | None = None,  # kept for call-site compat; unused
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "_SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        # Classic weight_norm preserves checkpoint key parity with upstream.
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        # Right-pad so the last strided window is full: round the effective
        # length up to the next multiple of ``stride`` (EnCodec convention).
        extra_padding = (kernel_size - padding_total - x.shape[-1]) % stride
        if self.causal:
            x = F.pad(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = F.pad(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class _SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in asymmetric / causal padding trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",  # kept for call-site compat; always weight_norm
        trim_right_ratio: float = 1.0,
        norm_kwargs: dict | None = None,  # kept for call-site compat; unused
    ):
        super().__init__()
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert 0.0 <= self.trim_right_ratio <= 1.0
        padding_total = kernel_size - stride
        if causal:
            # Causal: no native padding; trimming is done in forward via inline slice.
            self.convtr = weight_norm(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
            )
        else:
            # Non-causal: use native padding= to remove padding_total//2 per side.
            assert padding_total % 2 == 0, (
                f"Non-causal _SConvTranspose1d requires even padding_total "
                f"(kernel_size-stride={padding_total}); got kernel_size={kernel_size}, "
                f"stride={stride}."
            )
            self.convtr = weight_norm(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding_total // 2,
                )
            )
        self._padding_total = padding_total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(self._padding_total * self.trim_right_ratio)
            padding_left = self._padding_total - padding_right
            y = y[..., padding_left : y.shape[-1] - padding_right]
        return y


# SLSTM


class _SLSTM(nn.Module):
    """LSTM over convolutional layout (channels-last time axis)."""

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(
            dimension, dimension, num_layers, bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.bidirectional:
            x = x.repeat(1, 1, 2)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


# SEANet residual block + encoder/decoder


class _SEANetResBlock(nn.Module):
    """SEANet residual block (dilated convs + skip connection)."""

    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        activation: str = "ELU",
        activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict | None = None,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 1]
        if dilations is None:
            dilations = [1, 1]
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        norm_params = norm_params or {}
        assert len(kernel_sizes) == len(dilations), (
            "Number of kernel sizes must match number of dilations"
        )
        act = getattr(nn, activation)
        hidden = dim // compress
        block: list[nn.Module] = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                _SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = _SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class _SEANetEncoder(nn.Module):
    """SEANet encoder: strided conv stack (``prod(ratios)`` downsampling) + LSTM."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: list[int] | None = None,
        activation: str = "ELU",
        activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 4, 2]
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        norm_params = norm_params or {}
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))

        act = getattr(nn, activation)
        mult = 1
        model: list[nn.Module] = [
            _SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        for ratio in self.ratios:
            for j in range(n_residual_layers):
                model += [
                    _SEANetResBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            model += [
                act(**activation_params),
                _SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [
                _SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)
            ]

        mult = mult * 2 if bidirectional else mult
        model += [
            act(**activation_params),
            _SConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _SEANetDecoder(nn.Module):
    """SEANet decoder: mirror of the encoder (transposed-conv upsampling)."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: list[int] | None = None,
        activation: str = "ELU",
        activation_params: dict | None = None,
        final_activation: str | None = None,
        final_activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 4, 2]
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        norm_params = norm_params or {}
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: list[nn.Module] = [
            _SConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [
                _SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)
            ]

        for ratio in self.ratios:
            model += [
                act(**activation_params),
                _SConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            for j in range(n_residual_layers):
                model += [
                    _SEANetResBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            mult //= 2

        model += [
            act(**activation_params),
            _SConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


# Residual Vector Quantization


def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


# Rotation-trick STE (Fifty et al. 2024, §4.2 https://arxiv.org/abs/2410.06424).


def _efficient_rotation_trick_transform(u, q, e):
    # Householder reflection + rank-1 correction: e - 2<e,w>w + 2<e,u>q, w = normalize(u+q).
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()
    ew = (e * w).sum(dim=1, keepdim=True)
    eu = (e * u.detach()).sum(dim=1, keepdim=True)
    return e - 2 * ew * w + 2 * eu * q.detach()


def _rotate_to(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Rotation-trick straight-through estimator (Fifty et al. 2024)."""
    orig_shape = src.shape
    src = src.reshape(-1, orig_shape[-1])
    tgt = tgt.reshape(-1, orig_shape[-1])
    norm_src = src.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_tgt = tgt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    rotated = _efficient_rotation_trick_transform(src / norm_src, tgt / norm_tgt, src)
    rotated = rotated * (norm_tgt / norm_src).detach()
    return rotated.reshape(orig_shape)


class _Codebook(nn.Module):
    """EMA-updated Euclidean codebook; buffer keys match upstream checkpoint."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.codebook_size = codebook_size

        embed = torch.empty(codebook_size, dim)
        nn.init.kaiming_uniform_(embed)
        # 'inited' buffer kept so state-dict keys match upstream checkpoint.
        self.register_buffer("inited", torch.tensor([True]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def _sample_vectors(self, samples: torch.Tensor, num: int) -> torch.Tensor:
        num_samples, device = samples.shape[0], samples.device
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)
        return samples[indices]

    def replace_(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        modified = torch.where(
            mask[..., None],
            self._sample_vectors(samples, self.codebook_size),
            self.embed,
        )
        self.embed.data.copy_(modified)

    def expire_codes_(self, batch_samples: torch.Tensor) -> None:
        if self.threshold_ema_dead_code == 0:
            return
        expired = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired):
            return
        batch_samples = rearrange(batch_samples, "... dim -> (...) dim")
        self.replace_(batch_samples, expired)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        embed = self.embed.t().float()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        return dist.argmin(dim=-1)

    def dequantize(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_ind, self.embed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = rearrange(x, "... dim -> (...) dim")
        embed_ind = self.quantize(x)
        return embed_ind.view(*shape[:-1])

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return self.dequantize(embed_ind)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x = rearrange(x, "... dim -> (...) dim")
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.dequantize(embed_ind).type(dtype)

        if self.training:
            self.expire_codes_(x)
            one_hot_sum = embed_onehot.sum(0)
            _ema_inplace(self.cluster_size, one_hot_sum, self.decay)
            embed_sum = (embed_onehot.t() @ x).to(torch.float32)
            _ema_inplace(self.embed_avg, embed_sum, self.decay)
            smoothed = (self.cluster_size + self.epsilon) / (
                self.cluster_size.sum() + self.epsilon * self.codebook_size
            )
            cluster_size = smoothed * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class _VQ(nn.Module):
    """Single-layer vector quantisation with optional rotation-trick STE."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        threshold_ema_dead_code: int = 2,
        rotation_trick: bool = True,
    ):
        super().__init__()
        _codebook_dim: int = codebook_dim if codebook_dim is not None else dim
        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self._codebook = _Codebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.rotation_trick = rotation_trick
        self.codebook_size = codebook_size

    @property
    def codebook(self) -> torch.Tensor:
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        return self._codebook.encode(x)

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        q = self._codebook.decode(embed_ind)
        return self.project_out(q)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            if self.rotation_trick:
                quantize = _rotate_to(x, quantize).to(input_dtype)
            else:
                quantize = x + (quantize - x).detach()
        loss = F.mse_loss(x.float(), quantize.detach().float()) * 0.25
        if not self.training:
            loss = loss.detach()
        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class _ResidualVQ(nn.Module):
    """Residual vector quantisation with L2-normalisation (EMA path, no quantize-dropout)."""

    def __init__(
        self,
        dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool = True,
        quantize_optimize_method: str = "ema",
    ):
        super().__init__()
        if quantize_optimize_method != "ema":
            raise ValueError(f"Only 'ema' supported, got {quantize_optimize_method!r}")
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList(
            [
                _VQ(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    rotation_trick=rotation_trick,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.normalize(x, p=2.0, dim=-1)
        quantized_out: torch.Tensor | float = 0.0
        residual = x
        all_losses: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        for vq in self.layers:
            quantized, indices, loss = vq(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_indices.append(indices)
        all_losses_t = torch.stack(all_losses, dim=-1)
        all_indices_t = torch.stack(all_indices, dim=-1)
        return quantized_out, all_indices_t, all_losses_t.mean()  # type: ignore[return-value]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2.0, dim=-1)
        residual = x
        all_indices: list[torch.Tensor] = []
        for vq in self.layers:
            quantized, indices, _ = vq(residual)
            residual = residual - quantized.detach()
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)


class _SensorEmbedding(nn.Module):
    """Embed per-channel position+orientation (B,C,6) and type (B,C) -> (B,C,n_dim)."""

    def __init__(self, n_dim: int) -> None:
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2),
            nn.SELU(),
            nn.Linear(n_dim // 2, n_dim),
        )
        self.aggregate_mlp = _FeedForward(n_dim, 0.0)
        self.norm = nn.RMSNorm(n_dim, eps=1e-6)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding_layer(pos)
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


class _ForwardSolution(nn.Module):
    """Cross-attention: sensor embeddings query the neural-token key/values."""

    def __init__(self, n_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        sensor_embedding: torch.Tensor,
        neurons: torch.Tensor,
    ) -> torch.Tensor:
        batch, chans, _ = sensor_embedding.shape
        kv = self.kv(neurons)
        k, v = torch.split(kv, split_size_or_sections=self.n_dim, dim=-1)
        q = rearrange(
            sensor_embedding,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        k = rearrange(
            k,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        v = rearrange(
            v,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(batch, chans, -1)
        return self.proj(output)


# _BackwardSolution: neuros (query) attend to sensor+feature keys / channel values


class _BackwardSolution(nn.Module):
    """Cross-attention: neural queries attend to pre-projected sensor keys/values."""

    def __init__(self, n_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.v = nn.Linear(n_dim, n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        neuros: torch.Tensor,
        k: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_queries, _ = neuros.shape
        q = rearrange(
            neuros,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        k = rearrange(
            k,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        v = rearrange(
            self.v(x),
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(batch, n_queries, -1)
        return self.proj(output)


class _TokenizerEncoder(nn.Module):
    """SEANet conv-encode windows, then collapse channels to ``n_neuro`` queries.

    ``(B, C, N, L)`` -> ``(B, n_neuro, N, T, n_dim)`` with ``T = L / prod(ratios)``.
    """

    def __init__(
        self,
        n_filters: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        dropout: float,
        n_neuro: int,
    ) -> None:
        super().__init__()
        self.seanet_encoder = _SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = _BackwardSolution(
            n_dim=n_dim, n_head=n_head, dropout=dropout
        )
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        x: torch.Tensor,
        sensor_embedding: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert sensor_embedding is not None, "sensor_embedding is required"
        batch, chans, nwin, wlen = x.shape
        # Each (sample, channel, window) is an independent 1-D waveform for SEANet.
        x = rearrange(x, "batch chans nwin wlen -> (batch chans nwin) 1 wlen")
        x = self.seanet_encoder(x)
        # Re-group encoded windows back so tokens = nwin*tok is the joint time axis.
        x = rearrange(
            x,
            "(batch chans nwin) dim tok -> batch chans (nwin tok) dim",
            batch=batch,
            chans=chans,
            nwin=nwin,
        )
        batch, chans, tokens, _ = x.shape
        # Expand sensor embedding to match every time step, then flatten for cross-attention.
        sensor_embedding = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, tokens, 1),
            "batch chans tokens dim -> (batch tokens) chans dim",
        )
        x = rearrange(x, "batch chans tokens dim -> (batch tokens) chans dim")
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        # Collapse the n_neuro-channel axis back into the batch; split window from token dims.
        x = rearrange(
            x,
            "(batch nwin tok) chans dim -> batch chans (nwin tok) dim",
            batch=batch,
            nwin=nwin,
        )
        return rearrange(
            x, "batch chans (nwin tok) dim -> batch chans nwin tok dim", nwin=nwin
        )


class _TokenizerDecoder(nn.Module):
    """Expand ``n_neuro`` tokens back to channels, then SEANet conv-decode.

    ``(B, n_neuro, N, T, n_dim)`` -> ``(B, C, N, L)`` reconstructed waveforms.
    """

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        n_filters: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.forwardsolution = _ForwardSolution(n_dim, n_head, dropout)
        self.seanet_decoder = _SEANetDecoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        sensor_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert sensor_embedding is not None, "sensor_embedding is required"
        batch, chans, nwin, tok, dim = x.shape
        # Flatten batch × window × token into one axis so cross-attention sees
        # each (sample, window, token) independently across n_neuro latent channels.
        x = rearrange(x, "batch chans nwin tok dim -> (batch nwin tok) chans dim")
        # Match sensor embedding shape to the flattened batch axis.
        sensor_embedding = rearrange(
            sensor_embedding.view(batch, -1, 1, 1, dim).repeat(1, 1, nwin, tok, 1),
            "batch chans nwin tok dim -> (batch nwin tok) chans dim",
        )
        x = self.forwardsolution(sensor_embedding, x)
        # Regroup so each (sample, channel, window) is a separate 1-D sequence for SEANet.
        x = rearrange(
            x,
            "(batch nwin tok) chans dim -> (batch chans nwin) dim tok",
            batch=batch,
            nwin=nwin,
            tok=tok,
        )
        x = self.seanet_decoder(x)
        # Split the fused batch back into sample, channel, and window dimensions.
        return rearrange(
            x,
            "(batch chans nwin) 1 wlen -> batch chans nwin wlen",
            batch=batch,
            nwin=nwin,
        )


# Public BrainTokenizer model (VQ-VAE pretraining module)


class BrainTokenizer(EEGModuleMixin, nn.Module):
    r"""BrainOmni VQ-VAE tokenizer for EEG and MEG signals.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    ``BrainTokenizer`` is the pretrainable tokenizer backbone described in
    [brainomni]_.  It encodes raw multi-channel EEG/MEG windows into discrete
    neural tokens via a SEANet-based convolutional encoder, residual vector
    quantization (RVQ), and a cross-attention decoder that reconstructs the
    original waveform.  Sensor geometry (position + orientation) is derived
    from ``chs_info`` at initialisation and used by the ``_SensorEmbedding``
    to build geometry-aware channel embeddings.

    .. rubric:: Pretrained Weights

    .. important::

        Pre-trained weights for ``BrainTokenizer`` are available on
        `HuggingFace <https://huggingface.co/OpenTSLab/BrainOmni>`_
        (see the repository for the *tiny* and *base* tokenizer checkpoints).
        Load with ``BrainTokenizer.from_pretrained("OpenTSLab/BrainOmni")``.

    Parameters
    ----------
    emb_dim : int
        Embedding dimensionality throughout the model.
    n_neuro : int
        Number of virtual "neural source" tokens produced by the encoder.
    window_length : int
        Length (samples) of each analysis window fed to SEANet.
    n_filters : int
        Base number of filters in SEANet.
    ratios : tuple of int
        Downsampling ratios in SEANet (encoder) / upsampling ratios
        (decoder).  Total compression equals the product of ratios.
    kernel_size : int
        Kernel size for SEANet convolutions.
    last_kernel_size : int
        Kernel size for the first and last SEANet conv layer.
    tokenizer_num_heads : int
        Number of attention heads in the cross-attention tokenizer blocks.
    codebook_dim : int
        Projected dimension inside each VQ codebook.
    codebook_size : int
        Number of entries per VQ codebook.
    num_quantizers : int
        Number of residual VQ stages.
    rotation_trick : bool
        Whether to use the rotation trick when updating codebook entries.
    quantize_optimize_method : str
        Codebook optimisation method (``"ema"``; the only currently
        supported option).
    drop_prob : float
        Dropout probability applied inside the tokenizer attention blocks.
    activation : type[nn.Module]
        Accepted for braindecode API symmetry; the VQ-VAE uses fixed
        activations baked into the pretrained weights.

    Notes
    -----
    Input is expected at 256 Hz.  When using pretrained weights, apply
    sensor-type-wise group z-score normalisation (separately for EEG, MAG,
    and GRAD channels) before forwarding.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025.
       Online: https://arxiv.org/abs/2505.18185
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        emb_dim: int = 256,
        n_neuro: int = 16,
        window_length: int = 512,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (8, 4, 2),
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        tokenizer_num_heads: int = 4,
        codebook_dim: int = 256,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        rotation_trick: bool = True,
        quantize_optimize_method: str = "ema",
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.SELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, input_window_seconds

        if int(round(self.sfreq)) != 256:
            warnings.warn(
                f"BrainOmni pretrained weights expect sfreq=256 Hz, got "
                f"{self.sfreq}. Use only for training from scratch.",
                UserWarning,
            )

        pos, sensor_type = _geometry_from_chs_info(self.chs_info)
        self.register_buffer("pos", torch.from_numpy(pos))
        self.register_buffer("sensor_type", torch.from_numpy(sensor_type))

        self.emb_dim = emb_dim
        self.n_neuro = n_neuro
        self.window_length = window_length
        self.drop_prob = drop_prob
        self.activation = activation

        self.sensor_embed = _SensorEmbedding(emb_dim)
        self.encoder = _TokenizerEncoder(
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=emb_dim,
            n_head=tokenizer_num_heads,
            dropout=drop_prob,
            n_neuro=n_neuro,
        )
        self.quantizer = _ResidualVQ(
            dim=emb_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )
        self.final_layer = _TokenizerDecoder(
            n_dim=emb_dim,
            n_head=tokenizer_num_heads,
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=drop_prob,
        )

    def _unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0) -> torch.Tensor:
        """Slice ``x`` (B, C, T) into ``(B, C, N, window_length)`` windows."""
        step = round(self.window_length * (1 - overlap_ratio))
        pad = max(self.window_length - x.shape[-1], 0)  # ensure >= 1 window
        if step < self.window_length:  # overlapping: fill the trailing window
            remainder = (x.shape[-1] + pad - self.window_length) % step
            pad += (step - remainder) % step
        if pad:
            x = F.pad(x, (0, pad))
        return x.unfold(dimension=-1, size=self.window_length, step=step)

    def _sensor_embedding(self, batch_size: int) -> torch.Tensor:
        # pos/sensor_type are registered buffers -> already on the model device
        pos = self.pos.unsqueeze(0).expand(batch_size, -1, -1)
        stype = self.sensor_type.unsqueeze(0).expand(batch_size, -1)
        return self.sensor_embed(pos, stype)

    def _encode_quantize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        """Unfold -> sensor-embed -> encode -> RVQ. Returns (feat_q, indices, commit, sensor_embedding)."""
        xu = self._unfold(x, overlap_ratio)  # (B, C, N, L)
        se = self._sensor_embedding(xu.shape[0])  # (B, C, emb_dim)
        feat = self.encoder(xu, se)  # (B, n_neuro, N, T, D)
        feat_q, indices, commit = self.quantizer(feat)
        return feat_q, indices, commit, se

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VQ-VAE reconstruction. ``x`` (B, C, T) -> reconstruction (B, C, T)."""
        feat_q, _, _, se = self._encode_quantize(x)
        recon = self.final_layer(feat_q, se)  # (B, C, N, L)
        # _unfold pads to a multiple of window_length, so N*L >= T; trim to original.
        recon = recon.reshape(recon.shape[0], recon.shape[1], -1)
        return recon[..., : x.shape[-1]]

    def encode_decode(self, x: torch.Tensor):
        """Reconstruction pass returning ``(recon (B,C,T), commitment_loss, indices)`` for training loops."""
        feat_q, indices, commit, se = self._encode_quantize(x)
        recon = self.final_layer(feat_q, se)
        # _unfold pads to a multiple of window_length, so N*L >= T; trim to original.
        recon = recon.reshape(recon.shape[0], recon.shape[1], -1)[..., : x.shape[-1]]
        return recon, commit, indices

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        """Encode ``x`` into quantised features and discrete codebook indices.

        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape ``(B, C, T)``.
        overlap_ratio : float
            Fraction of overlap between consecutive windows (0 = no overlap).

        Returns
        -------
        feat : torch.Tensor
            Quantised features, shape ``(B, n_neuro, N*T_enc, emb_dim)``.
        indices : torch.Tensor
            Codebook indices, shape ``(B, n_neuro, N*T_enc, num_quantizers)``.

        Notes
        -----
        This method internally switches the tokenizer to eval mode before
        encoding so that VQ codebooks are never EMA-updated (which would
        corrupt them) and dropout is disabled for deterministic output.  The
        prior training/eval mode is restored when the call returns, so calling
        ``tokenize`` while the module is in train mode is safe::

            model.train()
            feat, indices = model.tokenize(x)  # codebooks frozen, mode restored
        """
        was_training = self.training
        self.eval()  # freeze VQ codebooks (no EMA) and disable dropout
        try:
            feat_q, indices, _, _ = self._encode_quantize(x, overlap_ratio)
        finally:
            if was_training:
                self.train()
        # Merge windows (nwin) and per-window tokens (tok) into a single token sequence.
        feat = rearrange(
            feat_q, "batch chans nwin tok dim -> batch chans (nwin tok) dim"
        )
        indices = rearrange(
            indices, "batch chans nwin tok nquant -> batch chans (nwin tok) nquant"
        )
        return feat, indices


# Public BrainOmni classifier model


class BrainOmni(EEGModuleMixin, nn.Module):
    r"""BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    ``BrainOmni`` is the downstream classifier described in [brainomni]_.
    It wraps a frozen :class:`BrainTokenizer` backbone with a stack of
    spatial-temporal factored attention blocks (``_STBlock``) and a linear
    classification head, matching the ``DownstreamModel`` architecture from
    the published BrainOmni codebase.

    The tokenizer slides over the input with stride
    ``window_length * (1 - overlap_ratio)`` to produce a temporal sequence
    of neural-source embeddings, which the transformer then processes.
    During fine-tuning only the transformer ``blocks`` and the final
    classification head are trainable; the :class:`BrainTokenizer` backbone
    (VQ codebooks and all convolutional layers) is frozen in eval mode via
    a :meth:`train` override.

    .. rubric:: Pretrained Weights

    .. important::

        Pre-trained weights for both the tokenizer backbone and the full
        downstream model are available on
        `HuggingFace <https://huggingface.co/OpenTSLab/BrainOmni>`_
        (*tiny* and *base* variants).  Load with
        ``BrainOmni.from_pretrained("OpenTSLab/BrainOmni")``.

    Parameters
    ----------
    emb_dim : int
        Tokenizer embedding dimension.
    n_neuro : int
        Number of virtual neural-source tokens (spatial dimension).
    window_length : int
        Analysis window length (samples) fed to the tokenizer.
    overlap_ratio : float
        Fractional overlap between consecutive tokenizer windows.
        Note: ``BrainOmni`` defaults to ``0.25`` here, whereas
        :meth:`BrainTokenizer.tokenize` defaults to ``0.0`` — features will
        differ if the two are mixed manually without aligning this value.
    n_filters : int
        Base filter count for the SEANet encoder inside the tokenizer.
    ratios : tuple of int
        Downsampling ratios for the SEANet encoder.
    kernel_size : int
        Conv kernel size in the SEANet encoder.
    last_kernel_size : int
        Kernel size for the first and last SEANet conv layer.
    tokenizer_num_heads : int
        Attention heads in the tokenizer cross-attention blocks.
    codebook_dim : int
        Projected dimension inside each VQ codebook.
    codebook_size : int
        Number of entries per VQ codebook.
    num_quantizers : int
        Number of residual VQ stages.
    rotation_trick : bool
        Whether to use the rotation-trick STE for codebook updates.
    quantize_optimize_method : str
        Codebook optimisation strategy (``"ema"``).
    lm_dim : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads in the transformer blocks.
    depth : int
        Total number of transformer blocks.  Note: the last block is
        excluded from ``encode`` / ``forward`` (kept for checkpoint parity).
    drop_prob : float
        Dropout probability throughout the model.
    activation : type[nn.Module]
        Activation used in the classification head.

    Notes
    -----
    Input is expected at 256 Hz.  When using pretrained weights, apply
    sensor-type-wise group z-score normalisation (separately for EEG, MAG,
    and GRAD channels) before forwarding.  The :class:`BrainTokenizer`
    backbone (including VQ codebooks) is always kept in eval mode and its
    parameters receive no gradients during fine-tuning; only ``blocks`` and
    ``final_layer`` are trainable.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025.
       Online: https://arxiv.org/abs/2505.18185
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        emb_dim: int = 256,
        n_neuro: int = 16,
        window_length: int = 512,
        overlap_ratio: float = 0.25,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (8, 4, 2),
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        tokenizer_num_heads: int = 4,
        codebook_dim: int = 256,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        rotation_trick: bool = True,
        quantize_optimize_method: str = "ema",
        lm_dim: int = 256,
        num_heads: int = 8,
        depth: int = 12,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.SELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, input_window_seconds

        self.lm_dim = lm_dim
        self.n_neuro = n_neuro
        self.overlap_ratio = overlap_ratio
        self.drop_prob = drop_prob
        self.activation = activation

        self.tokenizer = BrainTokenizer(
            chs_info=self.chs_info,
            n_times=self.n_times,
            sfreq=self.sfreq,
            emb_dim=emb_dim,
            n_neuro=n_neuro,
            window_length=window_length,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            tokenizer_num_heads=tokenizer_num_heads,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.projection: nn.Module = (
            nn.Linear(emb_dim, lm_dim) if emb_dim != lm_dim else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [_STBlock(lm_dim, num_heads, drop_prob, causal=False) for _ in range(depth)]
        )
        self._head_in = n_neuro * lm_dim
        self.final_layer: nn.Module = nn.Identity()  # overwritten by reset_head
        self.reset_head(self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        """Re-create the classification head for ``n_outputs`` classes."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Sequential(
            nn.Dropout(self.drop_prob),
            nn.Linear(self._head_in, self.lm_dim),
            self.activation(),
            nn.Linear(self.lm_dim, n_outputs),
        )

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize ``x`` and project to ``lm_dim``.

        Returns ``(B, n_neuro, W, lm_dim)`` with gradients stopped at the
        tokenizer boundary.
        """
        feat, _ = self.tokenizer.tokenize(x, overlap_ratio=self.overlap_ratio)
        neuro = self.tokenizer.encoder.neuros.detach().to(feat.dtype)
        feat = feat + neuro.view(1, feat.shape[1], 1, -1)
        return self.projection(feat)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone embedding: blocks[:-1] then L2-normalize (parity w/ upstream).

        Returns ``(B, n_neuro, W, lm_dim)``.
        """
        h = self._tokens(x)
        for block in self.blocks[:-1]:
            h = block(h)
        return F.normalize(h, p=2.0, dim=-1, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify ``x`` (B, C, T) -> logits (B, n_outputs)."""
        feat = self.encode(x)  # (B, n_neuro, W, lm_dim)
        feat = feat.mean(dim=2)  # pool over tokens -> (B, n_neuro, lm_dim)
        feat = feat.reshape(feat.shape[0], -1)  # (B, n_neuro * lm_dim)
        return self.final_layer(feat)

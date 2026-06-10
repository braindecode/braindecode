# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
#
# Ported from https://github.com/OpenTSLab/BrainOmni (MIT License, 2025 OpenTSLab).
# SEANet/conv/LSTM submodules derive from Meta's EnCodec (MIT License).
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Classic weight_norm is used (not torch.nn.utils.parametrizations.weight_norm)
# to keep state-dict keys as ``conv.weight_g`` / ``conv.weight_v``, matching
# the upstream EnCodec/BrainOmni checkpoint format.  torch >=2.0 emits a
# deprecation warning for this API — that is expected and intentional.
from torch.nn.utils import weight_norm  # noqa: F401

from braindecode.models.base import EEGModuleMixin

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
    # EEG already returned above; only MEG is valid here.
    if kind != _FIFF_MEG_CH:
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


class _RMSNorm(nn.Module):
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
        # The ``rotate`` cache is complex64; a real-dtype cast (e.g. .half())
        # would silently discard its imaginary part. Regenerate it afterwards
        # on the (possibly new) device of ``freqs``.
        prev_dtype = self.rotate.dtype
        result = super()._apply(fn, recurse=recurse)
        if self.rotate.dtype != prev_dtype:
            self._set_rotate_cache(self.max_seq_len_cache)
        return result

    def reshape_for_broadcast(self, x: torch.Tensor):
        """x: Batch seq n_head d_head  /  rotate: seq dim."""
        B, T, H, D = x.shape
        if T > self.max_seq_len_cache:
            self._set_rotate_cache(T)
        rotate = self.rotate[:T, :]
        assert H * D == rotate.shape[1], (
            f"RoPE cache shape mismatch: H={H}, D={D}, rotate.shape[1]={rotate.shape[1]}"
        )
        return rearrange(rotate, "T (H D)-> T H D", H=H).unsqueeze(0)

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

        # SDPA does not gate dropout on training mode, so gate it explicitly here;
        # dropout is disabled in eval for deterministic inference.
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

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
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
        # Match upstream exactly (spatial first, temporal second) — the halves
        # are intentionally swapped relative to the input split. Required for
        # pretrained-weight parity; see BrainOmni/model_utils/attn.py.
        return torch.cat([xs, xt], dim=-1)


# ---------------------------------------------------------------------------
# SEANet conv helpers (ported from Meta EnCodec / BrainOmni model_utils/conv.py)
# ---------------------------------------------------------------------------

# Supported parametrisation norms for this port (weight_norm path only).
_CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])


def _apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    """Apply weight_norm or return module unchanged."""
    assert norm in _CONV_NORMALIZATIONS, f"Unsupported norm: {norm!r}"
    if norm == "weight_norm":
        # Classic weight_norm keeps state-dict keys ``conv.weight_g`` /
        # ``conv.weight_v``, required for checkpoint compatibility.
        return weight_norm(module)  # type: ignore[arg-type]
    return module


def _get_norm_module(
    module: nn.Module,  # noqa: ARG001 – unused but kept for API symmetry
    causal: bool = False,  # noqa: ARG001
    norm: str = "none",
    **norm_kwargs,  # noqa: ARG002
) -> nn.Module:
    """Return a post-conv normalisation module.

    Always ``nn.Identity()`` for this port: ``_CONV_NORMALIZATIONS`` only
    covers ``"none"`` and ``"weight_norm"`` (applied pre-conv via
    ``_apply_parametrization_norm``). Parameters ``module`` and ``causal``
    are retained for API symmetry with the full EnCodec implementation.
    """
    assert norm in _CONV_NORMALIZATIONS, f"Unsupported norm: {norm!r}"
    return nn.Identity()


def _get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Compute extra right-padding so that the last convolution window is full."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def _pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "zero",
    value: float = 0.0,
) -> torch.Tensor:
    """Thin wrapper around F.pad that handles reflect padding on very short inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def _unpad1d(x: torch.Tensor, paddings: tuple[int, int]) -> torch.Tensor:
    """Remove padding from both ends of a 1-D tensor."""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class _NormConv1d(nn.Module):
    """Conv1d with optional weight-norm and post-conv normalisation."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = _apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class _NormConvTranspose1d(nn.Module):
    """ConvTranspose1d with optional weight-norm and post-conv normalisation."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.convtr = _apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = _get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtr(x)
        x = self.norm(x)
        return x


class _SConv1d(nn.Module):
    """Conv1d with built-in asymmetric / causal padding and optional normalisation."""

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
        norm: str = "none",
        norm_kwargs: dict | None = None,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "_SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        norm_kwargs = norm_kwargs or {}
        self.conv = _NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        extra_padding = _get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            x = _pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = _pad1d(
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
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.convtr = _NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert 0.0 <= self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = _unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = _unpad1d(y, (padding_left, padding_right))
        return y


# ---------------------------------------------------------------------------
# SLSTM (ported from Meta EnCodec / BrainOmni model_utils/lstm.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SEANet residual block + encoder/decoder
# (ported from Meta EnCodec / BrainOmni model_utils/seanet.py)
# ---------------------------------------------------------------------------


class _SEANetResnetBlock(nn.Module):
    """Residual block from the SEANet model.

    Parameters
    ----------
    dim : int
        Feature dimension.
    kernel_sizes : list of int
        Kernel sizes for each conv in the block.
    dilations : list of int
        Dilations for each conv in the block.
    activation : str
        ``nn`` activation class name (e.g. ``"ELU"``).
    activation_params : dict or None
        Keyword arguments for the activation constructor.
    norm : str
        ``"weight_norm"`` or ``"none"``.
    norm_params : dict or None
        Extra kwargs forwarded to the norm module constructor.
    causal : bool
        Whether to use causal convolutions.
    pad_mode : str
        Padding mode (``"reflect"`` or ``"zero"``).
    compress : int
        Channel compression factor in the hidden layers.
    true_skip : bool
        If ``True``, use identity skip; otherwise use a 1×1 conv skip.
    """

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
    """SEANet encoder.

    Parameters
    ----------
    channels : int
        Input signal channels (1 for mono waveform).
    dimension : int
        Output latent dimension.
    n_filters : int
        Base channel width.
    n_residual_layers : int
        Number of residual blocks per downsampling stage.
    ratios : list of int
        Downsampling ratios (applied in *reverse* order internally).
    activation : str
        Activation class name in ``torch.nn``.
    activation_params : dict or None
        Keyword arguments for the activation constructor.
    norm : str
        ``"weight_norm"`` or ``"none"``.
    norm_params : dict or None
        Extra kwargs for the norm module.
    kernel_size : int
        Kernel size of the first convolution.
    last_kernel_size : int
        Kernel size of the final projection convolution.
    residual_kernel_size : int
        Kernel size used inside residual blocks.
    dilation_base : int
        Dilation base; residual block ``j`` uses ``dilation_base**j``.
    causal : bool
        Causal convolutions only.
    pad_mode : str
        Padding mode.
    true_skip : bool
        Identity vs. conv skip in residual blocks.
    compress : int
        Channel compression in residual blocks.
    lstm : int
        Number of LSTM layers after the convolutional stack (0 = none).
    bidirectional : bool
        Bidirectional LSTM.
    """

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
                    _SEANetResnetBlock(
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
    """SEANet decoder.

    Parameters
    ----------
    channels : int
        Output signal channels.
    dimension : int
        Latent input dimension.
    n_filters : int
        Base channel width.
    n_residual_layers : int
        Residual blocks per upsampling stage.
    ratios : list of int
        Upsampling ratios (applied in order).
    activation : str
        Activation class name in ``torch.nn``.
    activation_params : dict or None
        Keyword arguments for the activation constructor.
    final_activation : str or None
        Optional final activation after the output convolution.
    final_activation_params : dict or None
        Kwargs for the final activation constructor.
    norm : str
        ``"weight_norm"`` or ``"none"``.
    norm_params : dict or None
        Extra kwargs for the norm module.
    kernel_size : int
        Kernel size of the first convolution.
    last_kernel_size : int
        Kernel size of the final projection convolution.
    residual_kernel_size : int
        Kernel size inside residual blocks.
    dilation_base : int
        Dilation base.
    causal : bool
        Causal convolutions only.
    pad_mode : str
        Padding mode.
    true_skip : bool
        Identity vs. conv skip in residual blocks.
    compress : int
        Channel compression in residual blocks.
    lstm : int
        Number of LSTM layers before the upsampling stack (0 = none).
    trim_right_ratio : float
        Right-trim ratio for causal transposed convolutions.
    bidirectional : bool
        Bidirectional LSTM.
    """

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
                    _SEANetResnetBlock(
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


# =============================================================================
# Residual Vector Quantization (EMA path only, no deepspeed / einx)
#
# Ported from BrainOmni/model_utils/vq.py (MIT License, 2025 OpenTSLab) and
# BrainOmni/model_utils/module.py.
#
# Key changes vs upstream:
#   - deepspeed.comm replaced by no-op stubs (single-process only).
#   - einx / vector_quantize_pytorch.rotate_to replaced by local helpers.
#   - kmeans warm-start omitted; uniform init always (safe because pretrained
#     checkpoints overwrite all codebook buffers on load).
#   - SimVQ class dropped entirely (not used in BrainTokenizer/BrainOmni).
# =============================================================================


# ---------------------------------------------------------------------------
# Distributed no-ops (deepspeed not imported)
# ---------------------------------------------------------------------------


def _broadcast_tensors(tensors, src_rank=0):
    return  # single-process: nothing to sync


def _all_reduce_tensors(tensors, op=None):
    return  # single-process: nothing to sync


# ---------------------------------------------------------------------------
# EMA / smoothing helpers
# ---------------------------------------------------------------------------


def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _laplace_smoothing(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return (x + epsilon) / (x.sum() + epsilon * len(x))


# ---------------------------------------------------------------------------
# Codebook initialisation
# ---------------------------------------------------------------------------


def _uniform_init(*shape: int) -> torch.Tensor:
    # kaiming_uniform_ (scaled uniform); buffers are overwritten on checkpoint load
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


# ---------------------------------------------------------------------------
# Rotation-trick straight-through estimator (Fifty et al. 2024)
# Local replacement for vector_quantize_pytorch.rotate_to.
# Reference: https://arxiv.org/abs/2410.06424 §4.2
# ---------------------------------------------------------------------------


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / den.clamp(min=eps)


def _efficient_rotation_trick_transform(
    u: torch.Tensor, q: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    # Section 4.2 of https://arxiv.org/abs/2410.06424
    e = e.unsqueeze(1)  # (b, 1, d)
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()  # (b, d)
    out = (
        e
        - 2 * (e @ w.unsqueeze(2) @ w.unsqueeze(1))
        + 2 * (e @ u.unsqueeze(2).detach() @ q.unsqueeze(1).detach())
    )
    return out.squeeze(1)  # (b, d)


def _rotate_to(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    # Rotation-trick STE (Fifty et al. 2024): rotate src onto tgt direction,
    # gradient flows as a (detached) rotation+scale. Local replacement for
    # vector_quantize_pytorch.rotate_to.
    orig_shape = src.shape
    src = src.reshape(-1, orig_shape[-1])
    tgt = tgt.reshape(-1, orig_shape[-1])
    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)
    rotated_tgt = _efficient_rotation_trick_transform(
        _safe_div(src, norm_src), _safe_div(tgt, norm_tgt), src
    )
    rotated = rotated_tgt * _safe_div(norm_tgt, norm_src).detach()
    return rotated.reshape(orig_shape)


# ---------------------------------------------------------------------------
# EuclideanCodebook
# ---------------------------------------------------------------------------


class _EuclideanCodebook(nn.Module):
    """EMA-updated Euclidean codebook.

    Buffer names/shapes intentionally match the upstream BrainOmni checkpoint
    so that ``load_state_dict`` works without remapping.

    Notes
    -----
    The kmeans warm-start from upstream is omitted; uniform initialisation is
    always used (pretrained weights overwrite these buffers on load).
    """

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

        embed = _uniform_init(codebook_size, dim)
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
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, expired)
        _broadcast_tensors(list(self.buffers()))

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
        x = rearrange(x, "... d -> (...) d")
        embed_ind = self.quantize(x)
        return embed_ind.view(*shape[:-1])

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return self.dequantize(embed_ind)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x = rearrange(x, "... d -> (...) d")
        # No init call (kmeans warm-start omitted; uniform init always).
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.dequantize(embed_ind).type(dtype)

        if self.training:
            self.expire_codes_(x)
            one_hot_sum = embed_onehot.sum(0)
            _all_reduce_tensors([one_hot_sum])
            _ema_inplace(self.cluster_size, one_hot_sum, self.decay)
            embed_sum = (embed_onehot.t() @ x).to(torch.float32)
            _all_reduce_tensors([embed_sum])
            _ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = (
                _laplace_smoothing(self.cluster_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


# ---------------------------------------------------------------------------
# VectorQuantization
# ---------------------------------------------------------------------------


class _VectorQuantization(nn.Module):
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
        self._codebook = _EuclideanCodebook(
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


# ---------------------------------------------------------------------------
# ResidualVQ
# ---------------------------------------------------------------------------


class _ResidualVQ(nn.Module):
    """Residual vector quantisation (EMA path, no quantize-dropout)."""

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
                _VectorQuantization(
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
        residual = x
        all_indices: list[torch.Tensor] = []
        # Intentionally calls each layer's forward (vq(residual)) rather than
        # vq.encode, matching upstream RVQ.encode. The two diverge only when
        # codebook_dim != emb_dim; using forward is upstream-faithful here.
        for vq in self.layers:
            quantized, indices, _ = vq(residual)
            residual = residual - quantized.detach()
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)


# ---------------------------------------------------------------------------
# BrainQuantizer
# ---------------------------------------------------------------------------


class _BrainQuantizer(nn.Module):
    """Normalise → RVQ wrapper used by BrainTokenizer.

    Input ``x`` is L2-normalised before quantisation, matching upstream.
    """

    def __init__(
        self,
        n_dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
    ):
        super().__init__()
        self.rvq = _ResidualVQ(
            dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.rvq.encode(F.normalize(x, p=2.0, dim=-1))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, W, D) token features.

        Returns:
            x_q:    (B, W, D) quantised features.
            indices:(B, W, num_quantizers) codebook indices.
            loss:   scalar commit loss.
        """
        x_q, indices, loss = self.rvq(F.normalize(x, p=2.0, dim=-1))
        return x_q, indices, loss


# ---------------------------------------------------------------------------
# BrainSensorModule
# ---------------------------------------------------------------------------


class _BrainSensorModule(nn.Module):
    """Map sensor positions + types to a per-channel embedding.

    Args:
        n_dim: Embedding dimensionality.

    Inputs:
        pos:         (B, C, 6) float — position + orientation.
        sensor_type: (B, C)    long  — 0=EEG, 1=MAG, 2=GRAD.

    Returns:
        (B, C, n_dim) normalised sensor embedding.
    """

    def __init__(self, n_dim: int) -> None:
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2),
            nn.SELU(),
            nn.Linear(n_dim // 2, n_dim),
        )
        self.aggregate_mlp = _FeedForward(n_dim, 0.0)
        self.norm = _RMSNorm(n_dim)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding_layer(pos)
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# ForwardSolution — sensor_embedding (query) × neurons (key/value)
# ---------------------------------------------------------------------------


class _ForwardSolution(nn.Module):
    """Cross-attention: sensor embeddings query neural token key/values.

    Args:
        n_dim:   Model dimensionality.
        n_head:  Number of attention heads (must divide n_dim).
        dropout: Attention dropout probability.

    Inputs:
        sensor_embedding: (B, C, n_dim)
        neurons:          (B, T_kv, n_dim)

    Returns:
        (B, C, n_dim)
    """

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
        B, C, _ = sensor_embedding.shape
        kv = self.kv(neurons)
        k, v = torch.split(kv, split_size_or_sections=self.n_dim, dim=-1)
        q = rearrange(sensor_embedding, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)
        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)
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
        output = output.view(B, C, -1)
        return self.proj(output)


# ---------------------------------------------------------------------------
# _BackwardSolution: neuros (query) attend to sensor+feature keys / channel values
# ---------------------------------------------------------------------------


class _BackwardSolution(nn.Module):
    """Cross-attention: neural queries × pre-projected sensor keys/values.

    Args:
        n_dim:   Model dimensionality.
        n_head:  Number of attention heads.
        dropout: Attention dropout probability.

    Inputs:
        neuros: (B, N_q, n_dim) — learned neural queries.
        k:      (B, T_kv, n_dim) — pre-projected keys from sensor space.
        x:      (B, T_kv, n_dim) — raw values projected by self.v.

    Returns:
        (B, N_q, n_dim)
    """

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
        B, N_q, _ = neuros.shape
        q = rearrange(neuros, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(
            k, "B T (H D) -> B H T D", H=self.n_head
        )  # reshape pre-projected keys for multi-head attention
        v = rearrange(self.v(x), "B T (H D) -> B H T D", H=self.n_head)
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
        output = output.view(B, N_q, -1)
        return self.proj(output)


# ---------------------------------------------------------------------------
# BrainTokenizerEncoder
# ---------------------------------------------------------------------------


class _BrainTokenizerEncoder(nn.Module):
    """Encode multi-channel EEG/MEG windows into neural token sequences.

    Pipeline: SEANet conv encode → spatial collapse via BackwardSolution.

    Args:
        n_filters:        Base filter count for SEANet.
        ratios:           Temporal downsampling ratios (list[int]).
        kernel_size:      SEANet conv kernel size.
        last_kernel_size: SEANet final conv kernel size.
        n_dim:            Embedding dimensionality.
        n_head:           Attention heads (must divide n_dim).
        dropout:          Attention dropout.
        n_neuro:          Number of learned neural queries (spatial collapse).

    Inputs:
        x:                (B, C, N, L) — batch, channel, n_splits, window.
        sensor_embedding: (B, C, n_dim) — output of _BrainSensorModule.

    Returns:
        (B, n_neuro, N, T, n_dim) where T = L / prod(ratios).
        Note: the second axis is n_neuro (channels collapse to neuro queries).
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
        B, C, N, L = x.shape
        x = rearrange(x, "B C N L -> (B C N) 1 L")
        x = self.seanet_encoder(x)
        x = rearrange(x, "(B C N) D T -> B C (N T) D", B=B, C=C, N=N)
        B, C, W, _ = x.shape
        sensor_embedding = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, W, 1),
            "B C W D -> (B W) C D",
        )
        x = rearrange(x, "B C W D -> (B W) C D")
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        x = rearrange(x, "(B N T) C D -> B C (N T) D", B=B, N=N)
        return rearrange(x, "B C (N T) D -> B C N T D", N=N)


# ---------------------------------------------------------------------------
# BrainTokenizerDecoder
# ---------------------------------------------------------------------------


class _BrainTokenizerDecoder(nn.Module):
    """Decode neural token sequences back to per-channel waveforms.

    Pipeline: ForwardSolution spatial expansion → SEANet conv decode.

    Args:
        n_dim:            Embedding dimensionality.
        n_head:           Attention heads.
        n_filters:        Base filter count for SEANet.
        ratios:           Temporal upsampling ratios (list[int]).
        kernel_size:      SEANet conv kernel size.
        last_kernel_size: SEANet final conv kernel size.
        dropout:          Attention dropout.

    Inputs:
        x:                (B, C, N, T, n_dim) — encoder output (C=n_neuro).
        sensor_embedding: (B, C_orig, n_dim)  — sensor embeddings.

    Returns:
        (B, C_orig, N, L) reconstructed waveforms.
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
        B, C, N, T, D = x.shape
        x = rearrange(x, "B C N T D -> (B N T) C D")
        sensor_embedding = rearrange(
            sensor_embedding.view(B, -1, 1, 1, D).repeat(1, 1, N, T, 1),
            "B C N T D -> (B N T) C D",
        )
        x = self.forwardsolution(sensor_embedding, x)
        x = rearrange(x, "(B N T) C D -> (B C N) D T", B=B, N=N, T=T)
        x = self.seanet_decoder(x)
        return rearrange(x, "(B C N) 1 L -> B C N L", B=B, N=N)


# ---------------------------------------------------------------------------
# Public BrainTokenizer model (VQ-VAE pretraining module)
# ---------------------------------------------------------------------------


class BrainTokenizer(EEGModuleMixin, nn.Module):
    r"""BrainOmni VQ-VAE tokenizer for EEG and MEG signals.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    ``BrainTokenizer`` is the pretrainable tokenizer backbone described in
    [brainomni]_.  It encodes raw multi-channel EEG/MEG windows into discrete
    neural tokens via a SEANet-based convolutional encoder, residual vector
    quantization (RVQ), and a cross-attention decoder that reconstructs the
    original waveform.  Sensor geometry (position + orientation) is derived
    from ``chs_info`` at initialisation and used by the ``_BrainSensorModule``
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

        try:
            eff_sfreq = self.sfreq
        except ValueError:
            eff_sfreq = None
        if eff_sfreq is not None and int(round(eff_sfreq)) != 256:
            warnings.warn(
                f"BrainOmni pretrained weights expect sfreq=256 Hz, got "
                f"{eff_sfreq}. Use only for training from scratch.",
                UserWarning,
            )

        pos, sensor_type = _geometry_from_chs_info(self.chs_info)
        self.register_buffer("pos", torch.from_numpy(pos))
        self.register_buffer("sensor_type", torch.from_numpy(sensor_type))

        self.emb_dim = emb_dim
        self.n_neuro = n_neuro
        self.window_length = window_length
        self.drop_prob = drop_prob
        # activation is accepted for braindecode API symmetry; the VQ-VAE uses
        # the fixed SELU/ELU activations baked into the pretrained weights.
        self.activation = activation

        self.sensor_embed = _BrainSensorModule(emb_dim)
        self.encoder = _BrainTokenizerEncoder(
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=emb_dim,
            n_head=tokenizer_num_heads,
            dropout=drop_prob,
            n_neuro=n_neuro,
        )
        self.quantizer = _BrainQuantizer(
            n_dim=emb_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )
        # reconstruction decoder == the model's output head -> name final_layer
        self.final_layer = _BrainTokenizerDecoder(
            n_dim=emb_dim,
            n_head=tokenizer_num_heads,
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=drop_prob,
        )

    def _unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0) -> torch.Tensor:
        """Segment ``x`` (B, C, T) into windows (B, C, N, window_length)."""
        # faithful port of upstream BrainTokenizer.unfold
        if x.shape[-1] < self.window_length:
            x = F.pad(x, pad=(0, self.window_length - x.shape[-1]))
        if overlap_ratio > 0.0:
            stride = int(self.window_length * (1 - overlap_ratio))
            right_remain = (x.shape[-1] - self.window_length) % stride
            if right_remain > 0:
                x = F.pad(x, pad=(0, stride - right_remain))
        step = int(self.window_length * (1 - overlap_ratio))
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
        feat = rearrange(feat_q, "B C N T D -> B C (N T) D")
        indices = rearrange(indices, "B C N T Q -> B C (N T) Q")
        return feat, indices


# ---------------------------------------------------------------------------
# Public BrainOmni classifier model
# ---------------------------------------------------------------------------


class BrainOmni(EEGModuleMixin, nn.Module):
    r"""BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    ``BrainOmni`` is the downstream classifier described in [brainomni]_.
    It wraps a frozen :class:`BrainTokenizer` backbone with a stack of
    spatial-temporal factored attention blocks (``_SpatialTemporalAttentionBlock``) and a linear
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

        # backbone tokenizer (the public class, so state-dict keys match the
        # converted checkpoint). It derives geometry from chs_info and emits the
        # sfreq!=256 warning. The tokenizer is always frozen (see train()).
        try:
            eff_sfreq = self.sfreq
        except ValueError:
            eff_sfreq = None
        self.tokenizer = BrainTokenizer(
            chs_info=self.chs_info,
            n_times=self.n_times,
            sfreq=eff_sfreq,
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
            [
                _SpatialTemporalAttentionBlock(
                    lm_dim, num_heads, drop_prob, causal=False
                )
                for _ in range(depth)
            ]
        )
        self._head_in = n_neuro * lm_dim
        self.final_layer: nn.Module = nn.Identity()  # overwritten by reset_head
        self.reset_head(self.n_outputs)

    def train(self, mode: bool = True) -> "BrainOmni":
        """Set training mode, keeping the tokenizer permanently frozen in eval."""
        super().train(mode)
        self.tokenizer.eval()
        return self

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

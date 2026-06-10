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

        # dropout_p is passed unconditionally (matches upstream); SDPA does not
        # gate on self.training. Use model.eval() with dropout=0 for determinism.
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

# Authors: Meta Platforms, Inc. and affiliates (original)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
#
# Architecture adapted from the official implementation at
# https://github.com/facebookresearch/generic-neuromotor-interface
# released under CC BY-NC 4.0. This derivative inherits the same
# noncommercial terms and is not covered by braindecode's BSD-3 license!

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal, cast

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin

_LPadType = int | Literal["none", "steady", "full"]


def _normalize_frequency_bins(
    frequency_bins: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...] | None:
    if frequency_bins is None:
        return None

    normalized = []
    for band in frequency_bins:
        if len(band) != 2:
            raise ValueError("Each frequency bin must contain exactly two values")
        start, end = band
        if end <= start:
            raise ValueError("Frequency bin end must be greater than start")
        normalized.append((float(start), float(end)))
    return tuple(normalized)


class _ChannelwiseSTFT(nn.Module):
    """Short-time Fourier transform applied independently to every channel."""

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        window = torch.hann_window(n_fft, periodic=False)
        self.register_buffer("window", window)
        self.register_buffer("window_norm", torch.linalg.vector_norm(window))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = inputs.shape
        stft = torch.stft(
            rearrange(inputs, "batch channels time -> (batch channels) time"),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=cast(torch.Tensor, self.window),
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stft = stft / cast(torch.Tensor, self.window_norm)
        return rearrange(
            stft,
            "(batch channels) freqs time -> batch channels freqs time",
            batch=batch_size,
            channels=num_channels,
        )


class _FrequencyBandAverager(nn.Module):
    """Average frequency bins into fixed ``(low, high)`` Hz bands."""

    def __init__(
        self,
        n_fft: int,
        fs: float,
        frequency_bins: Sequence[tuple[float, float]] | None,
    ) -> None:
        super().__init__()
        self.frequency_bins = frequency_bins
        self.n_output_freqs = n_fft // 2 + 1
        if frequency_bins is not None:
            freq_masks = self._build_freq_masks(n_fft, fs, frequency_bins)
            self.register_buffer("freq_masks", freq_masks)
            self.n_output_freqs = len(frequency_bins)

    @staticmethod
    def _build_freq_masks(
        n_fft: int,
        fs: float,
        frequency_bins: Sequence[tuple[float, float]],
    ) -> torch.Tensor:
        freqs_hz = torch.fft.rfftfreq(n_fft, d=1.0 / fs)
        freq_masks = torch.stack(
            [
                torch.logical_and(freqs_hz > start, freqs_hz <= end)
                for start, end in frequency_bins
            ]
        ).to(dtype=torch.float32)
        if (freq_masks.sum(dim=1) == 0).any():
            raise ValueError("Each frequency bin must contain at least one FFT bin")
        return rearrange(freq_masks, "band freq -> 1 1 band freq 1 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frequency_bins is None:
            return x

        freq_masks = cast(torch.Tensor, self.freq_masks)
        weighted = rearrange(
            x,
            "batch time freq channels1 channels2 -> "
            "batch time 1 freq channels1 channels2",
        )
        weighted = weighted * freq_masks
        return weighted.sum(dim=3) / freq_masks.sum(dim=3)


class _SlidingWindowLast(nn.Module):
    """Unfold the last axis into overlapping windows of fixed ``size`` and ``step``.

    Input ``(..., time)`` -> output ``(..., num_windows, size)``.
    """

    def __init__(self, size: int, step: int) -> None:
        super().__init__()
        self.size = size
        self.step = step

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unfold(dimension=-1, size=self.size, step=self.step)


class _CrossSpectralDensity(nn.Module):
    """Estimate channel-pair cross-spectral density inside each MPF window."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        window_size = inputs.shape[-1]
        outputs, packed_shape = pack([inputs], "* channels window")
        outputs = (
            outputs
            @ rearrange(
                outputs.conj(), "items channels window -> items window channels"
            )
        ) / window_size
        outputs = outputs.abs().pow(2)
        return unpack(outputs, packed_shape, "* channels1 channels2")[0]


class _SPDMatrixLog(nn.Module):
    """Apply a dtype-stable SPD matrix logarithm."""

    @staticmethod
    def _eigval_floor(dtype: torch.dtype) -> float:
        return 100.0 * torch.finfo(dtype).eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(inputs)
        eigvals = eigvals.clamp_min(self._eigval_floor(eigvals.dtype)).log()
        return (eigvecs * rearrange(eigvals, "... channels -> ... 1 channels")) @ (
            rearrange(eigvecs, "... row col -> ... col row")
        )


class _MultivariatePowerFrequencyFeatures(nn.Module):
    """Convert raw multichannel signal to multivariate power frequency (MPF) features.

    Input: ``(batch, channels, time)``. Output:
    ``(batch, num_freq_bins, channels, channels, time')``: the per-band matrix
    logarithm of the cross-spectral density between channel pairs.

    Parameters
    ----------
    window_length : int
        Window length for computation of MPF features. Must be larger than ``n_fft``.
    stride : int
        Number of samples to stride between consecutive MPF windows.
    n_fft : int
        FFT size (also STFT window size).
    fft_stride : int
        Hop size of the STFT. Must divide ``stride`` and be ``<= n_fft``.
    fs : float
        Sampling frequency of the input in Hz.
    frequency_bins : sequence of (float, float) or None
        Average FFT frequencies within each ``(low, high)`` Hz bin. If ``None``,
        all FFT frequencies are returned as is.
    """

    def __init__(
        self,
        window_length: int,
        stride: int,
        n_fft: int,
        fft_stride: int,
        fs: float = 2000.0,
        frequency_bins: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        if window_length < n_fft:
            raise ValueError("window_length must be greater than n_fft")
        if fft_stride > n_fft:
            raise ValueError("fft_stride must be lower than n_fft")
        if fft_stride > stride:
            raise ValueError("stride must be greater than fft_stride")
        if stride % fft_stride != 0:
            raise ValueError("stride must be a multiple of fft_stride")

        self.window_length = window_length
        self.stride = stride
        self.n_fft = n_fft
        self.fft_stride = fft_stride
        self.fs = fs
        self.frequency_bins = _normalize_frequency_bins(frequency_bins)

        self.stft = _ChannelwiseSTFT(n_fft=self.n_fft, hop_length=self.fft_stride)
        self.frame = _SlidingWindowLast(
            size=self.window_length // self.fft_stride,
            step=self.stride // self.fft_stride,
        )
        self.csd = _CrossSpectralDensity()
        self.band_averager = _FrequencyBandAverager(
            n_fft=self.n_fft,
            fs=self.fs,
            frequency_bins=self.frequency_bins,
        )
        self.spd_log = _SPDMatrixLog()

        # amount by which output time length is shorter than input time length
        # when considering only the featurizer.
        self.left_context = self.window_length - self.fft_stride + self.n_fft - 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stft(inputs)
        outputs = self.frame(outputs)
        outputs = rearrange(
            outputs,
            "batch channels freqs windows window -> "
            "batch windows freqs channels window",
        )

        outputs = self.csd(outputs)
        outputs = self.band_averager(outputs)
        outputs = self.spd_log(outputs)

        return rearrange(
            outputs,
            "batch time freqs channels1 channels2 -> "
            "batch freqs channels1 channels2 time",
        )

    def compute_time_downsampling(self, input_lengths: torch.Tensor) -> torch.Tensor:
        cospectrum_len = 1 + (input_lengths - self.n_fft) // self.fft_stride
        return (cospectrum_len - self.window_length // self.fft_stride) // (
            self.stride // self.fft_stride
        ) + 1


class _VectorizeSymmetricMatrix(nn.Module):
    """Vectorize a symmetric matrix, keeping only adjacent off-diagonals.

    Assumes an input of 6 dimensions whose last two (``num_channels, num_channels``)
    describe a symmetric matrix. Keeps the upper-triangular values whose row-column
    distance (modulo ``num_channels`` for circular adjacency) is at most
    ``num_adjacent_cov``.
    """

    def __init__(
        self,
        num_channels: int,
        num_adjacent_cov: int | None = None,
    ) -> None:
        super().__init__()
        max_adjacent_cov = num_channels // 2
        if num_adjacent_cov is None:
            num_adjacent_cov = max_adjacent_cov
        if num_adjacent_cov > max_adjacent_cov:
            raise ValueError(
                f"num_adjacent_cov={num_adjacent_cov} must be <= {max_adjacent_cov}"
            )

        triu_row, triu_col = torch.triu_indices(num_channels, num_channels)
        adj_mask = self._get_adjacent_cov_mask(num_channels, num_adjacent_cov)
        keep = adj_mask[triu_row, triu_col] == 1
        row_indices = triu_row[keep]
        col_indices = triu_col[keep]
        flattened_matrix_indices = num_channels * row_indices + col_indices
        self.register_buffer(
            "flattened_matrix_indices",
            flattened_matrix_indices,
            persistent=False,
        )
        self.flatten_matrix = Rearrange(
            "batch time rotation freqs row col -> batch time rotation freqs (row col)"
        )

    @staticmethod
    def _get_adjacent_cov_mask(
        num_channels: int, num_adjacent_diagonals: int
    ) -> torch.Tensor:
        # 1 where |i - j| mod num_channels <= num_adjacent_diagonals (circular adjacency).
        idx = torch.arange(num_channels)
        circular_diff = (idx[:, None] - idx[None, :]) % num_channels
        is_adjacent = (circular_diff <= num_adjacent_diagonals) | (
            circular_diff >= num_channels - num_adjacent_diagonals
        )
        return is_adjacent.to(torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.flatten_matrix(inputs)
        indices = cast(torch.Tensor, self.flattened_matrix_indices)
        return torch.index_select(inputs, dim=-1, index=indices)


class _RotationInvariantMPFMLP(nn.Module):
    """Rotation-invariant projection of MPF features.

    Rotates the MPF cross-channel matrix by ``offsets`` along the (circular)
    channel dimension, vectorizes each rotation, passes each through the same
    MLP, then averages the projections.

    Assumes **circular channel adjacency** (as in a 16-electrode EMG armband).
    This is not meaningful for arbitrary EEG montages.
    """

    def __init__(
        self,
        num_channels: int,
        num_freqs: int,
        hidden_dims: Sequence[int],
        offsets: Sequence[int] = (-1, 0, 1),
        num_adjacent_cov: int = 3,
        activation: type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty")
        self.offsets = list(offsets)
        self.activation = activation()
        self.vectorize = _VectorizeSymmetricMatrix(
            num_channels=num_channels,
            num_adjacent_cov=num_adjacent_cov,
        )
        self.flatten_features = Rearrange(
            "batch time rotation freqs covariance -> "
            "batch time rotation (freqs covariance)"
        )
        self.fully_connected_layers = nn.ModuleList()
        dim = num_freqs * min(
            num_channels * (num_adjacent_cov + 1),
            num_channels * (num_channels + 1) // 2,
        )
        for hidden_dim in hidden_dims:
            self.fully_connected_layers.append(nn.Linear(dim, hidden_dim))
            dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.stack(
            [
                inputs.roll(shifts=[offset, offset], dims=[2, 3])
                for offset in self.offsets
            ],
            dim=-1,
        )
        x = rearrange(
            x,
            "batch freqs channels1 channels2 time rotation -> "
            "batch time rotation freqs channels1 channels2",
        )
        x = self.vectorize(x)
        x = self.flatten_features(x)
        for layer in self.fully_connected_layers:
            x = self.activation(layer(x))
        x = reduce(x, "batch time rotation features -> batch time features", "mean")
        return rearrange(x, "batch time features -> batch features time")


class _AxesMask(nn.Module):
    """Samples and applies a filler mask along one or more axes (SpecAugment).

    The same mask of length drawn from ``[0, max_mask_length]`` is applied to every
    axis in ``axes``; all masked axes must have the same length.
    """

    def __init__(
        self,
        max_mask_length: int,
        axes: tuple[int, ...],
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        for axis in axes:
            if axis <= 0:
                raise ValueError("Cannot mask batch dim")
        self.max_mask_length = max_mask_length
        self.axes = axes
        self.mask_value = mask_value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs
        N = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype
        data_length = inputs.size(self.axes[0])
        max_mask_length = min(self.max_mask_length, data_length)
        value = torch.rand(N, device=device, dtype=dtype) * max_mask_length
        min_value = torch.rand(N, device=device, dtype=dtype) * (data_length - value)
        mask_start = min_value.long()
        mask_end = mask_start + value.long()
        for _ in range(inputs.ndim - 1):
            mask_start = mask_start.unsqueeze(-1)
            mask_end = mask_end.unsqueeze(-1)
        idx = torch.arange(0, data_length, device=device, dtype=dtype)
        mask_idx = (idx >= mask_start) & (idx < mask_end)
        x = inputs
        for axis in self.axes:
            if x.size(axis) != data_length:
                raise ValueError("All axes to mask must have the same length")
            x = x.transpose(axis, -1)
            x = x.masked_fill(mask_idx, self.mask_value)
            x = x.transpose(-1, axis)
        return x


class _RepeatedRandomMask(nn.Module):
    """Apply one mask layer a random number of times during training."""

    def __init__(self, max_num_masks: int, mask: _AxesMask) -> None:
        super().__init__()
        self.max_num_masks = max_num_masks
        self.mask = mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n_masks = int(torch.randint(self.max_num_masks + 1, size=()).item())
        x = inputs
        for _ in range(n_masks):
            x = self.mask(x)
        return x


class _MaskAug(nn.Module):
    """SpecAugment on MPF features (time and frequency masking, train-only).

    Parameters
    ----------
    max_num_masks : sequence of int
        Max number of masks per dim (order matches ``dims``).
    max_mask_lengths : sequence of int
        Max length of each mask per dim (order matches ``dims``).
    dims : str
        Ordered coordinates to mask. One of ``"T"``, ``"F"``, ``"C"``, or any
        combination (e.g. ``"TF"``).
    axes_by_coord : dict[str, tuple[int, ...]] or None
        Mapping of supported dims to tensor axes. Defaults to
        ``{'N':(0,), 'F':(1,), 'C':(2, 3), 'T':(4,)}``.
    mask_value : float
        Filler value for masked positions.
    """

    def __init__(
        self,
        max_num_masks: Sequence[int],
        max_mask_lengths: Sequence[int],
        dims: str = "TF",
        axes_by_coord: dict[str, tuple[int, ...]] | None = None,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        if len(max_num_masks) != len(dims):
            raise ValueError("max_num_masks length must match dims length")
        if len(max_mask_lengths) != len(dims):
            raise ValueError("max_mask_lengths length must match dims length")
        if axes_by_coord is None:
            axes_by_coord = {"N": (0,), "F": (1,), "C": (2, 3), "T": (4,)}

        self.masks = nn.ModuleList()
        for dim in "CFT":
            if dim not in dims or dim not in axes_by_coord:
                continue
            dim_idx = dims.index(dim)
            self.masks.append(
                _RepeatedRandomMask(
                    max_num_masks=max_num_masks[dim_idx],
                    mask=_AxesMask(
                        max_mask_length=max_mask_lengths[dim_idx],
                        axes=axes_by_coord[dim],
                        mask_value=mask_value,
                    ),
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs
        x = inputs
        for mask in self.masks:
            x = mask(x)
        return x


class _TakeEvery(nn.Module):
    """Select every ``step`` samples along the last axis."""

    def __init__(self, step: int) -> None:
        super().__init__()
        self.step = step

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[..., :: self.step]


class _Window(nn.Module):
    """Causal sliding-window extraction for ``(batch, time, ...)`` tensors.

    The layer pads the left context, unfolds the time axis with ``stride``, and
    appends a final axis containing the sampled window. ``dilation`` is applied
    on that final axis, so callers receive windows with exactly ``kernel_size``
    effective samples.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        lpad: _LPadType = 0,
    ) -> None:
        super().__init__()
        self.receptive_field = 1 + dilation * (kernel_size - 1)
        self.stride = stride
        self.state_size = self.receptive_field - self.stride
        if self.receptive_field < self.stride:
            raise ValueError(
                f"receptive_field={self.receptive_field} < stride={self.stride} "
                "is not supported"
            )
        if isinstance(lpad, int):
            self.lpad: int = lpad
        else:
            if lpad not in {"none", "steady", "full"}:
                raise ValueError(f"Invalid lpad={lpad!r}")
            self.lpad = {
                "none": 0,
                "steady": self.state_size,
                "full": self.receptive_field - 1,
            }[lpad]
        if self.lpad >= self.receptive_field:
            raise ValueError(
                f"lpad={self.lpad} should be < receptive_field={self.receptive_field}"
            )
        self.extra_left_context = self.receptive_field - 1 - self.lpad
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.dilation_selector = _TakeEvery(dilation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.state_size > 0:
            buffered = F.pad(inputs, (0, 0, self.state_size, 0), "constant", 0)
        else:
            buffered = inputs
        windows = buffered.unfold(
            dimension=1, size=self.receptive_field, step=self.stride
        )
        return self.dilation_selector(windows)


class _Residual(nn.Module):
    """Residual wrapper that slices the skip connection to match child output."""

    def __init__(
        self, child: nn.Module, dropout: float = 0.0, weight: float = 1.0
    ) -> None:
        super().__init__()
        self.child = child
        self.extra_left_context = int(getattr(child, "extra_left_context"))
        self.stride = int(getattr(child, "stride"))
        self.dropout = nn.Dropout(dropout)
        self.weight = weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs[:, self.extra_left_context :: self.stride]
        x = self.child(inputs)
        x = x + self.weight * self.dropout(residual)
        return x


class _SlicedSequential(nn.Sequential):
    """``nn.Sequential`` that tracks cumulative ``extra_left_context`` and ``stride``.

    Used to compute the time downsampling factor of the full conformer encoder
    without running a dummy forward.
    """

    def __init__(self, *modules) -> None:
        super().__init__(*modules)
        self.extra_left_context, self.stride = self._get_extra_left_context_and_stride(
            list(self)
        )

    @staticmethod
    def _get_extra_left_context_and_stride(seq) -> tuple[int, int]:
        left, stride = 0, 1
        for mod in seq:
            if hasattr(mod, "extra_left_context") and hasattr(mod, "stride"):
                left += int(getattr(mod, "extra_left_context")) * stride
                stride *= int(getattr(mod, "stride"))
        return left, stride


class _MultiHeadAttention(nn.Module):
    """Causal multi-head attention over a sliding window.

    Causality is enforced by windowing: the attention at step ``t`` sees the
    previous ``window_size`` samples only (via the zero left-pad plus a
    triangular attention mask).
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        window_size: int,
        stride: int = 1,
        lpad: _LPadType = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.op = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_window = _Window(
            kernel_size=window_size,
            stride=stride,
            lpad=lpad,
        )
        self.lpad = self.attention_window.lpad
        if self.lpad > 0 and stride > 1:
            raise NotImplementedError(
                "MultiHeadAttention only supports unit stride when lpad > 0"
            )
        self.extra_left_context = self.attention_window.extra_left_context
        self.stride = self.attention_window.stride
        self._init_and_register_attn_mask()

    def _init_and_register_attn_mask(self) -> None:
        # 1s indicate positions that are masked out (PyTorch MHA convention
        # is 0 to attend). The mask is anti-diagonal-triangular so that
        # causal histories are preserved in each window.
        attn_mask = torch.ones(self.window_size, self.window_size, dtype=torch.bool)
        attn_mask = attn_mask.triu(diagonal=1).flip(dims=[1])
        self.register_buffer("attn_mask", attn_mask)

    def _get_attn_mask(self) -> torch.Tensor:
        return cast(torch.Tensor, self.attn_mask)

    def _windows(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = rearrange(inputs, "batch time ... -> batch time (...)")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected flattened input dim {self.input_dim}, got {inputs.shape[-1]}"
            )
        windows = self.attention_window(inputs)
        return rearrange(
            windows, "batch time channels window -> batch time window channels"
        )

    def _attn_params(self, windows: torch.Tensor):
        windows = rearrange(
            windows, "batch time window channels -> (batch time) window channels"
        )
        query = windows[:, -1]
        # Match the upstream key/value ordering from transpose(...).reshape(...).
        kv = rearrange(
            windows, "batch_time window channels -> batch_time (channels window)"
        )
        kv = rearrange(
            kv,
            "batch_time (window channels) -> batch_time window channels",
            window=self.window_size,
            channels=self.input_dim,
        )
        return query, kv, kv

    def _attn_mask_op(
        self,
        windows: torch.Tensor,
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        T_out = windows.shape[1]
        warmup_idx = slice(
            max(t_start, self.extra_left_context),
            min(t_end, self.window_size - 1),
            self.stride,
        )
        attn_mask = self._get_attn_mask()
        attn_mask_warmup = attn_mask[warmup_idx]
        T_out_warmup = len(attn_mask_warmup)
        T_out_steady = T_out - T_out_warmup
        if T_out_steady < 0:
            raise ValueError("Inconsistent attention mask length")
        attn_mask_steady = attn_mask[-1:, :].expand(T_out_steady, -1)
        return torch.cat([attn_mask_warmup, attn_mask_steady], dim=0)

    def _full_attn_mask(
        self,
        windows: torch.Tensor,
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        batch_size, output_time = windows.shape[:2]
        attn_mask = self._attn_mask_op(windows, t_start, t_end)
        return repeat(
            attn_mask,
            "time window -> (batch time heads) window",
            batch=batch_size,
            heads=self.op.num_heads,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        windows = self._windows(inputs)
        query, key, value = self._attn_params(windows)
        batch_size, input_time = inputs.shape[:2]
        attn_mask = self._full_attn_mask(windows, 0, input_time)
        output, _ = self.op(
            query=rearrange(query, "batch_time channels -> batch_time 1 channels"),
            key=key,
            value=value,
            attn_mask=rearrange(
                attn_mask, "batch_time_heads window -> batch_time_heads 1 window"
            ),
        )
        output = rearrange(
            output, "(batch time) 1 channels -> batch time 1 channels", batch=batch_size
        )
        return output


class _Conv1d(nn.Module):
    """1D convolution for ``(batch, time, channels)`` tensors.

    PyTorch's :class:`~torch.nn.Conv1d` expects channel-first input. This wrapper
    keeps the conformer blocks in time-major layout while exposing the stride and
    left-context metadata used by :class:`_SlicedSequential`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.to_channels_first = Rearrange("batch time channels -> batch channels time")
        self.to_time_first = Rearrange("batch channels time -> batch time channels")
        self.receptive_field = 1 + dilation * (kernel_size - 1)
        self.state_size = self.receptive_field - stride
        self.stride = stride
        self.extra_left_context = self.receptive_field - 1
        self.net = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.to_time_first(self.net(self.to_channels_first(inputs)))


def _time_reduction_layer(stride: int, lpad: _LPadType = 0) -> nn.Module:
    """Frame stacking: ``(B, T, C)`` -> ``(B, T', C*stride)``."""
    return _SlicedSequential(
        _Window(kernel_size=stride, stride=stride, lpad=lpad),
        Rearrange("batch time channels stride -> batch time (stride channels)"),
    )


def _conformer_encoder_block(
    input_dim: int,
    ffn_dim: int,
    kernel_size: int,
    stride: int,
    num_heads: int,
    attn_window_size: int,
    attn_lpad: _LPadType = "steady",
    drop_prob: float = 0.0,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Module:
    """Single conformer encoder block: FF -> MHA -> Conv -> FF -> LayerNorm."""
    ff_block1 = _Residual(
        _SlicedSequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ffn_dim),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(drop_prob),
        ),
        weight=0.5,
    )
    ff_block2 = _Residual(
        _SlicedSequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ffn_dim),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(drop_prob),
        ),
        weight=0.5,
    )

    if attn_window_size > 0:
        attn_block: nn.Module = _Residual(
            _SlicedSequential(
                nn.LayerNorm(input_dim),
                Rearrange("batch time channels -> batch time 1 channels"),
                _MultiHeadAttention(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    window_size=attn_window_size,
                    stride=1,
                    lpad=attn_lpad,
                    dropout=drop_prob,
                ),
                Rearrange("batch time 1 channels -> batch time channels"),
                nn.Dropout(drop_prob),
            )
        )
    else:
        attn_block = nn.Identity()

    if kernel_size > 0:
        conv_block: nn.Module = _Residual(
            _SlicedSequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 2 * input_dim),
                nn.GLU(dim=-1),
                _Conv1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=input_dim,
                ),
                nn.LayerNorm(input_dim),
                activation(),
                nn.Linear(input_dim, input_dim),
                nn.Dropout(drop_prob),
            )
        )
    else:
        conv_block = nn.Identity()

    return _SlicedSequential(
        OrderedDict(
            {
                "ff_block1": ff_block1,
                "attn_block": attn_block,
                "conv_block": conv_block,
                "ff_block2": ff_block2,
                "layer_norm": nn.LayerNorm(input_dim),
            }
        )
    )


def _as_layer_list(val, num_layers: int, name: str):
    """Return a per-layer list, validating explicit per-layer schedules."""
    if isinstance(val, (list, tuple)):
        out = list(val)
        if len(out) != num_layers:
            raise ValueError(f"{name} length {len(out)} != num_layers {num_layers}")
        return out
    return [val] * num_layers


def _conformer_encoder(
    input_dim: int,
    ffn_dim: int,
    kernel_size,
    stride,
    num_heads: int,
    attn_window_size,
    num_layers: int,
    attn_lpad="steady",
    drop_prob: float = 0.0,
    activation: type[nn.Module] = nn.SiLU,
) -> _SlicedSequential:
    kernel_size = _as_layer_list(kernel_size, num_layers, "kernel_size")
    attn_window_size = _as_layer_list(attn_window_size, num_layers, "attn_window_size")
    attn_lpad = _as_layer_list(attn_lpad, num_layers, "attn_lpad")
    if isinstance(stride, (list, tuple)):
        stride = list(stride)
        if len(stride) != num_layers:
            raise ValueError("stride length must match num_layers")
    else:
        # scalar: apply only to the last block (entire encoder downsamples by stride)
        stride = [1] * (num_layers - 1) + [stride]

    seq = [
        _conformer_encoder_block(
            input_dim=input_dim,
            ffn_dim=ffn_dim,
            kernel_size=kernel_size[i],
            stride=stride[i],
            num_heads=num_heads,
            attn_window_size=attn_window_size[i],
            attn_lpad=attn_lpad[i],
            drop_prob=drop_prob,
            activation=activation,
        )
        for i in range(num_layers)
    ]
    return _SlicedSequential(*seq)


def _build_handwriting_encoder(
    in_dim: int,
    input_dim: int,
    ffn_dim: int,
    kernel_size,
    stride,
    num_heads: int,
    attn_window_size,
    num_layers: int,
    drop_prob: float,
    time_reduction_stride: int,
    activation: type[nn.Module],
) -> _SlicedSequential:
    """Build the encoder stack up to (but not including) the final readout.

    The final classification layer and optional log-softmax live on the outer
    :class:`GenericNeuromotorInterface` so that ``reset_head`` and the
    ``final_layer`` introspection convention of braindecode work uniformly.
    """
    seq: list[nn.Module] = []
    dim = in_dim
    if time_reduction_stride > 1:
        seq.extend(
            [
                _time_reduction_layer(stride=time_reduction_stride, lpad="none"),
                nn.Linear(dim * time_reduction_stride, input_dim),
            ]
        )
        dim = input_dim
    if dim != input_dim:
        seq.append(nn.Linear(dim, input_dim))

    seq.append(
        _conformer_encoder(
            input_dim=input_dim,
            ffn_dim=ffn_dim,
            kernel_size=kernel_size,
            stride=stride,
            num_heads=num_heads,
            attn_window_size=attn_window_size,
            num_layers=num_layers,
            attn_lpad="steady",
            drop_prob=drop_prob,
            activation=activation,
        )
    )
    return _SlicedSequential(*seq)


_DEFAULT_MPF_FREQUENCY_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 50.0),
    (30.0, 100.0),
    (100.0, 225.0),
    (225.0, 375.0),
    (375.0, 700.0),
    (700.0, 1000.0),
)
# Paper's 15-layer handwriting config. Only used when ``num_layers == 15`` and
# ``stride`` / ``attn_window_size`` are left unset.
_PAPER_CONFORMER_STRIDE_15: tuple[int, ...] = (1, 1, 1, 1, 2) * 2 + (1,) * 5
_PAPER_CONFORMER_ATTN_WINDOW_15: tuple[int, ...] = (16,) * 10 + (8,) * 5
_PAPER_NUM_LAYERS: int = len(_PAPER_CONFORMER_STRIDE_15)


def _resolve_conformer_stride(
    stride: int | Sequence[int] | None, num_layers: int
) -> int | Sequence[int]:
    """Resolve the default stride schedule.

    - ``None`` + ``num_layers == 15`` -> the paper's 15-layer schedule.
    - ``None`` otherwise -> ``1`` everywhere except the last block (``2``),
      i.e. a single 2x downsampling at the end of the encoder.
    - anything else -> passed through (validated later).
    """
    if stride is not None:
        return stride
    if num_layers == _PAPER_NUM_LAYERS:
        return _PAPER_CONFORMER_STRIDE_15
    return 2  # scalar applied only to the last block by _conformer_encoder


def _resolve_conformer_attn_window(
    attn_window: int | Sequence[int] | None, num_layers: int
) -> int | Sequence[int]:
    """Resolve the default attention-window schedule.

    - ``None`` + ``num_layers == 15`` -> the paper's 15-layer schedule.
    - ``None`` otherwise -> ``16`` for every block.
    """
    if attn_window is not None:
        return attn_window
    if num_layers == _PAPER_NUM_LAYERS:
        return _PAPER_CONFORMER_ATTN_WINDOW_15
    return 16


class GenericNeuromotorInterface(EEGModuleMixin, nn.Module):
    r"""Generic neuromotor interface for handwriting from Meta (2025) [gni2025]_.

    :bdg-info:`Attention/Transformer` :bdg-success:`Convolution`
    :bdg-primary:`CTC`

    .. rubric:: Architectural Overview

    Conformer-based surface-EMG-to-character decoder for the handwriting task of
    Meta's generic neuromotor interface (CTRL-labs at Reality Labs, Nature 2025).
    It takes raw 16-channel surface EMG recorded at the wrist and emits a
    per-token score sequence that feeds a CTC loss or decoder
    [graves2006ctc]_. The upstream repository
    (``facebookresearch/generic-neuromotor-interface``) has one architecture
    per task for 1-DOF wrist control, discrete gestures, and handwriting;
    only the handwriting / conformer head is ported here.

    The pipeline mirrors the paper's Methods (Fig. 1 / Extended Data Fig. 6):

    1. :class:`_MultivariatePowerFrequencyFeatures` (MPF): channel-wise STFT,
       pairwise cross-spectral density (CSD), averaging into ``(low, high)``
       Hz bands, and an SPD matrix logarithm (Barachant et al. 2012; pyRiemann
       [pyriemann]_). Produces a tensor of shape
       ``(batch, num_freq_bins, n_chans, n_chans, time')``.
    2. :class:`_MaskAug`: SpecAugment [park2019specaug]_ — time and frequency
       masking of the MPF feature map, active only during training.
    3. :class:`_RotationInvariantMPFMLP`: rolls the cross-channel matrix by
       ``invariance_offsets`` along the (circular) channel dimension,
       vectorizes the upper triangle keeping only ``num_adjacent_cov``
       off-diagonals, applies a small MLP to each rotation, then **averages
       over rotations**. The average enforces (approximate) invariance to
       rigid rotations of the 16-electrode armband around the wrist.
    4. Conformer encoder [gulati2020conformer]_: 15 blocks by default, each
       FF(½) → causal windowed MHA → depthwise conv → FF(½) → LayerNorm.
       Per-block stride ``2`` at blocks 5 and 10, attention window ``16`` for
       blocks 1–10 and ``8`` for blocks 11–15 (the paper's schedule).
       Attention is restricted to a fixed local window ending at the current
       frame, so the model runs as a streaming causal decoder.
    5. Linear readout to ``n_outputs`` classes, optionally followed by
       :class:`~torch.nn.LogSoftmax`.

    .. rubric:: Hardware, signal and training corpus

    The upstream sEMG-RD research wristband has 48 electrode pins arranged as
    **16 bipolar channels** aligned with the proximal–distal forearm axis, a
    2 kHz sample rate, a 2.46 μVrms noise floor, and an analog front-end with
    a 20 Hz high-pass and 850 Hz low-pass. Before featurization the raw
    signal is rescaled by ``2.46e-6`` (to unit noise s.d.) and digitally
    high-passed at 40 Hz (4th-order Butterworth) to suppress motion artifacts.

    The published handwriting decoder was trained on recordings from
    **6,627 participants** (≈1 h 15 min each) prompted to "write" text
    sampled from Simple English Wikipedia, the Google Schema-guided Dialogue
    dataset and Reddit, in three postures (seated on surface, seated on leg,
    standing on leg). Participants wrote letters, digits, words and phrases;
    spaces were either implicit or prompted by a right-dash token that the
    participant produced with a right index swipe. Training sizes scale
    geometrically from 25 to 6,527 participants; validation and test sets
    hold 50 participants each.

    .. rubric:: MPF featurizer (paper defaults)

    ``sEMG (2 kHz)`` →
    ``STFT(n_fft=64 samples / 32 ms, hop=10 samples / 5 ms)`` →
    per-pair complex cross-spectrum → squared magnitude, band-averaged into
    6 bins, then **matrix-log** on each 16×16 SPD matrix, produced every
    ``mpf_stride = 40 samples (20 ms)`` over a ``mpf_window_length = 160
    samples (80 ms)`` window. Output rate: 50 Hz before the conformer's
    ``time_reduction_stride`` and the 2× internal strides.

    The paper's frequency bins are non-overlapping (``0–62.5``, ``62.5–125``,
    ``125–250``, ``250–375``, ``375–687.5``, ``687.5–1000`` Hz), but the
    upstream training config — matched by :data:`_DEFAULT_MPF_FREQUENCY_BINS`
    — uses slightly overlapping bins (``0–50``, ``30–100``, ``100–225``,
    ``225–375``, ``375–700``, ``700–1000`` Hz); the code default reproduces
    the released checkpoints.

    .. rubric:: Training recipe (paper values, not defaults of this class)

    - **Loss**: CTC [graves2006ctc]_ with FastEmit regularization
      [fastemit2021]_ to reduce streaming latency.
    - **Vocabulary**: lowercase ``[a-z]``, digits ``[0-9]``, punctuation
      ``[,.?'!]`` and four control gestures (``space``, ``dash``,
      ``backspace``, ``pinch``); the deployed networks used
      ``vocab_size = 100`` (the default) to reserve blank / unused slots.
      Greedy CTC decoding (collapse repeats) was used at test time.
    - **Optimizer**: AdamW, ``weight_decay = 5e-2``.
    - **Learning rate**: cosine annealing from ``6e-4`` (1 M-parameter model)
      or ``3e-4`` (60 M) with a 1,500-step warmup and ``min_lr = 0``.
    - **Batching**: global batch size 512 (= 32 processes × 16), prompts
      zero-padded to the longest in the batch; gradient clipping at norm
      ``0.1``; 200 epochs. Training the largest model took ≈4 d 17 h on
      4 × NVIDIA A10G GPUs.
    - **Augmentation**: SpecAugment on the MPF features (time / frequency
      masks; ``mask_max_num_masks=(3, 2)``, ``mask_max_lengths=(5, 1)``)
      plus random circular channel rotations of ``{-1, 0, +1}``.

    Reported closed-loop performance: **20.9 WPM** on held-out naive users
    (n = 20), compared with 25.1 WPM on a pen-and-paper baseline and 36 WPM
    on a mobile keyboard; personalization with 20 min of data improves
    offline CER by ≈16 %.

    .. rubric:: Output shape and CTC usage

    The forward pass returns a tensor of shape ``(batch, T_out, n_outputs)``,
    which is the natural layout for CTC. ``T_out`` is the downsampled
    emission sequence length and can be obtained from the input length via
    :meth:`compute_output_lengths`. For :class:`~torch.nn.CTCLoss`, move the
    time dimension first: ``emissions.transpose(0, 1)``.

    .. warning::

        The rotation-invariant MLP assumes **circular channel adjacency** (the
        16-electrode EMG armband used in the paper). For arbitrary EEG
        montages, the rotation invariance is not meaningful and this model
        should not be used as-is.

    .. warning::

        **License — noncommercial use only.** This module is a derivative of
        Meta's reference implementation and is released under
        `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_, the
        same license as the upstream repository. The paper itself is
        distributed under CC BY-NC-ND 4.0. Neither is covered by
        braindecode's BSD-3 license, and both must not be used in commercial
        products or services. Using the pretrained weights carries the same
        restriction.

    .. versionadded:: 1.4

    Parameters
    ----------
    n_outputs : int
        Vocabulary size for CTC. Defaults to ``100`` (handwriting charset).
    n_chans : int
        Number of EMG channels. Defaults to ``16`` (one armband).
    sfreq : float
        Sampling frequency in Hz. Defaults to ``2000``.
    mpf_window_length : int
        MPF window length in samples.
    mpf_stride : int
        MPF frame stride in samples.
    mpf_n_fft : int
        STFT window / FFT size.
    mpf_fft_stride : int
        STFT hop size. Must divide ``mpf_stride`` and be ``<= mpf_n_fft``.
    mpf_frequency_bins : sequence of (float, float) or None
        ``(low, high)`` Hz bands to average the cross-spectrum over. If ``None``,
        all FFT frequency bins are used.
    mask_max_num_masks : sequence of int
        Max number of SpecAugment masks per dim (order matches ``mask_dims``).
    mask_max_lengths : sequence of int
        Max mask length per dim (order matches ``mask_dims``).
    mask_dims : str
        Axes to mask, among ``"CFT"``. Defaults to ``"TF"``.
    mask_value : float
        Filler value for masked regions.
    invariance_hidden_dims : sequence of int
        Hidden layer sizes of the per-rotation MLP. Output feature dim is
        ``invariance_hidden_dims[-1]``.
    invariance_offsets : sequence of int
        Circular channel rotations to average over.
    num_adjacent_cov : int
        Number of adjacent off-diagonals of the cross-channel cov matrix to keep.
    conformer_input_dim : int
        Conformer embedding dimension ``D``.
    conformer_ffn_dim : int
        Feed-forward hidden dim inside each block.
    conformer_kernel_size : int or sequence of int
        Depthwise-conv kernel size per block.
    conformer_stride : int, sequence of int, or None
        Depthwise-conv stride per block. As a scalar, applied only to the last
        block (entire encoder downsamples by ``stride``); as a list of length
        ``conformer_num_layers``, applied per block. When ``None`` (default),
        resolves to the paper's 15-layer schedule if ``conformer_num_layers==15``
        and to ``2`` (a single 2x downsampling at the end) otherwise.
    conformer_num_heads : int
        Number of attention heads.
    conformer_attn_window_size : int, sequence of int, or None
        Attention receptive field per block. When ``None`` (default), resolves
        to the paper's 15-layer schedule if ``conformer_num_layers==15`` and to
        ``16`` (uniform) otherwise.
    conformer_num_layers : int
        Number of conformer blocks.
    drop_prob : float
        Dropout probability applied throughout the conformer (FFN, conv and
        attention blocks).
    time_reduction_stride : int
        Frame-stacking stride applied **before** the conformer. ``1`` disables it.
    log_softmax : bool
        If ``True``, apply :func:`torch.nn.functional.log_softmax` to the
        emissions. Disabled by default (braindecode models return logits).
    activation : type of nn.Module
        Activation class used inside the conformer feed-forward and convolution
        blocks. Defaults to :class:`torch.nn.SiLU`.
    invariance_activation : type of nn.Module
        Activation class used inside the rotation-invariant MLP. Defaults to
        :class:`torch.nn.LeakyReLU`.
    n_times : int, optional
        Forwarded to :class:`EEGModuleMixin`.
    input_window_seconds : float, optional
        Forwarded to :class:`EEGModuleMixin`.
    chs_info : list of dict, optional
        Forwarded to :class:`EEGModuleMixin`.

    Examples
    --------
    Load Meta's pretrained handwriting checkpoint
    (`download script`_ in the upstream repo)::

        import torch
        from braindecode.models import GenericNeuromotorInterface

        ckpt = torch.load("model_checkpoint.ckpt", weights_only=False)
        sd = {k[len("network."):]: v for k, v in ckpt["state_dict"].items()}

        model = GenericNeuromotorInterface(n_times=32000, log_softmax=True)
        # load_state_dict applies the class-level ``mapping`` for upstream keys.
        model.load_state_dict(sd, strict=True)

    .. _download script: https://github.com/facebookresearch/generic-neuromotor-interface#download-the-data-and-models

    References
    ----------
    .. [gni2025] CTRL-labs at Reality Labs (Kaifosh, P., Reardon, T. R. et al.),
        2025. A generic non-invasive neuromotor interface for human-computer
        interaction. Nature 645, 702–710.
        https://doi.org/10.1038/s41586-025-09255-w.
        Code: https://github.com/facebookresearch/generic-neuromotor-interface
    .. [gulati2020conformer] Gulati, A. et al., 2020. Conformer: convolution-
        augmented transformer for speech recognition. Proc. Interspeech,
        5036–5040.
    .. [graves2006ctc] Graves, A., Fernández, S., Gomez, F., Schmidhuber, J.,
        2006. Connectionist temporal classification: labelling unsegmented
        sequence data with recurrent neural networks. Proc. ICML, 369–376.
    .. [park2019specaug] Park, D. S. et al., 2019. SpecAugment: a simple data
        augmentation method for automatic speech recognition. Proc.
        Interspeech, 2613–2617.
    .. [fastemit2021] Yu, J. et al., 2021. FastEmit: low-latency streaming
        ASR with sequence-level emission regularization. Proc. ICASSP.
    .. [pyriemann] Barachant, A. et al. pyRiemann: Biosignals classification
        with Riemannian geometry. https://github.com/pyRiemann/pyRiemann
    """

    mapping = {
        "featurizer.window": "featurizer.stft.window",
        "featurizer.window_normalization_factor": "featurizer.stft.window_norm",
        "featurizer.freq_masks": "featurizer.band_averager.freq_masks",
        "conformer.3.weight": "final_layer.weight",
        "conformer.3.bias": "final_layer.bias",
    }

    def __init__(
        self,
        n_outputs: int = 100,
        n_chans: int = 16,
        sfreq: float = 2000.0,
        # MPF featurizer
        mpf_window_length: int = 160,
        mpf_stride: int = 40,
        mpf_n_fft: int = 64,
        mpf_fft_stride: int = 10,
        mpf_frequency_bins: Sequence[Sequence[float]] | None = (
            _DEFAULT_MPF_FREQUENCY_BINS
        ),
        # SpecAugment
        mask_max_num_masks: Sequence[int] = (3, 2),
        mask_max_lengths: Sequence[int] = (5, 1),
        mask_dims: str = "TF",
        mask_value: float = 0.0,
        # Rotation-invariant MLP
        invariance_hidden_dims: Sequence[int] = (64,),
        invariance_offsets: Sequence[int] = (-1, 0, 1),
        num_adjacent_cov: int = 8,
        # Conformer
        conformer_input_dim: int = 64,
        conformer_ffn_dim: int = 128,
        conformer_kernel_size: int | Sequence[int] = 8,
        conformer_stride: int | Sequence[int] | None = None,
        conformer_num_heads: int = 4,
        conformer_attn_window_size: int | Sequence[int] | None = None,
        conformer_num_layers: int = 15,
        drop_prob: float = 0.1,
        time_reduction_stride: int = 2,
        log_softmax: bool = False,
        activation: type[nn.Module] = nn.SiLU,
        invariance_activation: type[nn.Module] = nn.LeakyReLU,
        # Standard braindecode args
        n_times=None,
        input_window_seconds=None,
        chs_info=None,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, sfreq, n_times, input_window_seconds, chs_info

        self.log_softmax = log_softmax
        mpf_frequency_bins = _normalize_frequency_bins(mpf_frequency_bins)
        n_mpf_freqs = (
            len(mpf_frequency_bins)
            if mpf_frequency_bins is not None
            else mpf_n_fft // 2 + 1
        )

        self.featurizer = _MultivariatePowerFrequencyFeatures(
            window_length=mpf_window_length,
            stride=mpf_stride,
            n_fft=mpf_n_fft,
            fft_stride=mpf_fft_stride,
            fs=self.sfreq,
            frequency_bins=mpf_frequency_bins,
        )

        self.specaug = _MaskAug(
            max_num_masks=list(mask_max_num_masks),
            max_mask_lengths=list(mask_max_lengths),
            dims=mask_dims,
            mask_value=mask_value,
        )

        self.rotation_invariant_mlp = _RotationInvariantMPFMLP(
            num_channels=self.n_chans,
            num_freqs=n_mpf_freqs,
            hidden_dims=list(invariance_hidden_dims),
            offsets=list(invariance_offsets),
            num_adjacent_cov=num_adjacent_cov,
            activation=invariance_activation,
        )
        self.features_to_sequence = Rearrange(
            "batch features time -> batch time features"
        )

        self.conformer = _build_handwriting_encoder(
            in_dim=invariance_hidden_dims[-1],
            input_dim=conformer_input_dim,
            ffn_dim=conformer_ffn_dim,
            kernel_size=conformer_kernel_size,
            stride=_resolve_conformer_stride(conformer_stride, conformer_num_layers),
            num_heads=conformer_num_heads,
            attn_window_size=_resolve_conformer_attn_window(
                conformer_attn_window_size, conformer_num_layers
            ),
            num_layers=conformer_num_layers,
            drop_prob=drop_prob,
            time_reduction_stride=time_reduction_stride,
            activation=activation,
        )

        # Slice describing the valid emission region (used by CTC length computation).
        self.output_slice: slice = slice(
            self.conformer.extra_left_context, -1, self.conformer.stride
        )

        # Final classification head (kept as a top-level module so that
        # ``reset_head`` and braindecode's "last child is final_layer" convention
        # work uniformly).
        self.final_layer = nn.Linear(conformer_input_dim, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Raw multi-channel input of shape ``(batch, n_chans, n_times)``.

        Returns
        -------
        emissions : torch.Tensor
            Shape ``(batch, T_out, n_outputs)``. Log-probabilities if
            ``log_softmax=True``, otherwise logits.
        """
        x = self.featurizer(x)
        x = self.specaug(x)
        x = self.rotation_invariant_mlp(x)
        x = self.features_to_sequence(x)
        x = self.conformer(x)
        x = self.final_layer(x)
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1)
        return x

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new number of outputs."""
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        self._n_outputs = n_outputs

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute the valid emission length for each input sequence.

        This is the length that should be passed to :class:`~torch.nn.CTCLoss`
        as ``input_lengths``.

        Parameters
        ----------
        input_lengths : torch.Tensor
            Integer tensor of shape ``(batch,)`` holding the input time
            lengths in samples.

        Returns
        -------
        torch.Tensor
            Integer tensor of shape ``(batch,)`` with emission lengths.
        """
        lengths = self.featurizer.compute_time_downsampling(input_lengths)
        slc = self.output_slice
        lengths = (
            torch.div(lengths - slc.start - 1, slc.step, rounding_mode="trunc") + 1
        )
        return lengths.clamp_min(0)

    def get_output_shape(self) -> tuple[int, ...]:
        """Shape of ``forward`` output for a batch of size 1.

        Overrides the base implementation to explicitly construct an input with
        the requested ``n_times`` (the default dummy may be too short for the
        MPF featurizer's left-context window).
        """
        min_n_times = (
            self.featurizer.window_length
            - self.featurizer.fft_stride
            + self.featurizer.n_fft
        )
        n_times = max(self.n_times, min_n_times)
        with torch.inference_mode():
            dtype = next(self.parameters()).dtype
            device = next(self.parameters()).device
            x = torch.zeros(1, self.n_chans, n_times, dtype=dtype, device=device)
            return tuple(self.forward(x).shape)

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from mne.filter import _check_zero_phase_length, create_filter, next_fast_len
from mne.utils import logger
from torch import nn


def _smart_pad_torch(x, n_pad, pad="reflect_limited"):
    """Pad vector x."""
    n_pad = np.asarray(n_pad)
    assert n_pad.shape == (2,)
    if (n_pad == 0).all():
        return x
    elif (n_pad < 0).any():
        raise RuntimeError("n_pad must be non-negative")
    if pad == "reflect_limited":
        # need to pad with zeros if len(x) <= npad
        left_zero_pad = torch.zeros(max(n_pad[0] - len(x) + 1, 0))
        right_zero_pad = torch.zeros(max(n_pad[1] - len(x) + 1, 0))
        reflection_left = 2 * x[0]
        reflection_right = 2 * x[-1]
        padded = torch.cat(
            [
                left_zero_pad,
                reflection_left - torch.flip(x[1 : n_pad[0] + 1], dims=[0]),
                x,
                reflection_right - torch.flip(x[-n_pad[1] - 2 + 1 : -1], dims=[0]),
                right_zero_pad,
            ]
        )

        return padded

    else:
        left_pad, right_pad = n_pad.tolist()
        return torch.nn.functional.pad(x, (left_pad, right_pad), mode=pad)


def _1d_overlap_filter_torch(x, n_h, n_edge, phase, h_fft, pad, n_fft):
    """Do one-dimensional overlap-add FFT FIR filtering using PyTorch."""

    # Pad to reduce ringing
    x_ext = _smart_pad_torch(x, (n_edge, n_edge), pad)
    n_x = x_ext.shape[0]
    x_filtered = torch.zeros_like(x_ext)

    n_seg = n_fft - n_h + 1
    n_segments = (n_x + n_seg - 1) // n_seg  # Equivalent to ceil division
    shift = ((n_h - 1) // 2 if phase.startswith("zero") else 0) + n_edge

    # Actual filtering step
    for seg_idx in range(n_segments):
        start = seg_idx * n_seg
        stop = (seg_idx + 1) * n_seg
        seg = x_ext[start:stop]

        # Pad segment to length n_fft
        seg = torch.cat(
            [seg, torch.zeros(n_fft - seg.shape[0], dtype=seg.dtype, device=seg.device)]
        )

        # FFT of the segment
        x_fft = torch.fft.rfft(seg, n=n_fft)
        x_fft *= h_fft
        prod = torch.fft.irfft(x_fft, n=n_fft)

        # Overlap-add operation
        start_filt = max(0, start - shift)
        stop_filt = min(start - shift + n_fft, n_x)
        start_prod = max(0, shift - start)
        stop_prod = start_prod + stop_filt - start_filt
        x_filtered[start_filt:stop_filt] += prod[start_prod:stop_prod]

    # Remove mirrored edges and cast to original dtype
    x_filtered = x_filtered[: n_x - 2 * n_edge].type_as(x)
    return x_filtered


def _overlap_add_filter_torch(
    x,
    h,
    n_fft=None,
    phase="zero",
    pad="reflect_limited",
):
    """Filter the signal x using h with overlap-add FFTs."""
    # set up array for filtering, reshape to 2D, operate on last axis
    orig_shape = x.shape
    nchans = orig_shape[1]
    # reshaping data to 2D
    x = x.view(-1, x.shape[-1])
    # Extend the signal by mirroring the edges to reduce transient filter
    # response
    _check_zero_phase_length(len(h), phase)
    if len(h) == 1:
        return x * h**2 if phase == "zero-double" else x * h
    n_edge = max(min(len(h), x.shape[1]) - 1, 0)
    logger.debug(f"Smart-padding with:  {n_edge} samples on each edge")
    n_x = x.shape[1] + 2 * n_edge

    # Determine FFT length to use
    min_fft = 2 * len(h) - 1
    if n_fft is None:
        max_fft = n_x
        if max_fft >= min_fft:
            # cost function based on number of multiplications
            N = 2 ** np.arange(
                np.ceil(np.log2(min_fft)), np.ceil(np.log2(max_fft)) + 1, dtype=int
            )
            cost = (
                np.ceil(n_x / (N - len(h) + 1).astype(np.float64))
                * N
                * (np.log2(N) + 1)
            )

            # add a heuristic term to prevent too-long FFT's which are slow
            # (not predicted by mult. cost alone, 4e-5 exp. determined)
            cost += 4e-5 * N * n_x

            n_fft = N[np.argmin(cost)]
        else:
            # Use only a single block
            n_fft = next_fast_len(min_fft)
    logger.debug(f"FFT block length:   {n_fft}")
    if n_fft < min_fft:
        raise ValueError(
            f"n_fft is too short, has to be at least 2 * len(h) - 1 ({min_fft}), got "
            f"{n_fft}"
        )

    # Figure out if we should use CUDA
    h_fft_torch = torch.fft.rfft(h, n=n_fft)

    # Process each row separately
    saving = [None] * nchans
    for chan in range(nchans):
        saving[chan] = _1d_overlap_filter_torch(
            x[chan], len(h), n_edge, phase, h_fft=h_fft_torch, pad=pad, n_fft=n_fft
        )

    return torch.stack(saving)


class FilterBank(nn.Module):
    """Filter bank layer using MNE to create the filter.

    XXXXXXXX:

    Parameters
    ----------


    """

    def __init__(
        self,
        nchans: int,
        sfreq: int,
        band_filters: Optional[List[Tuple[float, float]]] = None,
        filter_length: str | float | int = "auto",
        l_trans_bandwidth: str | float | int = "auto",
        h_trans_bandwidth: str | float | int = "auto",
        method: str = "fir",
        phase: str = "zero",
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        pad="reflect_limited",
    ):
        super(FilterBank, self).__init__()

        if band_filters is None:
            band_filters = [(4, 8), (8, 12), (12, 30)]

        self.band_filters = band_filters
        self.n_bands = len(band_filters)
        self.pad = pad
        self.phase = phase
        self.method = method

        self.conv_layers = nn.ModuleList()
        for idx, (l_freq, h_freq) in enumerate(band_filters):
            filt = create_filter(
                data=None,
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                filter_length=filter_length,
                l_trans_bandwidth=l_trans_bandwidth,
                h_trans_bandwidth=h_trans_bandwidth,
                method=method,
                iir_params=None,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
                verbose=True,
            )

            filt_tensor = torch.from_numpy(filt).float()
            filter_length = len(filt_tensor)

            filt_tensor = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0)
            # Shape: (1, 1, filter_length)
            conv = nn.Conv1d(
                in_channels=nchans,
                out_channels=nchans,
                kernel_size=filter_length,
                stride=1,
                padding=(filter_length - 1) // 2,
                groups=nchans,
                bias=False,
            )
            conv.weight = nn.Parameter(
                filt_tensor.repeat(nchans, 1, 1), requires_grad=False
            )

            self.conv_layers.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter bank to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_points).

        Returns
        -------
        torch.Tensor
            Filtered output tensor of shape (batch_size, n_bands, filtered_time_points).
        """
        # Initialize a list to collect filtered outputs
        filtered_outputs = []
        for conv in self.conv_layers:
            filtered = conv(x)  # Shape: (batch, channels, time)
            filtered = filtered.unsqueeze(1)  # Shape: (batch, 1, channels, time)
            filtered_outputs.append(filtered)
        output = torch.cat(
            filtered_outputs, dim=1
        )  # Shape: (batch, n_bands, channels, time)

        return output


if __name__ == "__main__":
    from torch import randn

    x = randn(16, 10, 1000)
    layer = FilterBank(sfreq=256, nchans=10)
    with torch.no_grad():
        out = layer(x)

    print(out.shape)

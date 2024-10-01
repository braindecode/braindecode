from __future__ import annotations

import numpy as np
import torch

from mne.filter import _check_zero_phase_length, create_filter, next_fast_len
from mne.utils import (
    logger,
)
from torch import nn


def _smart_pad_torch(x, n_pad, pad="reflect_limited"):
    """
    Pad tensor x.

    Parameters:
    - x (torch.Tensor): Input tensor of shape (N,)
    - n_pad (tuple or list or array-like): Number of pads on each side (left, right)
    - pad (str): Padding mode ('reflect_limited' or others compatible with torch.pad)

    Returns:
    - torch.Tensor: Padded tensor
    """
    n_pad = torch.as_tensor(n_pad)
    assert n_pad.shape == (2,), "n_pad must have shape (2,)"

    if torch.all(n_pad == 0):
        return x
    elif torch.any(n_pad < 0):
        raise RuntimeError("n_pad must be non-negative")

    if pad == "reflect_limited":
        left_pad, right_pad = n_pad.tolist()

        # Calculate zero padding required if n_pad > len(x) -1
        left_zero_pad_size = max(left_pad - (x.shape[0] - 1), 0)
        right_zero_pad_size = max(right_pad - (x.shape[0] - 1), 0)

        # Effective reflection lengths
        left_reflection_len = min(left_pad, x.shape[0] - 1)
        right_reflection_len = min(right_pad, x.shape[0] - 1)

        # Reflection on the left side
        if left_reflection_len > 0:
            # Slice for reflection: x[1:left_reflection_len+1]
            reflection_left = 2 * x[0] - x[1 : left_reflection_len + 1]
            # Reverse the reflected part
            reflection_left = torch.flip(reflection_left, dims=[0])
        else:
            reflection_left = torch.tensor([], dtype=x.dtype, device=x.device)

        # Reflection on the right side
        if right_reflection_len > 0:
            # Slice for reflection: x[-2:-right_reflection_len-2:-1]
            reflection_right = 2 * x[-1] - x[-2 : -right_reflection_len - 2 : -1]
        else:
            reflection_right = torch.tensor([], dtype=x.dtype, device=x.device)

        # Zero padding
        left_zero_pad = torch.zeros(left_zero_pad_size, dtype=x.dtype, device=x.device)
        right_zero_pad = torch.zeros(
            right_zero_pad_size, dtype=x.dtype, device=x.device
        )

        # Concatenate all parts
        padded = torch.cat(
            [left_zero_pad, reflection_left, x, reflection_right, right_zero_pad]
        )

        return padded
    else:
        # For other padding modes, use torch.nn.functional.pad
        # torch.pad expects pad as (left, right)
        left_pad, right_pad = n_pad.tolist()
        return torch.nn.functional.pad(x, (left_pad, right_pad), mode=pad)


def _1d_overlap_filter_torch(x, n_h, n_edge, phase, h_fft, pad, n_fft):
    """Do one-dimensional overlap-add FFT FIR filtering using PyTorch."""

    # Pad to reduce ringing
    x_ext = _smart_pad_torch(x, (n_edge, n_edge), pad)
    # x_ext = torch.from_numpy(x_ext)
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


def _smart_pad(x, n_pad, pad="reflect_limited"):
    """Pad vector x."""
    n_pad = np.asarray(n_pad)
    assert n_pad.shape == (2,)
    if (n_pad == 0).all():
        return x
    elif (n_pad < 0).any():
        raise RuntimeError("n_pad must be non-negative")
    if pad == "reflect_limited":
        # need to pad with zeros if len(x) <= npad
        l_z_pad = np.zeros(max(n_pad[0] - len(x) + 1, 0), dtype=x.dtype)
        r_z_pad = np.zeros(max(n_pad[1] - len(x) + 1, 0), dtype=x.dtype)
        return np.concatenate(
            [
                l_z_pad,
                2 * x[0] - x[n_pad[0] : 0 : -1],
                x,
                2 * x[-1] - x[-2 : -n_pad[1] - 2 : -1],
                r_z_pad,
            ]
        )
    else:
        return np.pad(x, (tuple(n_pad),), pad)


def _1d_overlap_filter(x, n_h, n_edge, phase, h_fft, pad, n_fft):
    """Do one-dimensional overlap-add FFT FIR filtering."""
    # pad to reduce ringing
    x_ext = _smart_pad(x, (n_edge, n_edge), pad)
    n_x = len(x_ext)
    x_filtered = np.zeros_like(x_ext)

    n_seg = n_fft - n_h + 1
    n_segments = int(np.ceil(n_x / float(n_seg)))
    shift = ((n_h - 1) // 2 if phase.startswith("zero") else 0) + n_edge

    # Now the actual filtering step is identical for zero-phase (filtfilt-like)
    # or single-pass
    for seg_idx in range(n_segments):
        start = seg_idx * n_seg
        stop = (seg_idx + 1) * n_seg
        seg = x_ext[start:stop]
        seg = np.concatenate([seg, np.zeros(n_fft - len(seg))])

        x_fft = np.fft.rfft(seg, n_fft)
        x_fft *= h_fft
        prod = np.fft.irfft(x_fft, n_fft)

        start_filt = max(0, start - shift)
        stop_filt = min(start - shift + n_fft, n_x)
        start_prod = max(0, shift - start)
        stop_prod = start_prod + stop_filt - start_filt
        x_filtered[start_filt:stop_filt] += prod[start_prod:stop_prod]

    # Remove mirrored edges that we added and cast (n_edge can be zero)
    x_filtered = x_filtered[: n_x - 2 * n_edge].astype(x.dtype)
    return x_filtered


def _overlap_add_filter(
    x,
    h,
    n_fft=None,
    phase="zero",
    pad="reflect_limited",
):
    """Filter the signal x using h with overlap-add FFTs."""
    # set up array for filtering, reshape to 2D, operate on last axis
    orig_shape = x.shape
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
    new_h = np.fft.rfft(h, n=n_fft)
    x_numpy = x.cpu().numpy().astype(np.float64)

    h_fft_torch = torch.fft.rfft(torch.from_numpy(h), n=n_fft)

    # Process each row separately
    picks = list(range(orig_shape[1]))
    for p in picks:
        x_numpy[p] = _1d_overlap_filter(
            x_numpy[p], len(h), n_edge, phase, h_fft=new_h, pad=pad, n_fft=n_fft
        )
        x_filtered_torch = _1d_overlap_filter_torch(
            x[p], len(h), n_edge, phase, h_fft=h_fft_torch, pad=pad, n_fft=n_fft
        )
        x_filtered_torch_np = x_filtered_torch.numpy()

        print(np.allclose(x_numpy[p], x_filtered_torch_np, atol=1e-6))
        print(np.max(np.abs(x_numpy[p] - x_filtered_torch_np)))

    x_numpy.shape = orig_shape
    return x_numpy


class FilterBank(nn.Module):
    """Filter bank layer using MNE to create the filter.

    XXXXXXXX:

    Parameters
    ----------


    """

    def __init__(
        self,
        sfreq: int,
        band_filters=None,
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
            band_filters = [(4, 8)]

        self.n_bands = len(band_filters)

        for l_freq, h_freq in band_filters:
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
            self.filter = filt

    def forward(self, x):
        """
        :meta private:
        """
        sample = x

        x = _overlap_add_filter(x=sample, h=self.filter)

        return x


######################


#############################
if __name__ == "__main__":
    from torch import randn

    x = randn(1500, 22, 1000)
    layer = FilterBank(sfreq=250)
    with torch.no_grad():
        out = layer(x)

    print(out.shape)

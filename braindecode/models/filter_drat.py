from __future__ import annotations


from collections import Counter
from copy import deepcopy
from functools import partial
from math import gcd

import numpy as np
from scipy import fft, signal
from scipy.stats import f as fstat

from mne._fiff.pick import _picks_to_idx
from mne._ola import _COLA
from mne.cuda import (
    _fft_multiply_repeated,
    _fft_resample,
    _setup_cuda_fft_multiply_repeated,
    _setup_cuda_fft_resample,
    _smart_pad,
)
from mne.fixes import minimum_phase
from mne.parallel import parallel_func
from mne.utils import (
    _check_option,
    _check_preload,
    _ensure_int,
    _pl,
    _validate_type,
    logger,
    sum_squared,
    verbose,
    warn,
)

import torch

from torch import nn

from mne.filter import (
    create_filter,
    _prep_for_filtering,
    _check_zero_phase_length,
    next_fast_len,
)
from torchaudio.functional import convolve


def _1d_overlap_filter(x, n_h, n_edge, phase, cuda_dict, pad, n_fft):
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

        prod = _fft_multiply_repeated(seg, cuda_dict)

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
    picks=None,
    n_jobs=None,
    copy=True,
    pad="reflect_limited",
):
    """Filter the signal x using h with overlap-add FFTs."""
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
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
    n_jobs, cuda_dict = _setup_cuda_fft_multiply_repeated(n_jobs, h, n_fft)

    # Process each row separately
    picks = _picks_to_idx(len(x), picks)
    parallel, p_fun, _ = parallel_func(_1d_overlap_filter, n_jobs)
    for p in picks:
        x[p] = _1d_overlap_filter(x[p], len(h), n_edge, phase, cuda_dict, pad, n_fft)

    x.shape = orig_shape
    return x


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
                verbose=False,
            )
            self.filter = filt

    def forward(self, x):
        """
        :meta private:
        """
        sample = x.cpu().numpy().astype(np.float64)

        x = _overlap_add_filter(x=sample, h=self.filter)

        return x


if __name__ == "__main__":
    from torch import randn

    x = randn(1500, 22, 1000)
    layer = FilterBank(sfreq=250)
    with torch.no_grad():
        out = layer(x)

    print(out.shape)

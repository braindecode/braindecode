from __future__ import annotations

import numpy as np
import torch
from scipy import signal
from mne.cuda import (
    _fft_multiply_repeated,
    _setup_cuda_fft_multiply_repeated,
)
from mne.filter import _check_zero_phase_length, create_filter, next_fast_len
from mne.utils import (
    logger,
)
from torch import nn


def _fft_multiply_repeated_b(x, n_fft, cuda_dict):
    """Do FFT multiplication by a filter function (possibly using CUDA).

    Parameters
    ----------
    h_fft : 1-d array or gpuarray
        The filtering array to apply.
    x : 1-d array
        The array to filter.
    n_fft : int
        The number of points in the FFT.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    # do the fourier-domain operations
    x_fft = np.fft.rfft(x, n_fft)
    x_fft *= cuda_dict["h_fft"]
    x = np.fft.irfft(x_fft, n_fft)
    return x


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
    n_jobs=None,
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
    n_jobs, cuda_dict = _setup_cuda_fft_multiply_repeated(n_jobs, h, n_fft)

    x = x.cpu().numpy().astype(np.float64)

    # Process each row separately
    picks = list(range(orig_shape[1]))
    for p in picks:
        x[p] = _1d_overlap_filter(
            x[p], len(h), n_edge, phase, cuda_dict=cuda_dict, pad=pad, n_fft=n_fft
        )

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
        sample = x

        x = _overlap_add_filter(x=sample, h=self.filter)

        return x


######################


class FilterBankTransformer:
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, banks, fs, filter_allowance=2, axis=1, filter_type="filter"):
        self.banks = banks
        self.fs = fs
        self.filter_allowance = filter_allowance
        self.axis = axis
        self.filter_type = filter_type

    @staticmethod
    def bandpass_filter(
        data, band_filt_cut_f, fs, filter_allowance=2, axis=1, filter_type="filter"
    ):
        """
         Filter a signal using cheby2 iir filtering.

        Args:
            data: 2d/ 3d np array
                trial x channels x time
            bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
                if any value is specified as None then only one sided filtering will be performed
            fs: sampling frequency
            filtAllowance: transition bandwidth in hertz
            filtType: string, available options are 'filtfilt' and 'filter'

        Returns:
            dataOut: 2d/ 3d np array after filtering
                Data after applying bandpass filter.
        """
        a_stop = 30  # stopband attenuation
        a_pass = 3  # passband attenuation
        n_freq = fs / 2  # Nyquist frequency

        if not band_filt_cut_f[0] and (
            not band_filt_cut_f[1] or (band_filt_cut_f[1] >= fs / 2.0)
        ):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif band_filt_cut_f[0] == 0 or band_filt_cut_f[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            f_pass = band_filt_cut_f[1] / n_freq
            f_stop = (band_filt_cut_f[1] + filter_allowance) / n_freq
            # find the order
            [N, ws] = signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
            b, a = signal.cheby2(N, a_stop, f_stop, "lowpass")

        elif (band_filt_cut_f[1] is None) or (band_filt_cut_f[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            f_pass = band_filt_cut_f[0] / n_freq
            f_stop = (band_filt_cut_f[0] - filter_allowance) / n_freq
            # find the order
            [N, ws] = signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
            b, a = signal.cheby2(N, a_stop, f_stop, "highpass")
        else:
            # band-pass filter
            # print("Using bandpass filter")
            f_pass = (np.array(band_filt_cut_f) / n_freq).tolist()
            f_stop = [
                (band_filt_cut_f[0] - filter_allowance) / n_freq,
                (band_filt_cut_f[1] + filter_allowance) / n_freq,
            ]
            # find the order
            [N, ws] = signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
            b, a = signal.cheby2(N, a_stop, f_stop, "bandpass")

        if filter_type == "filtfilt":
            return signal.filtfilt(b, a, data, axis=axis)
        else:
            return signal.lfilter(b, a, data, axis=axis)

    def __call__(self, data):
        # initialize output
        filter_banked_signals = np.zeros([*data.shape, len(self.banks)])

        # repetitively filter the data.
        for i, filter_band in enumerate(self.banks):
            filter_banked_signals[:, :, i] = self.bandpass_filter(
                data,
                filter_band,
                self.fs,
                self.filter_allowance,
                self.axis,
                self.filter_type,
            )

        # remove any redundant 3rd dimension
        if len(self.banks) <= 1:
            filter_banked_signals = np.squeeze(filter_banked_signals, axis=2)

        return filter_banked_signals


#############################
if __name__ == "__main__":
    from torch import randn

    x = randn(1500, 22, 1000)
    layer = FilterBank(sfreq=250)
    with torch.no_grad():
        out = layer(x)

    print(out.shape)

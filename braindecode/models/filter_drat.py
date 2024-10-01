from __future__ import annotations

from typing import List, Optional, Tuple
import torch
from torch import nn

from mne.filter import create_filter

from braindecode.models.functions import fftconvolve


class FilterBank(nn.Module):
    """Filter bank layer using MNE to create the filter.

    This layer constructs a bank of band-specific filters using MNE's `create_filter` function
    and applies them to multi-channel time-series data. Each filter in the bank corresponds to a
    specific frequency band and is applied to all channels of the input data. The filtering is
    performed using FFT-based convolution via the `fftconvolve` function from `braindecode.models.functions`.

    The default configuration creates 9 non-overlapping frequency bands with a 4 Hz bandwidth,
    spanning from 4 Hz to 40 Hz (i.e., 4-8 Hz, 8-12 Hz, ..., 36-40 Hz). This setup is based on the
    reference: *FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface*.

    Parameters
    ----------
    n_chans : int
        Number of channels in the input signal.
    sfreq : int
        Sampling frequency of the input signal in Hz.
    band_filters : Optional[List[Tuple[float, float]]], default=None
        List of frequency bands as (low_freq, high_freq) tuples. Each tuple defines the frequency range
        for one filter in the bank. If not provided, defaults to 9 non-overlapping bands with 4 Hz
        bandwidths spanning from 4 to 40 Hz.
    filter_length : Union[str, float, int], default='auto'
        Length of the filter. Can be an integer specifying the number of taps or 'auto' to let
        MNE determine the appropriate length based on other parameters.
    l_trans_bandwidth : Union[str, float, int], default='auto'
        Transition bandwidth for the low cutoff frequency in Hz. Can be specified as a float,
        integer, or 'auto' for automatic selection.
    h_trans_bandwidth : Union[str, float, int], default='auto'
        Transition bandwidth for the high cutoff frequency in Hz. Can be specified as a float,
        integer, or 'auto' for automatic selection.
    method : str, default='fir'
        Filter design method. Supported methods include 'fir' for FIR filters and 'iir' for IIR filters.
    phase : str, default='zero'
        Phase mode for the filter. Options:
            - 'zero': Zero-phase filtering (non-causal).
            - 'minimum': Minimum-phase filtering (causal).
    iir_params : Optional[dict], default=None
        Dictionary of parameters specific to IIR filter design, such as filter order and
        stopband attenuation. Required if `method` is set to 'iir'.
    fir_window : str, default='hamming'
        Window function to use for FIR filter design. Common choices include 'hamming', 'hann',
        'blackman', etc.
    fir_design : str, default='firwin'
        FIR filter design method. Common methods include 'firwin' and 'firwin2'.
    pad : str, default='reflect_limited'
        Padding mode to use when filtering the input signal. Options include 'reflect', 'constant',
        'replicate', etc., as supported by PyTorch's `torch.nn.functional.pad`.
    """

    def __init__(
        self,
        n_chans: int,
        sfreq: int,
        band_filters: Optional[List[Tuple[float, float]]] = None,
        filter_length: str | float | int = "auto",
        l_trans_bandwidth: str | float | int = "auto",
        h_trans_bandwidth: str | float | int = "auto",
        method: str = "fir",
        phase: str = "zero",
        iir_params: Optional[dict] = None,
        fir_window: str = "hamming",
        fir_design: str = "firwin",
    ):
        super(FilterBank, self).__init__()

        if band_filters is None:
            """
            the filter bank is constructed using 9 filters with non-overlapping
            frequency bands, each of 4Hz bandwidth, spanning from 4 to 40 Hz
            (4-8, 8-12, …, 36-40 Hz)

            Based on the reference: FBCNet: A Multi-view Convolutional Neural
            Network for Brain-Computer Interface
            """
            band_filters = [(low, low + 4) for low in range(4, 36 + 1, 4)]

        self.band_filters = band_filters
        self.n_bands = len(band_filters)
        self.phase = phase
        self.method = method
        self.n_chans = n_chans

        method_iir = True if self.method == "iir" else False

        if method_iir:
            raise ValueError("Not implemented yet")

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
                iir_params=iir_params,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
                verbose=True,
            )
            # Shape: (filter_length,)

            filt_tensor = torch.from_numpy(filt).float()
            self.register_buffer(f"filter_{idx}_b", filt_tensor)

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
        n_bands = self.n_bands
        output = []

        for band_idx in range(n_bands):
            # Shape: (nchans, filter_length)
            filt_b = getattr(self, f"filter_{band_idx}_b")

            # Expand to (nchans, filter_length)
            # Shape: (1, nchans, filter_length)
            filt_expanded = filt_b.unsqueeze(0).repeat(self.n_chans, 1).unsqueeze(0)

            # I think it will only work with FIR, check with MNE experts.
            filtered = fftconvolve(
                x, filt_expanded, mode="same"
            )  # Shape: (batch_size, nchans, time_points)

            # Add band dimension
            # Shape: (batch_size, 1, nchans, time_points)
            filtered = filtered.unsqueeze(1)
            output.append(filtered)

        # Shape: (batch_size, n_bands, nchans, time_points)
        output = torch.cat(output, dim=1)
        return output

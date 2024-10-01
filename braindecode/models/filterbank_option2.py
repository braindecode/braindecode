import numpy as np
import torch
import torch.nn as nn
import torchaudio
from typing import List, Optional, Tuple
from mne.filter import create_filter


class FilterBank(nn.Module):
    """
    Filter bank layer using MNE to create multiple band-specific filters
    and applies them using FFT-based convolution with padding to handle
    varying filter lengths.
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
        pad: str = "reflect_limited",
    ):
        super(FilterBank, self).__init__()

        if band_filters is None:
            band_filters = [(4, 8), (8, 12), (12, 30)]  # Default bands

        self.band_filters = band_filters
        self.n_bands = len(band_filters)
        self.pad = pad
        self.phase = phase
        self.method = method

        # Create filters and pad them to the same length
        self.register_buffer(
            "filters",
            self._create_padded_filters(
                nchans=nchans,
                sfreq=sfreq,
                filter_length=filter_length,
                l_trans_bandwidth=l_trans_bandwidth,
                h_trans_bandwidth=h_trans_bandwidth,
                method=method,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
            ),
        )

    def _create_padded_filters(
        self,
        nchans: int,
        sfreq: int,
        filter_length: str | float | int,
        l_trans_bandwidth: str | float | int,
        h_trans_bandwidth: str | float | int,
        method: str,
        phase: str,
        fir_window: str,
        fir_design: str,
    ) -> torch.Tensor:
        """
        Create and pad filters for each band to have the same length.
        Returns a tensor of shape (n_bands, nchans, max_filter_length)
        """
        filters = []
        max_filter_length = 0
        # First pass to find the maximum filter length
        for l_freq, h_freq in self.band_filters:
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
            filt_length = len(filt)
            if filt_length > max_filter_length:
                max_filter_length = filt_length
            filters.append((filt, filt_length))

        # Pad filters to have the same length
        padded_filters = []
        for filt, filt_length in filters:
            if filt_length < max_filter_length:
                # Pad with zeros at the end
                pad_width = max_filter_length - filt_length
                filt = np.pad(filt, (0, pad_width), mode="constant")
            filt_tensor = torch.from_numpy(filt).float()  # Shape: (filter_length,)
            # Expand to (nchans, filter_length)
            filt_tensor = filt_tensor.unsqueeze(0).repeat(nchans, 1)
            padded_filters.append(filt_tensor)

        # Stack filters to shape (n_bands, nchans, max_filter_length)
        filters_tensor = torch.stack(padded_filters, dim=0)
        return filters_tensor  # Shape: (n_bands, nchans, max_filter_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter bank to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nchans, time_points).

        Returns
        -------
        torch.Tensor
            Filtered output tensor of shape (batch_size, n_bands, nchans, time_points).
        """
        batch_size, nchans, time_points = x.shape
        n_bands = self.n_bands
        # max_filter_length = self.filters.shape[2]

        # Prepare the output tensor
        output = []

        for band_idx in range(n_bands):
            # Get the filter for the current band
            filt = self.filters[band_idx]  # Shape: (nchans, max_filter_length)

            # Apply FFT-based convolution
            filt_expanded = filt.unsqueeze(0)  # Shape: (1, nchans, max_filter_length)

            # Perform convolution
            filtered = torchaudio.functional.fftconvolve(
                x, filt_expanded, mode="same"
            )  # Shape: (batch_size, nchans, time_points)

            # Add band dimension
            filtered = filtered.unsqueeze(
                1
            )  # Shape: (batch_size, 1, nchans, time_points)
            output.append(filtered)

        # Concatenate along the band dimension
        output = torch.cat(
            output, dim=1
        )  # Shape: (batch_size, n_bands, nchans, time_points)

        return output


if __name__ == "__main__":
    from torch import randn

    x = randn(16, 10, 1000)
    layer = FilterBank(sfreq=256, nchans=10)
    with torch.no_grad():
        out = layer(x)

    print(out.shape)

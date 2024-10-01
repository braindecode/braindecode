from __future__ import annotations

from typing import List, Optional, Tuple
import torch
from torch import nn

from mne.filter import create_filter

from braindecode.models.functions import fftconvolve


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
        iir_params: Optional[dict] = None,
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        pad="reflect_limited",
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
        self.pad = pad
        self.phase = phase
        self.method = method

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
            # Expand to (nchans, filter_length)
            filt_tensor = filt_tensor.unsqueeze(0).repeat(nchans, 1)

            self.register_buffer(f"filter_{idx}", filt_tensor)

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
            filt = getattr(self, f"filter_{band_idx}")
            # Shape: (1, nchans, filter_length)
            filt_expanded = filt.unsqueeze(0)

            # I think it will only work with IRR, check with MNE experts.
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


if __name__ == "__main__":
    from torch import randn

    x = randn(16, 10, 1000)
    layer = FilterBank(sfreq=256, nchans=10)
    with torch.no_grad():
        out = layer(x)

    print(out.shape)

from __future__ import annotations

from typing import Optional

import torch
from mne.filter import _check_coefficients, create_filter
from mne.utils import warn
from torch import Tensor, from_numpy, nn
from torch.fft import fftfreq
from torchaudio.functional import fftconvolve, filtfilt


class FilterBankLayer(nn.Module):
    """Apply multiple band-pass filters to generate multiview signal representation.

    This layer constructs a bank of signals filtered in specific bands for each channel.
    It uses MNE's `create_filter` function to create the band-specific filters and
    applies them to multi-channel time-series data. Each filter in the bank corresponds to a
    specific frequency band and is applied to all channels of the input data. The filtering is
    performed using FFT-based convolution via the `fftconvolve` function from
    :func:`torchaudio.functional if the method is FIR, and `filtfilt` function from
    :func:`torchaudio.functional if the method is IIR.

    The default configuration creates 9 non-overlapping frequency bands with a 4 Hz bandwidth,
    spanning from 4 Hz to 40 Hz (i.e., 4-8 Hz, 8-12 Hz, ..., 36-40 Hz). This setup is based on the
    reference: *FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface*.

    Parameters
    ----------
    n_chans : int
        Number of channels in the input signal.
    sfreq : int
        Sampling frequency of the input signal in Hz.
    band_filters : Optional[list[tuple[float, float]]] or int, default=None
        List of frequency bands as (low_freq, high_freq) tuples. Each tuple defines
        the frequency range for one filter in the bank. If not provided, defaults
        to 9 non-overlapping bands with 4 Hz bandwidths spanning from 4 to 40 Hz.
    method : str, default='fir'
        ``'fir'`` will use FIR filtering, ``'iir'`` will use IIR
        forward-backward filtering (via :func:`~scipy.signal.filtfilt`).
        For more details, please check the `MNE Preprocessing Tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html>`_.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

        * **'auto' (default)**: The filter length is chosen based
          on the size of the transition regions (6.6 times the reciprocal
          of the shortest transition band for fir_window='hamming'
          and fir_design="firwin2", and half that for "firwin").
        * **str**: A human-readable time in
          units of "s" or "ms" (e.g., "10s" or "5500ms") will be
          converted to that number of samples if ``phase="zero"``, or
          the shortest power-of-two length at least that duration for
          ``phase="zero-double"``.
        * **int**: Specified length in samples. For fir_design="firwin",
          this should not be used.
    l_trans_bandwidth : Union[str, float, int], default='auto'
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Can be "auto"
        (default) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : Union[str, float, int], default='auto'
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    phase : str, default='zero'
        Phase of the filter.
        When ``method='fir'``, symmetric linear-phase FIR filters are constructed
        with the following behaviors when ``method="fir"``:

        ``"zero"`` (default)
            The delay of this filter is compensated for, making it non-causal.
        ``"minimum"``
            A minimum-phase filter will be constructed by decomposing the zero-phase filter
            into a minimum-phase and all-pass systems, and then retaining only the
            minimum-phase system (of the same length as the original zero-phase filter)
            via :func:`scipy.signal.minimum_phase`.
        ``"zero-double"``
            *This is a legacy option for compatibility with MNE <= 0.13.*
            The filter is applied twice, once forward, and once backward
            (also making it non-causal).
        ``"minimum-half"``
            *This is a legacy option for compatibility with MNE <= 1.6.*
            A minimum-phase filter will be reconstructed from the zero-phase filter with
            half the length of the original filter.

        When ``method='iir'``, ``phase='zero'`` (default) or equivalently ``'zero-double'``
        constructs and applies IIR filter twice, once forward, and once backward (making it
        non-causal) using :func:`~scipy.signal.filtfilt`; ``phase='forward'`` will apply
        the filter once in the forward (causal) direction using
        :func:`~scipy.signal.lfilter`.

           The behavior for ``phase="minimum"`` was fixed to use a filter of the requested
           length and improved suppression.
    iir_params : Optional[dict], default=None
        Dictionary of parameters to use for IIR filtering.
        If ``iir_params=None`` and ``method="iir"``, 4th order Butterworth will be used.
        For more information, see :func:`mne.filter.construct_iir_filter`.
    fir_window : str, default='hamming'
        The window to use in FIR design, can be "hamming" (default),
        "hann" (default in 0.13), or "blackman".
    fir_design : str, default='firwin'
        Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
        or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
        a time-domain design technique that generally gives improved
        attenuation using fewer samples than "firwin2".
    pad : str, default='reflect_limited'
        The type of padding to use. Supports all func:`numpy.pad()` mode options.
        Can also be "reflect_limited", which pads with a reflected version of
        each vector mirrored on the first and last values of the vector,
        followed by zeros. Only used for ``method='fir'``.
    verbose: bool | str | int | None, default=True
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the func:`mne.verbose` for details.
        Should only be passed as a keyword argument.
    """

    def __init__(
        self,
        n_chans: int,
        sfreq: float,
        band_filters: Optional[list[tuple[float, float]] | int] = None,
        method: str = "fir",
        filter_length: str | float | int = "auto",
        l_trans_bandwidth: str | float | int = "auto",
        h_trans_bandwidth: str | float | int = "auto",
        phase: str = "zero",
        iir_params: Optional[dict] = None,
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        verbose: bool = True,
    ):
        super(FilterBankLayer, self).__init__()

        # The first step here is to check the band_filters
        # We accept as None values.
        if band_filters is None:
            """
            the filter bank is constructed using 9 filters with non-overlapping
            frequency bands, each of 4Hz bandwidth, spanning from 4 to 40 Hz
            (4-8, 8-12, …, 36-40 Hz)

            Based on the reference: FBCNet: A Multi-view Convolutional Neural
            Network for Brain-Computer Interface
            """
            band_filters = [(low, low + 4) for low in range(4, 36 + 1, 4)]
        # We accept as int.
        if isinstance(band_filters, int):
            warn(
                "Creating the filter banks equally divided in the "
                "interval 4Hz to 40Hz with almost equal bandwidths. "
                "If you want a specific interval, "
                "please specify 'band_filters' as a list of tuples.",
                UserWarning,
            )
            start = 4.0
            end = 40.0

            total_band_width = end - start  # 4 Hz to 40 Hz

            band_width_calculated = total_band_width / band_filters
            band_filters = [
                (
                    float(start + i * band_width_calculated),
                    float(start + (i + 1) * band_width_calculated),
                )
                for i in range(band_filters)
            ]

        if not isinstance(band_filters, list):
            raise ValueError(
                "`band_filters` should be a list of tuples if you want to "
                "use them this way."
            )
        else:
            if any(len(bands) != 2 for bands in band_filters):
                raise ValueError(
                    "The band_filters items should be splitable in 2 values."
                )

        # and we accepted as
        self.band_filters = band_filters
        self.n_bands = len(band_filters)
        self.phase = phase
        self.method = method
        self.n_chans = n_chans

        self.method_iir = self.method == "iir"

        # Prepare ParameterLists
        self.fir_list = nn.ParameterList()
        self.b_list = nn.ParameterList()
        self.a_list = nn.ParameterList()

        if self.method_iir:
            if iir_params is None:
                iir_params = dict(output="ba")
            else:
                if "output" in iir_params:
                    if iir_params["output"] == "sos":
                        warn(
                            "It is not possible to use second-order section filtering with Torch. Changing to filter ba",
                            UserWarning,
                        )
                        iir_params["output"] = "ba"

        for l_freq, h_freq in band_filters:
            filt = create_filter(
                data=None,
                sfreq=sfreq,
                l_freq=float(l_freq),
                h_freq=float(h_freq),
                filter_length=filter_length,
                l_trans_bandwidth=l_trans_bandwidth,
                h_trans_bandwidth=h_trans_bandwidth,
                method=self.method,
                iir_params=iir_params,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
                verbose=verbose,
            )
            if not self.method_iir:
                # FIR filter
                filt = from_numpy(filt).float()
                self.fir_list.append(nn.Parameter(filt, requires_grad=False))
            else:
                a_coeffs = filt["a"]
                b_coeffs = filt["b"]

                _check_coefficients((b_coeffs, a_coeffs))

                b = torch.tensor(b_coeffs, dtype=torch.float64)
                a = torch.tensor(a_coeffs, dtype=torch.float64)

                self.b_list.append(nn.Parameter(b, requires_grad=False))
                self.a_list.append(nn.Parameter(a, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter bank to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, time_points).

        Returns
        -------
        torch.Tensor
            Filtered output tensor of shape (batch_size, n_bands, n_chans, filtered_time_points).
        """
        outs = []
        if self.method_iir:
            for b, a in zip(self.b_list, self.a_list):
                # Pass numerator and denominator directly
                outs.append(self._apply_iir(x=x, b_coeffs=b, a_coeffs=a))
        else:
            for fir in self.fir_list:
                # Pass FIR filter directly
                outs.append(self._apply_fir(x=x, filt=fir, n_chans=self.n_chans))

        return torch.cat(outs, dim=1)

    @staticmethod
    def _apply_fir(x, filt: Tensor, n_chans: int) -> Tensor:
        """
        Apply an FIR filter to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, n_chans, n_times).
        filter : dict
            Dictionary containing IIR filter coefficients.
            - "b": Tensor of numerator coefficients.
        n_chans: int
            Number of channels

        Returns
        -------
        Tensor
            Filtered tensor of shape (batch_size, 1, n_chans, n_times).
        """
        # Expand filter coefficients to match the number of channels
        # Original 'b' shape: (filter_length,)
        # After unsqueeze and repeat: (n_chans, filter_length)
        # After final unsqueeze: (1, n_chans, filter_length)
        filt_expanded = filt.to(x.device).unsqueeze(0).repeat(n_chans, 1).unsqueeze(0)

        # Perform FFT-based convolution
        # Input x shape: (batch_size, n_chans, n_times)
        # filt_expanded shape: (1, n_chans, filter_length)
        # After convolution: (batch_size, n_chans, n_times)

        filtered = fftconvolve(
            x, filt_expanded, mode="same"
        )  # Shape: (batch_size, nchans, time_points)

        # Add a new dimension for the band
        # Shape after unsqueeze: (batch_size, 1, n_chans, n_times)
        filtered = filtered.unsqueeze(1)
        # returning the filtered
        return filtered

    @staticmethod
    def _apply_iir(x: Tensor, b_coeffs: Tensor, a_coeffs: Tensor) -> Tensor:
        """
        Apply an IIR filter to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, n_chans, n_times).
        filter : dict
            Dictionary containing IIR filter coefficients

            - "b": Tensor of numerator coefficients.
            - "a": Tensor of denominator coefficients.

        Returns
        -------
        Tensor
            Filtered tensor of shape (batch_size, 1, n_chans, n_times).
        """
        # Apply filtering using torchaudio's filtfilt
        filtered = filtfilt(
            x,
            a_coeffs=a_coeffs.type_as(x).to(x.device),
            b_coeffs=b_coeffs.type_as(x).to(x.device),
            clamp=False,
        )
        # Rearrange dimensions to (batch_size, 1, n_chans, n_times)
        return filtered.unsqueeze(1)


class GeneralizedGaussianFilter(nn.Module):
    """Generalized Gaussian Filter from Ludwig et al (2024) [eegminer]_.

    Implements trainable temporal filters based on generalized Gaussian functions
    in the frequency domain.

    This module creates filters in the frequency domain using the generalized
    Gaussian function, allowing for trainable center frequency (`f_mean`),
    bandwidth (`bandwidth`), and shape (`shape`) parameters.

    The filters are applied to the input signal in the frequency domain, and can
    be optionally transformed back to the time domain using the inverse
    Fourier transform.

    The generalized Gaussian function in the frequency domain is defined as:

    .. math::

        F(x) = \\exp\\left( - \\left( \\frac{abs(x - \\mu)}{\\alpha} \\right)^{\\beta} \\right)

    where:
      - μ (mu) is the center frequency (`f_mean`).

      - α (alpha) is the scale parameter, reparameterized in terms of the full width at half maximum (FWHM) `h` as:

      .. math::

          \\alpha = \\frac{h}{2 \\left( \\ln(2) \\right)^{1/\\beta}}

      - β (beta) is the shape parameter (`shape`), controlling the shape of the filter.

    The filters are constructed in the frequency domain to allow full control
    over the magnitude and phase responses.

    A linear phase response is used, with an optional trainable group delay (`group_delay`).

      - Copyright (C) Cogitat, Ltd.
      - Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
      - Patent GB2609265 - Learnable filters for eeg classification
      - https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. Must be a multiple of `in_channels`.
    sequence_length : int
        Length of the input sequences (time steps).
    sample_rate : float
        Sampling rate of the input signals in Hz.
    inverse_fourier : bool, optional
        If True, applies the inverse Fourier transform to return to the time domain after filtering.
        Default is True.
    affine_group_delay : bool, optional
        If True, makes the group delay parameter trainable. Default is False.
    group_delay : tuple of float, optional
        Initial group delay(s) in milliseconds for the filters. Default is (20.0,).
    f_mean : tuple of float, optional
        Initial center frequency (frequencies) of the filters in Hz. Default is (23.0,).
    bandwidth : tuple of float, optional
        Initial bandwidth(s) (full width at half maximum) of the filters in Hz. Default is (44.0,).
    shape : tuple of float, optional
        Initial shape parameter(s) of the generalized Gaussian filters. Must be >= 2.0. Default is (2.0,).
    clamp_f_mean : tuple of float, optional
        Minimum and maximum allowable values for the center frequency `f_mean` in Hz.
        Specified as (min_f_mean, max_f_mean). Default is (1.0, 45.0).


    Notes
    -----
    The model and the module **have a patent** [eegminercode]_, and the **code is CC BY-NC 4.0**.

    .. versionadded:: 0.9

    References
    ----------
    .. [eegminer] Ludwig, S., Bakas, S., Adamos, D. A., Laskaris, N., Panagakis,
       Y., & Zafeiriou, S. (2024). EEGMiner: discovering interpretable features
       of brain activity with learnable filters. Journal of Neural Engineering,
       21(3), 036010.
    .. [eegminercode] Ludwig, S., Bakas, S., Adamos, D. A., Laskaris, N., Panagakis,
       Y., & Zafeiriou, S. (2024). EEGMiner: discovering interpretable features
       of brain activity with learnable filters.
       https://github.com/SMLudwig/EEGminer/.
       Cogitat, Ltd. "Learnable filters for EEG classification."
       Patent GB2609265.
       https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        sequence_length,
        sample_rate,
        inverse_fourier=True,
        affine_group_delay=False,
        group_delay=(20.0,),
        f_mean=(23.0,),
        bandwidth=(44.0,),
        shape=(2.0,),
        clamp_f_mean=(1.0, 45.0),
    ):
        super(GeneralizedGaussianFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.inverse_fourier = inverse_fourier
        self.affine_group_delay = affine_group_delay
        self.clamp_f_mean = clamp_f_mean
        if out_channels % in_channels != 0:
            raise ValueError("out_channels has to be multiple of in_channels")
        if len(f_mean) * in_channels != out_channels:
            raise ValueError("len(f_mean) * in_channels must equal out_channels")
        if len(bandwidth) * in_channels != out_channels:
            raise ValueError("len(bandwidth) * in_channels must equal out_channels")
        if len(shape) * in_channels != out_channels:
            raise ValueError("len(shape) * in_channels must equal out_channels")

        # Range from 0 to half sample rate, normalized
        self.n_range = nn.Parameter(
            torch.tensor(
                list(
                    fftfreq(n=sequence_length, d=1 / sample_rate)[
                        : sequence_length // 2
                    ]
                )
                + [sample_rate / 2]
            )
            / (sample_rate / 2),
            requires_grad=False,
        )

        # Trainable filter parameters
        self.f_mean = nn.Parameter(
            torch.tensor(f_mean * in_channels) / (sample_rate / 2), requires_grad=True
        )
        self.bandwidth = nn.Parameter(
            torch.tensor(bandwidth * in_channels) / (sample_rate / 2),
            requires_grad=True,
        )  # full width half maximum
        self.shape = nn.Parameter(torch.tensor(shape * in_channels), requires_grad=True)

        # Normalize group delay so that group_delay=1 corresponds to 1000ms
        self.group_delay = nn.Parameter(
            torch.tensor(group_delay * in_channels) / 1000,
            requires_grad=affine_group_delay,
        )

        # Construct filters from parameters
        self.filters = self.construct_filters()

    @staticmethod
    def exponential_power(x, mean, fwhm, shape):
        """
        Computes the generalized Gaussian function:

        .. math::

             F(x) = \\exp\\left( - \\left( \\frac{|x - \\mu|}{\\alpha} \\right)^{\\beta} \\right)

        where:

          - :math:`\\mu` is the mean (`mean`).

          - :math:`\\alpha` is the scale parameter, reparameterized using the FWHM :math:`h` as:

            .. math::

                \\alpha = \\frac{h}{2 \\left( \\ln(2) \\right)^{1/\\beta}}

          - :math:`\\beta` is the shape parameter (`shape`).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor representing frequencies, normalized between 0 and 1.
        mean : torch.Tensor
            The center frequency (`f_mean`), normalized between 0 and 1.
        fwhm : torch.Tensor
            The full width at half maximum (`bandwidth`), normalized between 0 and 1.
        shape : torch.Tensor
            The shape parameter (`shape`) of the generalized Gaussian.

        Returns
        -------
        torch.Tensor
            The computed generalized Gaussian function values at frequencies `x`.

        """
        mean = mean.unsqueeze(1)
        fwhm = fwhm.unsqueeze(1)
        shape = shape.unsqueeze(1)
        log2 = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        scale = fwhm / (2 * log2 ** (1 / shape))
        # Add small constant to difference between x and mean since grad of 0 ** shape is nan
        return torch.exp(-((((x - mean).abs() + 1e-8) / scale) ** shape))

    def construct_filters(self):
        """
        Constructs the filters in the frequency domain based on current parameters.

        Returns
        -------
        torch.Tensor
            The constructed filters with shape `(out_channels, freq_bins, 2)`.

        """
        # Clamp parameters
        self.f_mean.data = torch.clamp(
            self.f_mean.data,
            min=self.clamp_f_mean[0] / (self.sample_rate / 2),
            max=self.clamp_f_mean[1] / (self.sample_rate / 2),
        )
        self.bandwidth.data = torch.clamp(
            self.bandwidth.data, min=1.0 / (self.sample_rate / 2), max=1.0
        )
        self.shape.data = torch.clamp(self.shape.data, min=2.0, max=3.0)

        # Create magnitude response with gain=1 -> (channels, freqs)
        mag_response = self.exponential_power(
            self.n_range, self.f_mean, self.bandwidth, self.shape * 8 - 14
        )
        mag_response = mag_response / mag_response.max(dim=-1, keepdim=True)[0]

        # Create phase response, scaled so that normalized group_delay=1
        # corresponds to group delay of 1000ms.
        phase = torch.linspace(
            0,
            self.sample_rate,
            self.sequence_length // 2 + 1,
            device=mag_response.device,
            dtype=mag_response.dtype,
        )
        phase = phase.expand(mag_response.shape[0], -1)  # repeat for filter channels
        pha_response = -self.group_delay.unsqueeze(-1) * phase * torch.pi

        # Create real and imaginary parts of the filters
        real = mag_response * torch.cos(pha_response)
        imag = mag_response * torch.sin(pha_response)

        # Stack real and imaginary parts to create filters
        # -> (channels, freqs, 2)
        filters = torch.stack((real, imag), dim=-1)

        return filters

    def forward(self, x):
        """
        Applies the generalized Gaussian filters to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., in_channels, sequence_length)`.

        Returns
        -------
        torch.Tensor
            The filtered signal. If `inverse_fourier` is True, returns the signal in the time domain
            with shape `(..., out_channels, sequence_length)`. Otherwise, returns the signal in the
            frequency domain with shape `(..., out_channels, freq_bins, 2)`.

        """
        # Construct filters from parameters
        self.filters = self.construct_filters()
        # Preserving the original dtype.
        dtype = x.dtype
        # Apply FFT -> (..., channels, freqs, 2)
        x = torch.fft.rfft(x, dim=-1)
        x = torch.view_as_real(x)  # separate real and imag

        # Repeat channels in case of multiple filters per channel
        x = torch.repeat_interleave(x, self.out_channels // self.in_channels, dim=-3)

        # Apply filters in the frequency domain
        x = x * self.filters

        # Apply inverse FFT if requested
        if self.inverse_fourier:
            x = torch.view_as_complex(x)
            x = torch.fft.irfft(x, n=self.sequence_length, dim=-1)

        x = x.to(dtype)

        return x

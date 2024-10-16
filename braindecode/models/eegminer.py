"""
* Copyright (C) Cogitat, Ltd.
* Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
* Patent GB2609265 - Learnable filters for eeg classification
* https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0
"""

from functools import partial

import torch

import braindecode.models.functions as F

from torch.fft import fftfreq
from einops.layers.torch import Rearrange

from torch import nn
from braindecode.models.base import EEGModuleMixin

_eeg_miner_methods = ["mag", "corr", "plv"]


class EEGMiner(EEGModuleMixin, nn.Module):
    """EEGMiner from Ludwig et al (2024) [eegminer]_.

    .. figure:: https://content.cld.iop.org/journals/1741-2552/21/3/036010/revision2/jnead44d7f1_hr.jpg
       :align: center
       :alt: EEGMiner Architecture

    EEGMiner is a neural network model for EEG signal classification using
    learnable generalized Gaussian filters. The model leverages frequency domain
    filtering and connectivity metrics such as Phase Locking Value (PLV) to extract
    meaningful features from EEG data, enabling effective classification tasks.

    The model has the following steps:

    - **Generalized Gaussian** filters in the frequency domain to the input EEG signals.

    - **Connectivity estimators**, by default, we use phase locking value.
        - `'mag'`: Computes the magnitude of the filtered signals.
        - `'corr'`: Computes the correlation of the filtered signals.
        - `'plv'`: Computes the phase locking value of the filtered signals.

    - **Feature Normalization**
        - Apply batch normalization.

    - **Final Layer**
        - Feeds the batch-normalized features into a final linear layer for classification.

    Depending on the selected method (`mag`, `corr`, or `plv`),
    it computes the filtered signals' magnitude, correlation, or phase locking value.
    These features are then normalized and passed through a batch normalization layer
    before being fed into a final linear layer for classification.

    The input to EEGMiner should be a three-dimensional tensor representing EEG signals:

    ``(batch_size, n_channels, n_timesteps)``.

    Notes
    -----
    EEGMiner incorporates learnable parameters for filter characteristics, allowing the
    model to adaptively learn optimal frequency bands and phase delays for the classification task.
    The use of PLV as a connectivity metric makes EEGMiner suitable for tasks requiring
    the analysis of phase relationships between different EEG channels.

    The model has a patent [eegminercode]_, and the code is CC BY-NC 4.0.

    .. versionadded:: 0.9

    Parameters
    ----------
    method : str, default="plv"
        The method used for feature extraction. Options are:
        - "mag": Magnitude of the filtered signals.
        - "corr": Correlation between filtered channels.
        - "plv": Phase Locking Value connectivity metric.
    filter_f_mean : list of float, default=[23.0, 23.0]
        Mean frequencies for the generalized Gaussian filters.
    filter_bandwidth : list of float, default=[44.0, 44.0]
        Bandwidths for the generalized Gaussian filters.
    filter_shape : list of float, default=[2.0, 2.0]
        Shape parameters for the generalized Gaussian filters.
    group_delay : tuple of float, default=(20.0, 20.0)
        Group delay values for the filters in milliseconds.
    clamp_f_mean : tuple of float, default=(1.0, 45.0)
        Clamping range for the mean frequency parameters.

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
        self,  # Signal related parameters
        method: str = "plv",
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # model related
        filter_f_mean=(23.0, 23.0),
        filter_bandwidth=(44.0, 44.0),
        filter_shape=(2.0, 2.0),
        group_delay=(20.0, 20.0),
        clamp_f_mean=(1.0, 45.0),
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Initialize filter parameters
        self.filter_f_mean = filter_f_mean
        self.filter_bandwidth = filter_bandwidth
        self.filter_shape = filter_shape
        self.n_filters = len(self.filter_f_mean)
        self.group_delay = group_delay
        self.clamp_f_mean = clamp_f_mean
        self.method = method.lower()

        if self.method not in _eeg_miner_methods:
            raise ValueError(
                f"The method {self.method} is not one of the valid options"
                f" {_eeg_miner_methods}"
            )

        if self.method == "mag" or self.method == "corr":
            inverse_fourier = True
            in_channels = self.n_chans
            out_channels = self.n_chans * self.n_filters
        else:
            inverse_fourier = False
            in_channels = 1
            out_channels = 1 * self.n_filters

        # Generalized Gaussian Filter
        self.filter = GeneralizedGaussianFilter(
            in_channels=in_channels,
            out_channels=out_channels,
            sequence_length=self.n_times,
            sample_rate=self.sfreq,
            f_mean=self.filter_f_mean,
            bandwidth=self.filter_bandwidth,
            shape=self.filter_shape,
            affine_group_delay=False,
            inverse_fourier=inverse_fourier,
            group_delay=self.group_delay,
            clamp_f_mean=self.clamp_f_mean,
        )

        # Forward method
        if self.method == "mag":
            self.method_forward = self._apply_mag_forward
            self.n_features = self.n_chans * self.n_filters
            self.ensure_dim = nn.Identity()
        elif self.method == "corr":
            self.method_forward = partial(
                self._apply_corr_forward,
                n_chans=self.n_chans,
                n_filters=self.n_filters,
                n_times=self.n_times,
            )
            self.n_features = self.n_filters * self.n_chans * (self.n_chans - 1) // 2
            self.ensure_dim = nn.Identity()
        elif self.method == "plv":
            self.method_forward = partial(self._apply_plv, n_chans=self.n_chans)
            self.ensure_dim = Rearrange("... d -> ... 1 d")
            self.n_features = (self.n_filters * self.n_chans * (self.n_chans - 1)) // 2

        self.flatten_layer = nn.Flatten()
        # Classifier
        self.batch_layer = nn.BatchNorm1d(self.n_features, affine=False)
        self.final_layer = nn.Linear(self.n_features, self.n_outputs)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        batch = x.shape[0]
        x = self.ensure_dim(x)
        # Apply Gaussian filters in frequency domain
        # x -> (batch, electrodes * filters, time)
        x = self.filter(x)

        x = self.method_forward(x=x, batch=batch)
        # Classifier
        # Note that the order of dimensions before flattening the feature vector is important
        # for attributing feature weights during interpretation.
        x = x.reshape(batch, self.n_features)
        x = self.batch_layer(x)
        x = self.final_layer(x)

        return x

    @staticmethod
    def _apply_mag_forward(x, batch=None):
        # Signal magnitude
        x = x * x
        x = x.mean(dim=-1)
        x = torch.sqrt(x)
        return x

    @staticmethod
    def _apply_corr_forward(
        x, batch, n_chans, n_filters, n_times, epilson: float = 1e-6
    ):
        x = x.reshape(batch, n_chans, n_filters, n_times).transpose(-3, -2)
        x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            x.var(dim=-1, keepdim=True) + epilson
        )
        x = torch.matmul(x, x.transpose(-2, -1)) / x.shape[-1]
        # Original tensor shape: [batch, n_filters, chans, chans]
        x = x.permute(0, 2, 3, 1)
        # New tensor shape: [batch, chans, chans, n_filters]
        # move filter channels to the end
        x = x.abs()

        # Get upper triu of symmetric connectivity matrix
        triu = torch.triu_indices(n_chans, n_chans, 1)
        x = x[:, triu[0], triu[1], :]

        return x

    @staticmethod
    def _apply_plv(x, n_chans, batch=None):
        # Compute PLV connectivity
        # x -> (batch, electrodes, electrodes, filters)
        x = x.transpose(-4, -3)  # swap electrodes and filters
        # adjusting to compute the plv
        x = F.plv_time(x, forward_fourier=False)
        # batch, number of filters, connectivity matrix
        # [batch, n_filters, chans, chans]
        x = x.permute(0, 2, 3, 1)
        # [batch, chans, chans, n_filters]

        # Get upper triu of symmetric connectivity matrix
        triu = torch.triu_indices(n_chans, n_chans, 1)
        x = x[:, triu[0], triu[1], :]
        return x


class GeneralizedGaussianFilter(nn.Module):
    """Generalized Gaussian Filter from [eegminer]_.

    Implements trainable temporal filters based on generalized Gaussian functions
    in the frequency domain.

    This module creates filters in the frequency domain using the generalized
    Gaussian function, allowing for trainable center frequency (`f_mean`),
    bandwidth (`bandwidth`), and shape (`shape`) parameters.

    The filters are applied to the input signal in the frequency domain, and can
    be optionally transformed back to the time domain using the inverse
    Fourier transform.

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

    Methods
    -------
    forward(x)
        Applies the filters to the input tensor `x` and returns the filtered signal.

    Notes
    -----
    The generalized Gaussian function in the frequency domain is defined as:

        F(x) = exp(- (|x - μ| / α) ** β)

    where:
    - μ (mu) is the center frequency (`f_mean`).
    - α (alpha) is the scale parameter, reparameterized in terms of the full
    width at half maximum (FWHM) `h` as:

        α = h / (2 * (ln(2)) ** (1/β))

    - β (beta) is the shape parameter (`shape`), controlling the shape of the filter.

    The filters are constructed in the frequency domain to allow full control
    over the magnitude and phase responses.

    A linear phase response is used, with an optional trainable group delay (`group_delay`).

    References
    ----------
    .. [eegminer] Ludwig, S., Bakas, S., Adamos, D. A., Laskaris, N., Panagakis,
       Y., & Zafeiriou, S. (2024). EEGMiner: discovering interpretable features
       of brain activity with learnable filters. Journal of Neural Engineering,
       21(3), 036010.

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
        assert (
            out_channels % in_channels == 0
        ), "out_channels has to be multiple of in_channels"
        assert len(f_mean) * in_channels == out_channels
        assert len(bandwidth) * in_channels == out_channels
        assert len(shape) * in_channels == out_channels

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

            F(x) = \exp\left( - \left( \frac{|x - \mu|}{\alpha} \right)^{\beta} \right)

        where:

        - :math:`\mu` is the mean (`mean`).
        - :math:`\alpha` is the scale parameter, reparameterized using the FWHM :math:`h` as:

          .. math::

              \alpha = \frac{h}{2 \left( \ln(2) \right)^{1/\beta}}

        - :math:`\beta` is the shape parameter (`shape`).

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

"""
* Copyright (C) Cogitat, Ltd.
* Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
* Patent GB2609265 - Learnable filters for eeg classification
* https://www.ipo.gov.uk/p-ipsum/Case/ApplicationNumber/GB2113420.0
"""

from functools import partial

import torch
from einops.layers.torch import Rearrange
from torch import nn

import braindecode.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import GeneralizedGaussianFilter

_eeg_miner_methods = ["mag", "corr", "plv"]


class EEGMiner(EEGModuleMixin, nn.Module):
    """EEGMiner from Ludwig et al (2024) [eegminer]_.

    .. figure:: https://content.cld.iop.org/journals/1741-2552/21/3/036010/revision2/jnead44d7f1_hr.jpg
       :align: center
       :alt: EEGMiner Architecture

    EEGMiner is a neural network model for EEG signal classification using
    learnable generalized Gaussian filters. The model leverages frequency domain
    filtering and connectivity metrics or feature extraction, such as Phase Locking
    Value (PLV) to extract meaningful features from EEG data, enabling effective
    classification tasks.

    The model has the following steps:

    - **Generalized Gaussian** filters in the frequency domain to the input EEG signals.

    - **Connectivity estimators** (corr, plv) or **Electrode-Wise Band Power** (mag), by default (plv).
        - `'corr'`: Computes the correlation of the filtered signals.
        - `'plv'`: Computes the phase locking value of the filtered signals.
        - `'mag'`: Computes the magnitude of the filtered signals.

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
    By default, using the PLV as a connectivity metric makes EEGMiner suitable for tasks requiring
    the analysis of phase relationships between different EEG channels.

    The model and the module have patent [eegminercode]_, and the code is CC BY-NC 4.0.

    .. versionadded:: 0.9

    Parameters
    ----------
    method : str, default="plv"
        The method used for feature extraction. Options are:
        - "mag": Electrode-Wise band power of the filtered signals.
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

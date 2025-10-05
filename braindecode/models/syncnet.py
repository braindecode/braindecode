import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from numpy import arange, ceil

from braindecode.models.base import EEGModuleMixin


class SyncNet(EEGModuleMixin, nn.Module):
    """Synchronization Network (SyncNet) from Li, Y et al (2017) [Li2017]_.

    .. figure:: https://braindecode.org/dev/_static/model/SyncNet.png
        :align: center
        :alt: SyncNet Architecture

    SyncNet uses parameterized 1-dimensional convolutional filters inspired by
    the Morlet wavelet to extract features from EEG signals. The filters are
    dynamically generated based on learnable parameters that control the
    oscillation and decay characteristics.

    The filter for channel ``c`` and filter ``k`` is defined as:

    .. math::

        f_c^{(k)}(\\tau) = amplitude_c^{(k)} \\cos(\\omega^{(k)} \\tau + \\phi_c^{(k)}) \\exp(-\\beta^{(k)} \\tau^2)

    where:
    - :math:`amplitude_c^{(k)}` is the amplitude parameter (channel-specific).
    - :math:`\\omega^{(k)}` is the frequency parameter (shared across channels).
    - :math:`\\phi_c^{(k)}` is the phase shift (channel-specific).
    - :math:`\\beta^{(k)}` is the decay parameter (shared across channels).
    - :math:`\\tau` is the time index.

    Parameters
    ----------
    num_filters : int, optional
        Number of filters in the convolutional layer. Default is 1.
    filter_width : int, optional
        Width of the convolutional filters. Default is 40.
    pool_size : int, optional
        Size of the pooling window. Default is 40.
    activation : nn.Module, optional
        Activation function to apply after pooling. Default is ``nn.ReLU``.
    ampli_init_values : tuple of float, optional
        The initialization range for amplitude parameter using uniform
        distribution. Default is (-0.05, 0.05).
    omega_init_values : tuple of float, optional
        The initialization range for omega parameters using uniform
        distribution. Default is (0, 1).
    beta_init_values : tuple of float, optional
        The initialization range for beta parameters using uniform
        distribution. Default is (0, 1). Default is (0, 0.05).
    phase_init_values : tuple of float, optional
        The initialization range for phase parameters using `normal`
        distribution. Default is (0, 1). Default is (0, 0.05).


    Notes
    -----
    This implementation is not guaranteed to be correct! it has not been checked
    by original authors. The modifications are based on derivated code from
    [CodeICASSP2025]_.


    References
    ----------
    .. [Li2017] Li, Y., Dzirasa, K., Carin, L., & Carlson, D. E. (2017).
       Targeting EEG/LFP synchrony with neural nets. Advances in neural
       information processing systems, 30.
    .. [CodeICASSP2025] Code from Baselines for EEG-Music Emotion Recognition
       Grand Challenge at ICASSP 2025.
       https://github.com/SalvoCalcagno/eeg-music-challenge-icassp-2025-baselines

    """

    def __init__(
        self,
        # braindecode convention
        n_chans=None,
        n_times=None,
        n_outputs=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # model parameters
        num_filters=1,
        filter_width=40,
        pool_size=40,
        activation: nn.Module = nn.ReLU,
        ampli_init_values: tuple[float, float] = (-0.05, 0.05),
        omega_init_values: tuple[float, float] = (0.0, 1.0),
        beta_init_values: tuple[float, float] = (0.0, 0.05),
        phase_init_values: tuple[float, float] = (0.0, 0.05),
    ):
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.num_filters = num_filters
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.activation = activation()
        self.ampli_init_values = ampli_init_values
        self.omega_init_values = omega_init_values
        self.beta_init_values = beta_init_values
        self.phase_init_values = phase_init_values

        # Initialize parameters
        self.amplitude = nn.Parameter(
            torch.FloatTensor(1, 1, self.n_chans, self.num_filters).uniform_(
                self.ampli_init_values[0], self.ampli_init_values[1]
            )
        )
        self.omega = nn.Parameter(
            torch.FloatTensor(1, 1, 1, self.num_filters).uniform_(
                self.omega_init_values[0], self.omega_init_values[1]
            )
        )

        self.bias = nn.Parameter(torch.zeros(self.num_filters))

        # Calculate the output size after pooling
        self.classifier_input_size = int(
            ceil(float(self.n_times) / float(self.pool_size)) * self.num_filters
        )

        # Create time vector t
        if self.filter_width % 2 == 0:
            t_range = arange(-int(self.filter_width / 2), int(self.filter_width / 2))
        else:
            t_range = arange(
                -int((self.filter_width - 1) / 2), int((self.filter_width - 1) / 2) + 1
            )

        t_np = t_range.reshape(1, self.filter_width, 1, 1)
        self.t = nn.Parameter(torch.FloatTensor(t_np))
        # Phase Shift
        self.phi_ini = nn.Parameter(
            torch.FloatTensor(1, 1, self.n_chans, self.num_filters).normal_(
                self.beta_init_values[0], self.beta_init_values[1]
            )
        )
        self.beta = nn.Parameter(
            torch.FloatTensor(1, 1, 1, self.num_filters).uniform_(
                self.phase_init_values[0], self.phase_init_values[1]
            )
        )

        self.padding = self._compute_padding(filter_width=self.filter_width)
        self.pad_input = nn.ConstantPad1d(self.padding, 0.0)
        self.pad_res = nn.ConstantPad1d(self.padding, 0.0)

        # Define pooling and classifier layers
        self.pool = nn.MaxPool2d((1, self.pool_size), stride=(1, self.pool_size))

        self.ensuredim = Rearrange("batch ch time -> batch ch 1 time")

        self.final_layer = nn.Linear(self.classifier_input_size, self.n_outputs)

    def forward(self, x):
        """Forward pass of the SyncNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times)

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, n_outputs).

        """
        # Ensure input tensor has shape (batch_size, n_chans, 1, n_times)
        x = self.ensuredim(x)
        # Output: (batch_size, n_chans, 1, n_times)

        # Compute the oscillatory component
        W_osc = self.amplitude * torch.cos(self.t * self.omega + self.phi_ini)
        # W_osc is (1, filter_width, n_chans, 1)

        # Compute the decay component
        t_squared = torch.pow(self.t, 2)  # Shape: (filter_width,)
        t_squared_beta = t_squared * self.beta  # Shape: (filter_width, num_filters)
        W_decay = torch.exp(-t_squared_beta)
        # W_osc is (1, filter_width, 1, 1)

        # Combine oscillatory and decay components
        # W shape: (1, n_chans, num_filters, filter_width)
        W = W_osc * W_decay
        # W shape will be: (1, filter_width, n_chans, 1)

        W = W.view(self.num_filters, self.n_chans, 1, self.filter_width)

        # Apply convolution
        x_padded = self.pad_input(x.float())

        res = F.conv2d(x_padded, W.float(), bias=self.bias, stride=1)

        # Apply padding to the convolution result
        res_padded = self.pad_res(res)
        res_pooled = self.pool(res_padded)

        # Flatten the result
        res_flat = res_pooled.view(-1, self.classifier_input_size)

        # Ensure beta remains non-negative
        self.beta.data.clamp_(min=0)

        # Apply activation
        out = self.activation(res_flat)
        # Apply classifier
        out = self.final_layer(out)

        return out

    @staticmethod
    def _compute_padding(filter_width):
        # Compute padding
        P = filter_width - 2
        if P % 2 == 0:
            padding = (P // 2, P // 2 + 1)
        else:
            padding = (P // 2, P // 2)
        return padding

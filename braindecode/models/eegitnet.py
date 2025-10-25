# Authors: Ghaith Bouallegue <ghaithbouallegue@gmail.com>
#
# License: BSD-3
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import DepthwiseConv2d, Ensure4d, InceptionBlock


class EEGITNet(EEGModuleMixin, nn.Sequential):
    """EEG-ITNet from Salami, et al (2022) [Salami2022]_

    .. figure:: https://braindecode.org/dev/_static/model/eegitnet.jpg
        :align: center
        :alt: EEG-ITNet Architecture

    EEG-ITNet: An Explainable Inception Temporal
    Convolutional Network for motor imagery classification from
    Salami et al. 2022.

    See [Salami2022]_ for details.

    Code adapted from https://github.com/abbassalami/eeg-itnet

    Parameters
    ----------
    drop_prob: float
        Dropout probability.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    kernel_length : int, optional
        Kernel length for inception branches. Determines the temporal receptive field.
        Default is 16.
    pool_kernel : int, optional
        Pooling kernel size for the average pooling layer. Default is 4.
    tcn_in_channel : int, optional
        Number of input channels for Temporal Convolutional (TC) blocks. Default is 14.
    tcn_kernel_size : int, optional
        Kernel size for the TC blocks. Determines the temporal receptive field.
        Default is 4.
    tcn_padding : int, optional
        Padding size for the TC blocks to maintain the input dimensions. Default is 3.
    drop_prob : float, optional
        Dropout probability applied after certain layers to prevent overfitting.
        Default is 0.4.
    tcn_dilatation : int, optional
        Dilation rate for the first TC block. Subsequent blocks will have
        dilation rates multiplied by powers of 2. Default is 1.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper based on author implementation.


    References
    ----------
    .. [Salami2022] A. Salami, J. Andreu-Perez and H. Gillmeister, "EEG-ITNet:
        An Explainable Inception Temporal Convolutional Network for motor
        imagery classification," in IEEE Access,
        doi: 10.1109/ACCESS.2022.3161489.
    """

    def __init__(
        self,
        # Braindecode parameters
        n_outputs=None,
        n_chans=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        n_filters_time: int = 2,
        kernel_length: int = 16,
        pool_kernel: int = 4,
        tcn_in_channel: int = 14,
        tcn_kernel_size: int = 4,
        tcn_padding: int = 3,
        drop_prob: float = 0.4,
        tcn_dilatation: int = 1,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.mapping = {
            "classification.1.weight": "final_layer.clf.weight",
            "classification.1.bias": "final_layer.clf.weight",
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # ======== Handling EEG input ========================
        self.add_module(
            "input_preprocess",
            nn.Sequential(Ensure4d(), Rearrange("ba ch t 1 -> ba 1 ch t")),
        )
        # ======== Inception branches ========================
        block11 = self._get_inception_branch(
            in_channels=self.n_chans,
            out_channels=n_filters_time,
            kernel_length=kernel_length,
            activation=activation,
        )
        block12 = self._get_inception_branch(
            in_channels=self.n_chans,
            out_channels=n_filters_time * 2,
            kernel_length=kernel_length * 2,
            activation=activation,
        )
        block13 = self._get_inception_branch(
            in_channels=self.n_chans,
            out_channels=n_filters_time * 4,
            kernel_length=n_filters_time * 4,
            activation=activation,
        )
        self.add_module("inception_block", InceptionBlock((block11, block12, block13)))
        self.pool1 = self.add_module(
            "pooling",
            nn.Sequential(
                nn.AvgPool2d(kernel_size=(1, pool_kernel)), nn.Dropout(drop_prob)
            ),
        )
        # =========== TC blocks =====================
        self.add_module(
            "TC_block1",
            _TCBlock(
                in_ch=tcn_in_channel,
                kernel_length=tcn_kernel_size,
                dilatation=tcn_dilatation,
                padding=tcn_padding,
                drop_prob=drop_prob,
                activation=activation,
            ),
        )
        # ================================
        self.add_module(
            "TC_block2",
            _TCBlock(
                in_ch=tcn_in_channel,
                kernel_length=tcn_kernel_size,
                dilatation=tcn_dilatation * 2,
                padding=tcn_padding * 2,
                drop_prob=drop_prob,
                activation=activation,
            ),
        )
        # ================================
        self.add_module(
            "TC_block3",
            _TCBlock(
                in_ch=tcn_in_channel,
                kernel_length=tcn_kernel_size,
                dilatation=tcn_dilatation * 4,
                padding=tcn_padding * 4,
                drop_prob=drop_prob,
                activation=activation,
            ),
        )
        # ================================
        self.add_module(
            "TC_block4",
            _TCBlock(
                in_ch=tcn_in_channel,
                kernel_length=tcn_kernel_size,
                dilatation=tcn_dilatation * 8,
                padding=tcn_padding * 8,
                drop_prob=drop_prob,
                activation=activation,
            ),
        )

        # ============= Dimensionality reduction ===================
        self.add_module(
            "dim_reduction",
            nn.Sequential(
                nn.Conv2d(tcn_in_channel, tcn_in_channel * 2, kernel_size=(1, 1)),
                nn.BatchNorm2d(tcn_in_channel * 2),
                activation(),
                nn.AvgPool2d((1, tcn_kernel_size)),
                nn.Dropout(drop_prob),
            ),
        )
        # ============== Classifier ==================
        # Moved flatten to another layer
        self.add_module("flatten", nn.Flatten())

        num_features = self.get_output_shape()[-1]

        self.add_module("final_layer", nn.Linear(num_features, self.n_outputs))

    @staticmethod
    def _get_inception_branch(
        in_channels,
        out_channels,
        kernel_length,
        depth_multiplier=1,
        activation: nn.Module = nn.ELU,
    ):
        return nn.Sequential(
            nn.Conv2d(
                1,
                out_channels,
                kernel_size=(1, kernel_length),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            DepthwiseConv2d(
                out_channels,
                kernel_size=(in_channels, 1),
                depth_multiplier=depth_multiplier,
                bias=False,
                padding="valid",
            ),
            nn.BatchNorm2d(out_channels),
            activation(),
        )


class _TCBlock(nn.Module):
    """
    Temporal Convolutional (TC) block.

    This module applies two depthwise separable convolutions with dilation and residual
    connections, commonly used in temporal convolutional networks to capture long-range
    dependencies in time-series data.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    kernel_length : int
        Length of the convolutional kernels.
    dilatation : int
        Dilatation rate for the convolutions.
    padding : int
        Amount of padding to add to the input.
    drop_prob : float, optional
        Dropout probability. Default is 0.4.
    activation : nn.Module class, optional
        Activation function class to use. Should be a PyTorch activation module class
        like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    """

    def __init__(
        self,
        in_ch,
        kernel_length,
        dilatation,
        padding,
        drop_prob=0.4,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.pad = padding
        self.tc1 = nn.Sequential(
            DepthwiseConv2d(
                in_ch,
                kernel_size=(1, kernel_length),
                depth_multiplier=1,
                dilation=(1, dilatation),
                bias=False,
                padding="valid",
            ),
            nn.BatchNorm2d(in_ch),
            activation(),
            nn.Dropout(drop_prob),
        )

        self.tc2 = nn.Sequential(
            DepthwiseConv2d(
                in_ch,
                kernel_size=(1, kernel_length),
                depth_multiplier=1,
                dilation=(1, dilatation),
                bias=False,
                padding="valid",
            ),
            nn.BatchNorm2d(in_ch),
            activation(),
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        residual = x
        paddings = (self.pad, 0, 0, 0, 0, 0, 0, 0)
        x = nn.functional.pad(x, paddings)
        x = self.tc1(x)
        x = nn.functional.pad(x, paddings)
        x = self.tc2(x) + residual
        return x

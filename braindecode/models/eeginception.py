# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import torch
import torch.nn.functional as F

from torch import nn
from .modules import Expression, Ensure4d
from .functions import squeeze_final_output
from .eegnet import _glorot_weight_zero_bias


class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class _InceptionBlock1(nn.Module):

    def __init__(self, drop_prob, n_channels, scales_samples, n_filters=8,
                 momentum=0.01, activation=nn.ELU()):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_filters = n_filters
        self.momentum = momentum
        self.activation = activation
        self.n_channels = n_channels
        self.scales_samples = scales_samples

        self.inception_block1 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 2 - 1, scales_sample // 2,)),
                nn.Conv2d(1, self.n_filters, (scales_sample, 1)),
                nn.BatchNorm2d(self.n_filters, momentum=self.momentum),
                activation,
                nn.Dropout(self.drop_prob),
                Conv2dWithConstraint(self.n_filters,
                                     self.n_filters * 2,
                                     (1, self.n_channels),
                                     max_norm=1,
                                     stride=1,
                                     bias=False,
                                     groups=self.n_filters,
                                     padding=(0, 0),
                                     ),
                nn.BatchNorm2d(self.n_filters * 2, momentum=self.momentum),
                activation,
                nn.Dropout(self.drop_prob),
            ) for scales_sample in scales_samples
        ])

    def forward(self, x):
        return torch.cat([net(x) for net in self.inception_block1], 1)


class _InceptionBlock2(nn.Module):

    def __init__(self, n_filters, scales_samples, drop_prob,
                 activation=nn.ELU, momentum=0.001):
        super().__init__()
        self.inception_block2 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 8 -
                           1, scales_sample // 8,)),
                nn.Conv2d(
                    len(scales_samples) * 2 * n_filters,
                    n_filters, (scales_sample // 4, 1),
                    bias=False
                ),
                nn.BatchNorm2d(n_filters, momentum=momentum),
                activation,
                nn.Dropout(drop_prob),
            ) for scales_sample in scales_samples
        ])

    def forward(self, x):
        return torch.cat([net(x) for net in self.inception_block2], 1)


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


class EEGInception(nn.Sequential):
    """EEG Inception model from Santamaría-Vázquez, E. et al 2020.

    EEG Inception for ERP-based classification described in [Santamaria2020]]_.
    The code for the paper and this model is also available at [Santamaria2020]_
    and an adaptation for PyTorch [2]_.

    The model is strongly based on the original InceptionNet for an image. The main goal is
    to extract features in parallel with different scales. The authors extracted three scales
    proportional to the window sample size. The network had three parts:
    1-larger inception block largest, 2-smaller inception block followed by 3-bottleneck
    for classification.

    One advantage of the EEG-Inception block is that it allows a network
    to learn simultaneous components of low and high frequency associated with the signal.

    The model is fully described in [Santamaria2020]_.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper based on [2]_.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_classes : int
        Number of classes.
    input_size_ms : int
        Size of the input, in milliseconds. Set to 1000 in [1]_.
    sfreq : float
        EEG sampling frequency.
    drop_prob : float
        Dropout rate inside all the network.
    scales_time: list(int)
        Windows for inception block, must be a list with proportional values of
        the input_size_ms, in ms.
        According with the authors: temporal scale (ms) of the convolutions
        on each Inception module.
        This parameter determines the kernel sizes of the filters.
    n_filters : int
        Initial number of convolutional filters. Set to 8 in [Santamaria2020]_.
    activation: nn.Module
        Activation function, default: ELU activation.
    batch_norm_alpha: float
        Momentum for BatchNorm2d.

    References
    ----------
    .. [Santamaria2020] Santamaria-Vazquez, E., Martinez-Cagigal, V.,
       Vaquerizo-Villar, F., & Hornero, R. (2020).
       EEG-inception: A novel deep convolutional neural network for assistive
       ERP-based brain-computer interfaces.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering , v. 28.
       Online: http://dx.doi.org/10.1109/TNSRE.2020.3048106
    .. [2]  Grifcc. Implementation of the EEGInception in torch (2022).
       Online: https://github.com/Grifcc/EEG/tree/90e412a407c5242dfc953d5ffb490bdb32faf022
    """

    def __init__(
            self,
            n_channels,
            n_classes,
            input_size_ms=1000,
            sfreq=128,
            drop_prob=0.5,
            scales_samples=(64, 32, 16),
            n_filters=8,
            activation=nn.ELU(),
            batch_norm_alpha=0.01,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_size_ms = input_size_ms
        self.drop_prob = drop_prob
        self.sfreq = sfreq
        self.n_filters = n_filters
        self.scales_samples = scales_samples
        self.activation = activation
        self.batch_norm_alpha = batch_norm_alpha

        self.add_module("ensuredims", Ensure4d())

        self.add_module("dimshuffle", Expression(_transpose_to_b_1_c_0))

        self.add_module("inception_block_1", _InceptionBlock1(scales_samples=self.scales_samples,
                                                              n_filters=self.n_filters,
                                                              drop_prob=self.drop_prob,
                                                              momentum=self.batch_norm_alpha,
                                                              activation=self.activation,
                                                              n_channels=self.n_channels))

        self.add_module("avg_pool_1", nn.AvgPool2d((4, 1)))

        self.add_module("inception_block_2", _InceptionBlock2(n_filters=self.n_filters,
                                                              scales_samples=self.scales_samples,
                                                              drop_prob=self.drop_prob,
                                                              momentum=self.batch_norm_alpha,
                                                              activation=activation))
        self.add_module("avg_pool_2", nn.AvgPool2d((2, 1)))

        self.add_module("squeeze", Expression(expression_fn=squeeze_final_output))

        self.add_module("final_block", nn.Sequential(

            CustomPad((0, 0, 4, 3)),
            nn.Conv2d(
                24, n_filters * len(scales_samples) // 2, (8, 1),
                bias=False
            ),
            nn.BatchNorm2d(n_filters * len(scales_samples) // 2,
                           momentum=self.batch_norm_alpha),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(self.drop_prob),

            CustomPad((0, 0, 2, 1)),
            nn.Conv2d(
                12, n_filters * len(scales_samples) // 4, (4, 1),
                bias=False
            ),
            nn.BatchNorm2d(n_filters * len(scales_samples) // 4,
                           momentum=self.batch_norm_alpha),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(self.drop_prob),
        ))

        self.add_module("flatten", nn.Flatten(start_dim=1))

        self.add_module("classification", nn.Sequential(
            nn.Linear(4 * 1 * 6, self.n_classes),
            nn.Softmax(1)
        ))

        _glorot_weight_zero_bias(self)

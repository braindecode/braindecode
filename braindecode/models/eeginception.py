# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import torch
import torch.nn.functional as F

from torch import nn


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


class EEGInception(nn.Module):
    """EEG Inception model from Santamaría-Vázquez, E. et al 2020.

    EEG Inception for ERP-based classification described in [Santamaria2020]]_.
    The code for the paper and this model is also available at [Santamaria2020]_
    and an adaptation for PyTorch [1]_.

    The model is highly based on the original InceptionNet for an image. The main goal is
    to extract features in parallel with different scales. The author extracted three scales
    proportional to the window sample size. The network had three parts:
    1-inception block largest, 2--inception block smaller and bottleneck for classification.

    One advantage of the module such EEG-Inception for EEG is that it allows a network
    to learn simultaneous components of low and high frequency associated with the signal.

    The model is fully described in [Santamaria2020]_.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_classes : int
        Number of classes.
    input_size_s : int
        Size of the input, in milliseconds. Set to 1000 in [1]_.
    sfreq : float
        EEG sampling frequency.
    drop_prob : float
        Dropout rate inside all the network.
    n_filters : int
        Initial number of convolutional filters. Set to 8 in [Santamaria2020]_.
    scales_time: list(int)
        Windows for inception block, must be a list with proportional values of
        the input_window_samples.
        Acording with the author: temporal scale (ms) of the convolutions
        on each Inception module.
        This parameter determines the kernel sizes of the filters.
    activation: nn.Module
        Activation function as an parameter, ELU activation.

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
            n_channels,  # In the original implementation named as ncha8
            n_classes,  # n_classes
            input_size_s=1000,  # input_time
            sfreq=128,
            drop_prob=0.5,
            scales_time=(500, 250, 125),
            n_filters=8,
            activation=nn.ELU(inplace=True)
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_size_s = input_size_s
        self.drop_prob = drop_prob
        self.sfreq = sfreq
        self.n_filters = n_filters

        input_samples = int(input_size_s * sfreq / 1000)
        scales_samples = [int(s * sfreq / input_size_s) for s in scales_time]

        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception_block1 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 2 - 1, scales_sample // 2,)),
                nn.Conv2d(1, n_filters, (scales_sample, 1)),
                # kernel_initializer='he_normal',padding='same'
                nn.BatchNorm2d(n_filters, momentum=0.01),
                activation,
                nn.Dropout(drop_prob),
                Conv2dWithConstraint(n_filters,
                                     n_filters * 2,
                                     (1, n_channels),
                                     max_norm=1,
                                     stride=1,
                                     bias=False,
                                     groups=self.n_filters,
                                     padding=(0, 0),
                                     ),
                nn.BatchNorm2d(n_filters * 2, momentum=0.01),
                activation,
                nn.Dropout(drop_prob),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool1 = nn.AvgPool2d((4, 1))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception_block2 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 8 -
                           1, scales_sample // 8,)),
                nn.Conv2d(
                    len(scales_samples) * 2 * n_filters,
                    n_filters, (scales_sample // 4, 1),
                    bias=False  # kernel_initializer='he_normal', padding='same'
                ),
                nn.BatchNorm2d(n_filters, momentum=0.01),
                activation,
                nn.Dropout(drop_prob),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(

            CustomPad((0, 0, 4, 3)),
            nn.Conv2d(
                24, n_filters * len(scales_samples) // 2, (8, 1),
                bias=False
            ),  # kernel_initializer='he_normal', padding='same'
            nn.BatchNorm2d(n_filters * len(scales_samples) // 2, momentum=0.01),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(drop_prob),

            CustomPad((0, 0, 2, 1)),
            nn.Conv2d(
                12, n_filters * len(scales_samples) // 4, (4, 1),
                bias=False

            ),  # kernel_initializer='he_normal', padding='same'
            nn.BatchNorm2d(n_filters * len(scales_samples) // 4, momentum=0.01),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(drop_prob),
        )

        self.dense = nn.Sequential(
            nn.Linear(4 * 1 * 6, n_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception_block1], 1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception_block2], 1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x

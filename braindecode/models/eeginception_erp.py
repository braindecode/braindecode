# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Cedric Rommel <cedric.rommel@inria.fr>
#
# License: BSD (3-clause)
from numpy import prod

from torch import nn
from einops.layers.torch import Rearrange

from .modules import Ensure4d
from .eegnet import _glorot_weight_zero_bias
from .eegitnet import _InceptionBlock, _DepthwiseConv2d
from .base import EEGModuleMixin, deprecated_args


class EEGInceptionERP(EEGModuleMixin, nn.Sequential):
    """EEG Inception for ERP-based classification

    The code for the paper and this model is also available at [Santamaria2020]_
    and an adaptation for PyTorch [2]_.

    The model is strongly based on the original InceptionNet for an image. The main goal is
    to extract features in parallel with different scales. The authors extracted three scales
    proportional to the window sample size. The network had three parts:
    1-larger inception block largest, 2-smaller inception block followed by 3-bottleneck
    for classification.

    One advantage of the EEG-Inception block is that it allows a network
    to learn simultaneous components of low and high frequency associated with the signal.
    The winners of BEETL Competition/NeurIps 2021 used parts of the model [beetl]_.

    The model is fully described in [Santamaria2020]_.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper based on [2]_.

    Parameters
    ----------
    n_times : int, optional
        Size of the input, in number of sampels. Set to 128 (1s) as in
        [Santamaria2020]_.
    sfreq : float, optional
        EEG sampling frequency. Defaults to 128 as in [Santamaria2020]_.
    drop_prob : float, optional
        Dropout rate inside all the network. Defaults to 0.5 as in
        [Santamaria2020]_.
    scales_samples_s: list(float), optional
        Windows for inception block. Temporal scale (s) of the convolutions on
        each Inception module. This parameter determines the kernel sizes of
        the filters. Defaults to 0.5, 0.25, 0.125 seconds, as in
        [Santamaria2020]_.
    n_filters : int, optional
        Initial number of convolutional filters. Defaults to 8 as in
        [Santamaria2020]_.
    activation: nn.Module, optional
        Activation function. Defaults to ELU activation as in
        [Santamaria2020]_.
    batch_norm_alpha: float, optional
        Momentum for BatchNorm2d. Defaults to 0.01.
    depth_multiplier: int, optional
        Depth multiplier for the depthwise convolution. Defaults to 2 as in
        [Santamaria2020]_.
    pooling_sizes: list(int), optional
        Pooling sizes for the inception blocks. Defaults to 4, 2, 2 and 2, as
        in [Santamaria2020]_.
    in_channels : int
        Alias for n_chans.
    n_classes : int
        Alias for n_outputs.
    input_window_samples : int
        Alias for n_times.

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
    .. [beetl]_ Wei, X., Faisal, A.A., Grosse-Wentrup, M., Gramfort, A., Chevallier, S.,
       Jayaram, V., Jeunet, C., Bakas, S., Ludwig, S., Barmpas, K., Bahri, M., Panagakis,
       Y., Laskaris, N., Adamos, D.A., Zafeiriou, S., Duong, W.C., Gordon, S.M.,
       Lawhern, V.J., Śliwowski, M., Rouanne, V. &amp; Tempczyk, P.. (2022).
       2021 BEETL Competition: Advancing Transfer Learning for Subject Independence &amp;
       Heterogenous EEG Data Sets. <i>Proceedings of the NeurIPS 2021 Competitions and
       Demonstrations Track</i>, in <i>Proceedings of Machine Learning Research</i>
       176:205-219 Available from https://proceedings.mlr.press/v176/wei22a.html.

    """

    def __init__(
            self,
            n_chans=None,
            n_outputs=None,
            n_times=1000,
            sfreq=128,
            drop_prob=0.5,
            scales_samples_s=(0.5, 0.25, 0.125),
            n_filters=8,
            activation=nn.ELU(),
            batch_norm_alpha=0.01,
            depth_multiplier=2,
            pooling_sizes=(4, 2, 2, 2),
            chs_info=None,
            input_window_seconds=None,
            in_channels=None,
            n_classes=None,
            input_window_samples=None,
            add_log_softmax=True,
    ):
        n_chans, n_outputs, n_times, = deprecated_args(
            self,
            ('in_channels', 'n_chans', in_channels, n_chans),
            ('n_classes', 'n_outputs', n_classes, n_outputs),
            ('input_window_samples', 'n_times', input_window_samples, n_times),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_channels, n_classes, input_window_samples
        self.drop_prob = drop_prob
        self.n_filters = n_filters
        self.scales_samples_s = scales_samples_s
        self.scales_samples = tuple(
            int(size_s * self.sfreq) for size_s in self.scales_samples_s)
        self.activation = activation
        self.alpha_momentum = batch_norm_alpha
        self.depth_multiplier = depth_multiplier
        self.pooling_sizes = pooling_sizes

        self.mapping = {
            'classification.1.weight': 'final_layer.fc.weight',
            'classification.1.bias': 'final_layer.fc.bias'}

        self.add_module("ensuredims", Ensure4d())

        self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 C T"))

        # ======== Inception branches ========================
        block11 = self._get_inception_branch_1(
            in_channels=self.n_chans,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[0],
            alpha_momentum=self.alpha_momentum,
            activation=self.activation,
            drop_prob=self.drop_prob,
            depth_multiplier=self.depth_multiplier,
        )
        block12 = self._get_inception_branch_1(
            in_channels=self.n_chans,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[1],
            alpha_momentum=self.alpha_momentum,
            activation=self.activation,
            drop_prob=self.drop_prob,
            depth_multiplier=self.depth_multiplier,
        )
        block13 = self._get_inception_branch_1(
            in_channels=self.n_chans,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[2],
            alpha_momentum=self.alpha_momentum,
            activation=self.activation,
            drop_prob=self.drop_prob,
            depth_multiplier=self.depth_multiplier,
        )

        self.add_module("inception_block_1", _InceptionBlock((block11, block12, block13)))

        self.add_module("avg_pool_1", nn.AvgPool2d((1, self.pooling_sizes[0])))

        # ======== Inception branches ========================
        n_concat_filters = len(self.scales_samples) * self.n_filters
        n_concat_dw_filters = n_concat_filters * self.depth_multiplier
        block21 = self._get_inception_branch_2(
            in_channels=n_concat_dw_filters,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[0] // 4,
            alpha_momentum=self.alpha_momentum,
            activation=self.activation,
            drop_prob=self.drop_prob
        )
        block22 = self._get_inception_branch_2(
            in_channels=n_concat_dw_filters,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[1] // 4,
            alpha_momentum=self.alpha_momentum,
            activation=self.activation,
            drop_prob=self.drop_prob
        )
        block23 = self._get_inception_branch_2(
            in_channels=n_concat_dw_filters,
            out_channels=self.n_filters,
            kernel_length=self.scales_samples[2] // 4,
            alpha_momentum=self.alpha_momentum,
            activation=self.activation,
            drop_prob=self.drop_prob
        )

        self.add_module(
            "inception_block_2", _InceptionBlock((block21, block22, block23)))

        self.add_module("avg_pool_2", nn.AvgPool2d((1, self.pooling_sizes[1])))

        self.add_module("final_block", nn.Sequential(
            nn.Conv2d(
                n_concat_filters,
                n_concat_filters // 2,
                (1, 8),
                padding="same",
                bias=False
            ),
            nn.BatchNorm2d(n_concat_filters // 2,
                           momentum=self.alpha_momentum),
            activation,
            nn.Dropout(self.drop_prob),
            nn.AvgPool2d((1, self.pooling_sizes[2])),

            nn.Conv2d(
                n_concat_filters // 2,
                n_concat_filters // 4,
                (1, 4),
                padding="same",
                bias=False
            ),
            nn.BatchNorm2d(n_concat_filters // 4,
                           momentum=self.alpha_momentum),
            activation,
            nn.Dropout(self.drop_prob),
            nn.AvgPool2d((1, self.pooling_sizes[3])),
        ))

        spatial_dim_last_layer = (
                self.n_times // prod(self.pooling_sizes))
        n_channels_last_layer = self.n_filters * len(self.scales_samples) // 4

        self.add_module("flat", nn.Flatten())

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module("fc",
                          nn.Linear(
                              spatial_dim_last_layer * n_channels_last_layer,
                              self.n_outputs
                          ), )

        if self.add_log_softmax:
            module.add_module("logsoftmax", nn.LogSoftmax(dim=1))
        else:
            module.add_module("identity", nn.Identity())

        self.add_module("final_layer", module)

        _glorot_weight_zero_bias(self)

    @staticmethod
    def _get_inception_branch_1(in_channels, out_channels, kernel_length,
                                alpha_momentum, drop_prob, activation,
                                depth_multiplier):
        return nn.Sequential(
            nn.Conv2d(
                1,
                out_channels,
                kernel_size=(1, kernel_length),
                padding="same",
                bias=True
            ),
            nn.BatchNorm2d(out_channels, momentum=alpha_momentum),
            activation,
            nn.Dropout(drop_prob),
            _DepthwiseConv2d(
                out_channels,
                kernel_size=(in_channels, 1),
                depth_multiplier=depth_multiplier,
                bias=False,
                padding="valid",
            ),
            nn.BatchNorm2d(
                depth_multiplier * out_channels,
                momentum=alpha_momentum
            ),
            activation,
            nn.Dropout(drop_prob),
        )

    @staticmethod
    def _get_inception_branch_2(in_channels, out_channels, kernel_length,
                                alpha_momentum, drop_prob, activation):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_length),
                padding="same",
                bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum=alpha_momentum),
            activation,
            nn.Dropout(drop_prob),
        )

# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)
from collections import OrderedDict

import numpy as np
from torch import nn
from torch.nn import init
from einops.layers.torch import Rearrange

from ..util import np_to_th
from .modules import Expression, Ensure4d, CombinedConv
from .functions import safe_log, square, squeeze_final_output
from .base import EEGModuleMixin, deprecated_args


class ShallowFBCSPNet(EEGModuleMixin, nn.Sequential):
    """Shallow ConvNet model from Schirrmeister et al 2017.

    Model described in [Schirrmeister2017]_.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters.
    filter_time_length: int
        Length of the temporal filter.
    n_filters_spat: int
        Number of spatial filters.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    final_conv_length: int | str
        Length of the final convolution layer.
        If set to "auto", length of the input signal must be specified.
    conv_nonlin: callable
        Non-linear function to be used after convolution layers.
    pool_mode: str
        Method to use on pooling layers. "max" or "mean".
    pool_nonlin: callable
        Non-linear function to be used after pooling layers.
    split_first_layer: bool
        Split first layer into temporal and spatial layers (True) or just use temporal (False).
        There would be no non-linearity between the split layers.
    batch_norm: bool
        Whether to use batch normalisation.
    batch_norm_alpha: float
        Momentum for BatchNorm2d.
    drop_prob: float
        Dropout probability.
    in_chans : int
        Alias for `n_chans`.
    n_classes: int
        Alias for `n_outputs`.
    input_window_samples: int | None
        Alias for `n_times`.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
            self,
            n_chans=None,
            n_outputs=None,
            n_times=None,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            final_conv_length=30,
            conv_nonlin=square,
            pool_mode="mean",
            pool_nonlin=safe_log,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
            chs_info=None,
            input_window_seconds=None,
            sfreq=None,
            in_chans=None,
            n_classes=None,
            input_window_samples=None,
    ):
        n_chans, n_outputs, n_times = deprecated_args(
            self,
            ("in_chans", "n_chans", in_chans, n_chans),
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("input_window_samples", "n_times", input_window_samples, n_times),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples
        if final_conv_length == "auto":
            assert self.n_times is not None
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob

        self.add_module("ensuredims", Ensure4d())
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 T C"))
            self.add_module(
                "conv_time_spat",
                CombinedConv(
                    in_chans=self.n_chans,
                    n_filters_time=self.n_filters_time,
                    n_filters_spat=self.n_filters_spat,
                    filter_time_length=filter_time_length,
                    bias_time=True,
                    bias_spat=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.n_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        self.add_module("conv_nonlin_exp", Expression(self.conv_nonlin))
        self.add_module(
            "pool",
            pool_class(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
        )
        self.add_module("pool_nonlin_exp", Expression(self.pool_nonlin))
        self.add_module("drop", nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(
                np_to_th(
                    np.ones(
                        (1, self.n_chans, self.n_times, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                self.n_outputs,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time_spat.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time_spat.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_time_spat.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_time_spat.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Wrapper to allow for loading of a state_dict from a model before CombinedConv was
         implemented"""
        keys_to_change = [
            "conv_time.weight",
            "conv_spat.weight",
            "conv_time.bias",
            "conv_spat.bias",
        ]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in keys_to_change:
                k = f"conv_time_spat.{k}"
            new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, *args, **kwargs)

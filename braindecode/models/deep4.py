# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from .base import EEGModuleMixin, deprecated_args
from .functions import identity, squeeze_final_output
from .modules import AvgPool2dWithConv, CombinedConv, Ensure4d, Expression


class Deep4Net(EEGModuleMixin, nn.Sequential):
    """Deep ConvNet model from Schirrmeister et al 2017.

    Model described in [Schirrmeister2017]_.

    Parameters
    ----------
    final_conv_length: int | str
        Length of the final convolution layer.
        If set to "auto", n_times must not be None. Default: "auto".
    n_filters_time: int
        Number of temporal filters.
    n_filters_spat: int
        Number of spatial filters.
    filter_time_length: int
        Length of the temporal filter in layer 1.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    n_filters_2: int
        Number of temporal filters in layer 2.
    filter_length_2: int
        Length of the temporal filter in layer 2.
    n_filters_3: int
        Number of temporal filters in layer 3.
    filter_length_3: int
        Length of the temporal filter in layer 3.
    n_filters_4: int
        Number of temporal filters in layer 4.
    filter_length_4: int
        Length of the temporal filter in layer 4.
    first_conv_nonlin: callable
        Non-linear activation function to be used after convolution in layer 1.
    first_pool_mode: str
        Pooling mode in layer 1. "max" or "mean".
    first_pool_nonlin: callable
        Non-linear activation function to be used after pooling in layer 1.
    later_conv_nonlin: callable
        Non-linear activation function to be used after convolution in later layers.
    later_pool_mode: str
        Pooling mode in later layers. "max" or "mean".
    later_pool_nonlin: callable
        Non-linear activation function to be used after pooling in later layers.
    drop_prob: float
        Dropout probability.
    split_first_layer: bool
        Split first layer into temporal and spatial layers (True) or just use temporal (False).
        There would be no non-linearity between the split layers.
    batch_norm: bool
        Whether to use batch normalisation.
    batch_norm_alpha: float
        Momentum for BatchNorm2d.
    stride_before_pool: bool
        Stride before pooling.
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.


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
            final_conv_length="auto",
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=50,
            filter_length_2=10,
            n_filters_3=100,
            filter_length_3=10,
            n_filters_4=200,
            filter_length_4=10,
            first_conv_nonlin=elu,
            first_pool_mode="max",
            first_pool_nonlin=identity,
            later_conv_nonlin=elu,
            later_pool_mode="max",
            later_pool_nonlin=identity,
            drop_prob=0.5,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False,
            chs_info=None,
            input_window_seconds=None,
            sfreq=None,
            in_chans=None,
            n_classes=None,
            input_window_samples=None,
            add_log_softmax=True,
    ):
        n_chans, n_outputs, n_times = deprecated_args(
            self,
            ('in_chans', 'n_chans', in_chans, n_chans),
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
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples
        if final_conv_length == "auto":
            assert self.n_times is not None
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_conv_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_conv_nonlin = later_conv_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        # For the load_state_dict
        # When padronize all layers,
        # add the old's parameters here
        self.mapping = {
            "conv_time.weight": "conv_time_spat.conv_time.weight",
            "conv_spat.weight": "conv_time_spat.conv_spat.weight",
            "conv_time.bias": "conv_time_spat.conv_time.bias",
            "conv_spat.bias": "conv_time_spat.conv_spat.bias",
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias"
        }

        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        self.add_module("ensuredims", Ensure4d())
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
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
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        self.add_module("conv_nonlin", Expression(self.first_nonlin))
        self.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        self.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

        def add_conv_pool_block(
                model, n_filters_before, n_filters, filter_length, block_nr
        ):
            suffix = "_{:d}".format(block_nr)
            self.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            self.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                self.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            self.add_module("nonlin" + suffix, Expression(self.later_conv_nonlin))

            self.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            self.add_module("pool_nonlin" + suffix, Expression(self.later_pool_nonlin))

        add_conv_pool_block(
            self, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            self, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            self, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            self.final_conv_length = self.get_output_shape()[2]

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module("conv_classifier",
                          nn.Conv2d(
                            self.n_filters_4,
                            self.n_outputs,
                            (self.final_conv_length, 1),
                            bias=True, ))

        if self.add_log_softmax:
            self.add_module("logsoftmax", nn.LogSoftmax(dim=1))

        module.add_module("squeeze", Expression(squeeze_final_output))

        self.add_module("final_layer", module)

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time_spat.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time_spat.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_time_spat.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.final_layer.conv_classifier.weight, gain=1)
        init.constant_(self.final_layer.conv_classifier.bias, 0)

        # Start in eval mode
        self.eval()

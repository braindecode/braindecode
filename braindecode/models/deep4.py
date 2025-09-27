# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn
from torch.nn import init

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    AvgPool2dWithConv,
    CombinedConv,
    Ensure4d,
    SqueezeFinalOutput,
)


class Deep4Net(EEGModuleMixin, nn.Sequential):
    """Deep ConvNet model from Schirrmeister et al (2017) [Schirrmeister2017]_.

    :bdg-success:`Convolution`

    .. figure:: https://onlinelibrary.wiley.com/cms/asset/fc200ccc-d8c4-45b4-8577-56ce4d15999a/hbm23730-fig-0001-m.jpg
        :align: center
        :alt: Deep4Net Architecture
        :width: 600px


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
    activation_first_conv_nonlin: nn.Module, default is nn.ELU
        Non-linear activation function to be used after convolution in layer 1.
    first_pool_mode: str
        Pooling mode in layer 1. "max" or "mean".
    first_pool_nonlin: callable
        Non-linear activation function to be used after pooling in layer 1.
    activation_later_conv_nonlin: nn.Module, default is nn.ELU
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
        activation_first_conv_nonlin: nn.Module = nn.ELU,
        first_pool_mode="max",
        first_pool_nonlin: nn.Module = nn.Identity,
        activation_later_conv_nonlin: nn.Module = nn.ELU,
        later_pool_mode="max",
        later_pool_nonlin: nn.Module = nn.Identity,
        drop_prob=0.5,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
        # Braindecode EEGModuleMixin parameters
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
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
        self.first_nonlin = activation_first_conv_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_conv_nonlin = activation_later_conv_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        min_n_times = self._get_min_n_times()
        if self.n_times < min_n_times:
            scaling_factor = self.n_times / min_n_times
            warn(
                f"n_times ({self.n_times}) is smaller than the minimum required "
                f"({min_n_times}) for the current model parameters configuration. "
                "Adjusting parameters to ensure compatibility."
                "Reducing the kernel, pooling, and stride sizes accordingly."
                "Scaling factor: {:.2f}".format(scaling_factor),
                UserWarning,
            )
            # Calculate a scaling factor to adjust temporal parameters
            # Apply the scaling factor to all temporal kernel and pooling sizes
            self.filter_time_length = max(
                1, int(self.filter_time_length * scaling_factor)
            )
            self.pool_time_length = max(1, int(self.pool_time_length * scaling_factor))
            self.pool_time_stride = max(1, int(self.pool_time_stride * scaling_factor))
            self.filter_length_2 = max(1, int(self.filter_length_2 * scaling_factor))
            self.filter_length_3 = max(1, int(self.filter_length_3 * scaling_factor))
            self.filter_length_4 = max(1, int(self.filter_length_4 * scaling_factor))
        # For the load_state_dict
        # When padronize all layers,
        # add the old's parameters here
        self.mapping = {
            "conv_time.weight": "conv_time_spat.conv_time.weight",
            "conv_spat.weight": "conv_time_spat.conv_spat.weight",
            "conv_time.bias": "conv_time_spat.conv_time.bias",
            "conv_spat.bias": "conv_time_spat.conv_spat.bias",
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias",
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
        self.add_module("conv_nonlin", self.first_nonlin())
        self.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        self.add_module("pool_nonlin", self.first_pool_nonlin())

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
            self.add_module("nonlin" + suffix, self.later_conv_nonlin())

            self.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            self.add_module("pool_nonlin" + suffix, self.later_pool_nonlin())

        add_conv_pool_block(
            self, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            self, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            self, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        self.eval()
        if self.final_conv_length == "auto":
            self.final_conv_length = self.get_output_shape()[2]

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.n_filters_4,
                self.n_outputs,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )

        module.add_module("squeeze", SqueezeFinalOutput())

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
                init.constant_(self.conv_time_spat.conv_spat.bias, 0)
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

        self.train()

    def _get_min_n_times(self) -> int:
        """
        Calculate the minimum number of time samples required for the model
        to work with the given temporal parameters.
        """
        # Start with the minimum valid output length of the network (1)
        min_len = 1

        # List of conv kernel sizes and pool parameters for the 4 blocks, in reverse order
        # Each tuple: (filter_length, pool_length, pool_stride)
        block_params = [
            (self.filter_length_4, self.pool_time_length, self.pool_time_stride),
            (self.filter_length_3, self.pool_time_length, self.pool_time_stride),
            (self.filter_length_2, self.pool_time_length, self.pool_time_stride),
            (self.filter_time_length, self.pool_time_length, self.pool_time_stride),
        ]

        # Work backward from the last layer to the input
        for filter_len, pool_len, pool_stride in block_params:
            # Reverse the pooling operation
            # L_in = stride * (L_out - 1) + kernel_size
            min_len = pool_stride * (min_len - 1) + pool_len
            # Reverse the convolution operation (assuming stride=1)
            # L_in = L_out + kernel_size - 1
            min_len = min_len + filter_len - 1

        return min_len

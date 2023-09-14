# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.functional import elu

from .base import EEGModuleMixin, deprecated_args
from .functions import squeeze_final_output
from .modules import Ensure4d, Expression


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNetv4(EEGModuleMixin, nn.Sequential):
    """EEGNet v4 model from Lawhern et al 2018.

    See details in [EEGNet4]_.

    Parameters
    ----------
    final_conv_length : int | "auto"
        If int, final length of convolutional filters.
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------
    .. [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(
            self,
            n_chans=None,
            n_outputs=None,
            n_times=None,
            final_conv_length="auto",
            pool_mode="mean",
            F1=8,
            D=2,
            F2=16,  # usually set to F1*D (?)
            kernel_length=64,
            third_kernel_size=(8, 4),
            drop_prob=0.25,
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
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob
        # For the load_state_dict
        # When padronize all layers,
        # add the old's parameters here
        self.mapping = {
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias"
        }

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())

        self.add_module("dimshuffle",
                        Rearrange("batch ch t 1 -> batch 1 ch t"))
        self.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
        )
        self.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.n_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
        )

        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3
            ),
        )
        self.add_module("elu_1", Expression(elu))

        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 16),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, 16 // 2),
            ),
        )
        self.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
        )

        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        output_shape = self.get_output_shape()
        n_out_virtual_chans = output_shape[2]

        if self.final_conv_length == "auto":
            n_out_time = output_shape[3]
            self.final_conv_length = n_out_time

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module("conv_classifier",
                          nn.Conv2d(self.F2, self.n_outputs,
                                    (n_out_virtual_chans, self.final_conv_length), bias=True, ))

        if self.add_log_softmax:
            module.add_module("logsoftmax", nn.LogSoftmax(dim=1))

        # Transpose back to the logic of braindecode,
        # so time in third dimension (axis=2)
        module.add_module("permute_back", Rearrange("batch x y z -> batch x z y"), )

        module.add_module("squeeze", Expression(squeeze_final_output))

        self.add_module("final_layer", module)

        _glorot_weight_zero_bias(self)


class EEGNetv1(EEGModuleMixin, nn.Sequential):
    """EEGNet model from Lawhern et al. 2016.

    See details in [EEGNet]_.

    Parameters
    ----------
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------
    .. [EEGNet] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2016).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(
            self,
            n_chans=None,
            n_outputs=None,
            n_times=None,
            final_conv_length="auto",
            pool_mode="max",
            second_kernel_size=(2, 32),
            third_kernel_size=(8, 4),
            drop_prob=0.25,
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
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples
        if final_conv_length == "auto":
            assert self.n_times is not None
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.second_kernel_size = second_kernel_size
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob
        # For the load_state_dict
        # When padronize all layers,
        # add the old's parameters here
        self.mapping = {
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias"
        }

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())
        n_filters_1 = 16
        self.add_module(
            "conv_1",
            nn.Conv2d(self.n_chans, n_filters_1, (1, 1), stride=1, bias=True),
        )
        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(n_filters_1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_1", Expression(elu))
        # transpose to examples x 1 x (virtual, not EEG) channels x time
        self.add_module(
            "permute_1", Expression(lambda x: x.permute(0, 3, 1, 2))
        )

        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        n_filters_2 = 4
        # keras padds unequal padding more in front, so padding
        # too large should be ok.
        # Not padding in time so that cropped training makes sense
        # https://stackoverflow.com/questions/43994604/padding-with-even-kernel-size-in-a-convolutional-layer-in-keras-theano

        self.add_module(
            "conv_2",
            nn.Conv2d(
                1,
                n_filters_2,
                self.second_kernel_size,
                stride=1,
                padding=(self.second_kernel_size[0] // 2, 0),
                bias=True,
            ),
        )
        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(n_filters_2, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(2, 4), stride=(2, 4)))
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        n_filters_3 = 4
        self.add_module(
            "conv_3",
            nn.Conv2d(
                n_filters_2,
                n_filters_3,
                self.third_kernel_size,
                stride=1,
                padding=(self.third_kernel_size[0] // 2, 0),
                bias=True,
            ),
        )
        self.add_module(
            "bnorm_3",
            nn.BatchNorm2d(n_filters_3, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_3", Expression(elu))
        self.add_module("pool_3", pool_class(kernel_size=(2, 4), stride=(2, 4)))
        self.add_module("drop_3", nn.Dropout(p=self.drop_prob))

        output_shape = self.get_output_shape()
        n_out_virtual_chans = output_shape[2]

        if self.final_conv_length == "auto":
            n_out_time = output_shape[3]
            self.final_conv_length = n_out_time

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module("conv_classifier",
                          nn.Conv2d(n_filters_3, self.n_outputs,
                                    (n_out_virtual_chans, self.final_conv_length), bias=True, ))

        if self.add_log_softmax:
            module.add_module("softmax", nn.LogSoftmax(dim=1))
        # Transpose back to the logic of braindecode,

        # so time in third dimension (axis=2)
        module.add_module("permute_2", Rearrange("batch x y z -> batch x z y"), )

        module.add_module("squeeze", Expression(squeeze_final_output))

        self.add_module("final_layer", module)

        _glorot_weight_zero_bias(self)


def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if "BatchNorm" not in module.__class__.__name__:
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

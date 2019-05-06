import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from braindecode.models.base import BaseModel
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = th.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNetv4(BaseModel):
    """
    EEGNet v4 model from [EEGNet4]_.

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
        in_chans,
        n_classes,
        final_conv_length="auto",
        input_time_length=None,
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,  # usually set to F1*D (?)
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):

        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        model = nn.Sequential()
        # b c 0 1
        # now to b 1 0 c
        model.add_module("dimshuffle", Expression(_transpose_to_b_1_c_0))

        model.add_module(
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
        model.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )
        model.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
        )

        model.add_module(
            "bnorm_1",
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
        )
        model.add_module("elu_1", Expression(elu))

        model.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        model.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        model.add_module(
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
        model.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False, padding=(0, 0)
            ),
        )

        model.add_module(
            "bnorm_2", nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        )
        model.add_module("elu_2", Expression(elu))
        model.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        model.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        out = model(
            np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32)
            )
        )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        model.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.F2,
                self.n_classes,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )
        model.add_module("softmax", nn.LogSoftmax())
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        model.add_module("permute_back", Expression(_transpose_1_0))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        glorot_weight_zero_bias(model)
        return model


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class EEGNet(object):
    """
    EEGNet model from [EEGNet]_.

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
        in_chans,
        n_classes,
        final_conv_length="auto",
        input_time_length=None,
        pool_mode="max",
        second_kernel_size=(2, 32),
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):

        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        model = nn.Sequential()
        n_filters_1 = 16
        model.add_module(
            "conv_1", nn.Conv2d(self.in_chans, n_filters_1, (1, 1), stride=1, bias=True)
        )
        model.add_module(
            "bnorm_1", nn.BatchNorm2d(n_filters_1, momentum=0.01, affine=True, eps=1e-3)
        )
        model.add_module("elu_1", Expression(elu))
        # transpose to examples x 1 x (virtual, not EEG) channels x time
        model.add_module("permute_1", Expression(lambda x: x.permute(0, 3, 1, 2)))

        model.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        n_filters_2 = 4
        # keras padds unequal padding more in front, so padding
        # too large should be ok.
        # Not padding in time so that croped training makes sense
        # https://stackoverflow.com/questions/43994604/padding-with-even-kernel-size-in-a-convolutional-layer-in-keras-theano

        model.add_module(
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
        model.add_module(
            "bnorm_2", nn.BatchNorm2d(n_filters_2, momentum=0.01, affine=True, eps=1e-3)
        )
        model.add_module("elu_2", Expression(elu))
        model.add_module("pool_2", pool_class(kernel_size=(2, 4), stride=(2, 4)))
        model.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        n_filters_3 = 4
        model.add_module(
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
        model.add_module(
            "bnorm_3", nn.BatchNorm2d(n_filters_3, momentum=0.01, affine=True, eps=1e-3)
        )
        model.add_module("elu_3", Expression(elu))
        model.add_module("pool_3", pool_class(kernel_size=(2, 4), stride=(2, 4)))
        model.add_module("drop_3", nn.Dropout(p=self.drop_prob))

        out = model(
            np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32)
            )
        )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        model.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_3,
                self.n_classes,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )
        model.add_module("softmax", nn.LogSoftmax())
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        model.add_module("permute_2", Expression(lambda x: x.permute(0, 1, 3, 2)))
        model.add_module("squeeze", Expression(_squeeze_final_output))
        glorot_weight_zero_bias(model)
        return model

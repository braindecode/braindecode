from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import register_parametrization

from braindecode.util import np_to_th

from .parametrization import MaxNormParametrize


class AvgPool2dWithConv(nn.Module):
    """
    Compute average pooling using a convolution, to have the dilation parameter.

    Parameters
    ----------
    kernel_size: (int,int)
        Size of the pooling region.
    stride: (int,int)
        Stride of the pooling operation.
    dilation: int or (int,int)
        Dilation applied to the pooling filter.
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (
            in_channels,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape))
            or (self._pool_weights.is_cuda != x.is_cuda)
            or (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(np.ones(weight_shape, dtype=np.float32) / float(n_pool))
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        # initialize the weights
        nn.init.xavier_uniform_(self.weight, gain=1)
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))


class CombinedConv(nn.Module):
    """Merged convolutional layer for temporal and spatial convs in Deep4/ShallowFBCSP

    Numerically equivalent to the separate sequential approach, but this should be faster.

    Parameters
    ----------
    in_chans : int
        Number of EEG input channels.
    n_filters_time: int
        Number of temporal filters.
    filter_time_length: int
        Length of the temporal filter.
    n_filters_spat: int
        Number of spatial filters.
    bias_time: bool
        Whether to use bias in the temporal conv
    bias_spat: bool
        Whether to use bias in the spatial conv

    """

    def __init__(
        self,
        in_chans,
        n_filters_time=40,
        n_filters_spat=40,
        filter_time_length=25,
        bias_time=True,
        bias_spat=True,
    ):
        super().__init__()
        self.bias_time = bias_time
        self.bias_spat = bias_spat
        self.conv_time = nn.Conv2d(
            1, n_filters_time, (filter_time_length, 1), bias=bias_time, stride=1
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat, (1, in_chans), bias=bias_spat, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Merge time and spat weights
        combined_weight = (
            (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3))
            .sum(0)
            .unsqueeze(1)
        )

        bias = None
        calculated_bias: Optional[torch.Tensor] = None

        # Calculate bias terms
        if self.bias_time:
            time_bias = self.conv_time.bias
            if time_bias is None:
                raise RuntimeError("conv_time.bias is None despite bias_time=True")
            calculated_bias = (
                self.conv_spat.weight.squeeze()
                .sum(-1)
                .mm(time_bias.unsqueeze(-1))
                .squeeze()
            )
        if self.bias_spat:
            spat_bias = self.conv_spat.bias
            if spat_bias is None:
                raise RuntimeError("conv_spat.bias is None despite bias_spat=True")
            if calculated_bias is None:
                calculated_bias = spat_bias
            else:
                calculated_bias = calculated_bias + spat_bias

        bias = calculated_bias

        return F.conv2d(x, weight=combined_weight, bias=bias, stride=(1, 1))


class CausalConv1d(nn.Conv1d):
    """Causal 1-dimensional convolution

    Code modified from [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels (number of filters).
    kernel_size : int
        Kernel size.
    dilation : int, optional
        Dilation (number of elements to skip within kernel multiplication).
        Default to 1.
    **kwargs :
        Other keyword arguments to pass to torch.nn.Conv1d, except for
        `padding`!!

    References
    ----------
    .. [1] https://discuss.pytorch.org/t/causal-convolution/3456/4
    .. [2] https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        **kwargs,
    ):
        if "padding" in kwargs:
            raise ValueError(
                "The padding parameter is controlled internally by "
                f"{type(self).__name__} class. You should not try to override this"
                " parameter."
            )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs,
        )

    def forward(self, X):
        out = F.conv1d(
            X,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out[..., : -self.padding[0]]


class DepthwiseConv2d(torch.nn.Conv2d):
    """
    Depthwise convolution layer.

    This class implements a depthwise convolution, where each input channel is
    convolved separately with its own filter (channel multiplier), effectively
    performing a spatial convolution independently over each channel.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    depth_multiplier : int, optional
        Multiplier for the number of output channels. The total number of
        output channels will be `in_channels * depth_multiplier`. Default is 2.
    kernel_size : int or tuple, optional
        Size of the convolutional kernel. Default is 3.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple, optional
        Padding added to both sides of the input. Default is 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Padding mode to use. Options are 'zeros', 'reflect', 'replicate', or
        'circular'.
        Default is 'zeros'.
    """

    def __init__(
        self,
        in_channels,
        depth_multiplier=2,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

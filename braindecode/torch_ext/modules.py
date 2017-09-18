import numpy as np
import torch
import torch.nn.functional as F

from braindecode.torch_ext.util import np_to_var


class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.
    
    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if (hasattr(self.expression_fn, 'func') and
                  hasattr(self.expression_fn, 'kwargs')):
                expression_str = "{:s} {:s}".format(
                    self.expression_fn.func.__name__,
                    str(self.expression_fn.kwargs))
        else:
            expression_str = self.expression_fn.__name__
        return (self.__class__.__name__ + '(' +
                'expression=' + str(expression_str) + ')')


class AvgPool2dWithConv(torch.nn.Module):
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
    """
    def __init__(self, kernel_size, stride, dilation=1):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (in_channels, 1,
                        self.kernel_size[0], self.kernel_size[1])
        if self.weights is None or (
                (tuple(self.weights.size()) != tuple(weight_shape)) or (
                  self.weights.is_cuda != x.is_cuda
                ) or (
                    self.weights.data.type() != x.data.type()
                )):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_var(
                np.ones(weight_shape, dtype=np.float32) / float(n_pool))
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self.weights = weights

        pooled = F.conv2d(x, self.weights, bias=None, stride=self.stride,
                          dilation=self.dilation,
                          groups=in_channels,)
        return pooled

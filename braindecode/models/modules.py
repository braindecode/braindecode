import numpy as np

import torch
import torch.nn.functional as F

from ..util import np_to_var


class Ensure4d(torch.nn.Module):
    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        while(len(x.shape) < 4):
            x = x.unsqueeze(-1)
        return x


class Expression(torch.nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        """
        Initialize the given expression.

        Args:
            self: (todo): write your description
            expression_fn: (str): write your description
        """
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        """
        Evaluate the given function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.expression_fn(*x)

    def __repr__(self):
        """
        Return a repr representation of this expression.

        Args:
            self: (todo): write your description
        """
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )


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
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        """
        Initialize kernel.

        Args:
            self: (todo): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            dilation: (todo): write your description
            padding: (str): write your description
        """
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        """
        Forward computation. forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
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
            (tuple(self._pool_weights.size()) != tuple(weight_shape)) or
            (self._pool_weights.is_cuda != x.is_cuda) or
            (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_var(
                np.ones(weight_shape, dtype=np.float32) / float(n_pool)
            )
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


class IntermediateOutputWrapper(torch.nn.Module):
    """Wraps network model such that outputs of intermediate layers can be returned.
    forward() returns list of intermediate activations in a network during forward pass.

    Parameters
    ----------
    to_select : list
        list of module names for which activation should be returned
    model : model object
        network model

    Examples
    --------
    >>> model = Deep4Net()
    >>> select_modules = ['conv_spat','conv_2','conv_3','conv_4'] # Specify intermediate outputs
    >>> model_pert = IntermediateOutputWrapper(select_modules,model) # Wrap model
    """

    def __init__(self, to_select, model):
        """
        Add a module to_dict

        Args:
            self: (todo): write your description
            to_select: (bool): write your description
            model: (todo): write your description
        """
        if not len(list(model.children())) == len(list(model.named_children())):
            raise Exception("All modules in model need to have names!")

        super(IntermediateOutputWrapper, self).__init__()

        modules_list = model.named_children()
        for key, module in modules_list:
            self.add_module(key, module)
            self._modules[key].load_state_dict(module.state_dict())
        self._to_select = to_select

    def forward(self, x):
        """
        Forward forward forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        # Call modules individually and append activation to output if module is in to_select
        o = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                o.append(x)
        return o

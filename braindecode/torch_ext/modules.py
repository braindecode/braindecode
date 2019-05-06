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
        return self.__class__.__name__ + "(" + "expression=" + str(expression_str) + ")"


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
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (in_channels, 1, self.kernel_size[0], self.kernel_size[1])
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape))
            or (self._pool_weights.is_cuda != x.is_cuda)
            or (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_var(np.ones(weight_shape, dtype=np.float32) / float(n_pool))
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
        if not len(list(model.children())) == len(list(model.named_children())):
            raise Exception("All modules in model need to have names!")

        super(IntermediateOutputWrapper, self).__init__()

        modules_list = model.named_children()
        for key, module in modules_list:
            self.add_module(key, module)
            self._modules[key].load_state_dict(module.state_dict())
        self._to_select = to_select

    def forward(self, x):
        # Call modules individually and append activation to output if module is in to_select
        o = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                o.append(x)
        return o

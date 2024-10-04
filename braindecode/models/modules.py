# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor


from .functions import (
    drop_path,
    safe_log,
)
from braindecode.util import np_to_th


class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return "chomp_size={}".format(self.chomp_size)

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
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
        return self.__class__.__name__ + "(expression=%s) " % expression_str


class SafeLog(nn.Module):
    """
    Safe logarithm activation function module.

    :math:\text{SafeLog}(x) = \log\left(\max(x, \epsilon)\right)

    Parameters
    ----------
    eps : float, optional
        A small value to clamp the input tensor to prevent computing log(0) or log of negative numbers.
        Default is 1e-6.

    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x) -> Tensor:
        """
        Forward pass of the SafeLog module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying safe logarithm.
        """
        return safe_log(x=x, eps=self.eps)

    def extra_repr(self) -> str:
        eps_str = f"eps={self.eps}"
        return eps_str


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


class IntermediateOutputWrapper(nn.Module):
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


class TimeDistributed(nn.Module):
    """Apply module on multiple windows.

    Apply the provided module on a sequence of windows and return their
    concatenation.
    Useful with sequence-to-prediction models (e.g. sleep stager which must map
    a sequence of consecutive windows to the label of the middle window in the
    sequence).

    Parameters
    ----------
    module : nn.Module
        Module to be applied to the input windows. Must accept an input of
        shape (batch_size, n_channels, n_times).
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Sequence of windows, of shape (batch_size, seq_len, n_channels,
            n_times).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, seq_len, output_size).
        """
        b, s, c, t = x.shape
        out = self.module(x.view(b * s, c, t))
        return out.view(b, s, -1)


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
        assert "padding" not in kwargs, (
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
        out = super().forward(X)
        return out[..., : -self.padding[0]]


class MaxNormLinear(nn.Linear):
    """Linear layer with MaxNorm constraining on weights.

    Equivalent of Keras tf.keras.Dense(..., kernel_constraint=max_norm())
    [1, 2]_. Implemented as advised in [3]_.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.

    References
    ----------
    .. [1] https://keras.io/api/layers/core_layers/dense/#dense-class
    .. [2] https://www.tensorflow.org/api_docs/python/tf/keras/constraints/
           MaxNorm
    .. [3] https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-
           max-norm-constraint/96769
    """

    def __init__(
        self, in_features, out_features, bias=True, max_norm_val=2, eps=1e-5, **kwargs
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        self._max_norm_val = max_norm_val
        self._eps = eps

    def forward(self, X):
        self._max_norm()
        return super().forward(X)

    def _max_norm(self):
        with torch.no_grad():
            norm = self.weight.norm(2, dim=0, keepdim=True).clamp(
                min=self._max_norm_val / 2
            )
            desired = torch.clamp(norm, max=self._max_norm_val)
            self.weight *= desired / (self._eps + norm)


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

    def forward(self, x):
        # Merge time and spat weights
        combined_weight = (
            (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3))
            .sum(0)
            .unsqueeze(1)
        )

        # Calculate bias term
        if not self.bias_spat and not self.bias_time:
            bias = None
        else:
            bias = 0
            if self.bias_time:
                bias += (
                    self.conv_spat.weight.squeeze()
                    .sum(-1)
                    .mm(self.conv_time.bias.unsqueeze(-1))
                    .squeeze()
                )
            if self.bias_spat:
                bias += self.conv_spat.bias

        return F.conv2d(x, weight=combined_weight, bias=bias, stride=(1, 1))


class MLP(nn.Sequential):
    """Multilayer Perceptron (MLP) with GELU activation and optional dropout.

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric functions

    .. math:: h_{i + 1} = a_{i + 1}(h_i W_{i + 1}^T + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and output feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an
    MLP are its weights and biases :math:`\phi = \{W_i, b_i | i = 1, \dots, L\}`.

    Parameters:
    -----------
    in_features: int
        Number of input features.
    hidden_features: Sequential[int] (default=None)
        Number of hidden features, if None, set to in_features.
        You can increase the size of MLP just passing more int in the
        hidden features vector. The model size increase follow the
        rule 2n (hidden layers)+2 (in and out layers)
    out_features: int (default=None)
        Number of output features, if None, set to in_features.
    act_layer: nn.GELU (default)
        The activation function constructor. If :py:`None`, use
        :class:`torch.nn.GELU` instead.
    drop: float (default=0.0)
        Dropout rate.
    normalize: bool (default=False)
        Whether to apply layer normalization.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        activation=nn.GELU,
        drop=0.0,
        normalize=False,
    ):
        self.normalization = nn.LayerNorm if normalize else lambda: None
        self.in_features = in_features
        self.out_features = out_features or self.in_features
        if hidden_features:
            self.hidden_features = hidden_features
        else:
            self.hidden_features = (self.in_features, self.in_features)
        self.activation = activation

        layers = []

        for before, after in zip(
            (self.in_features, *self.hidden_features),
            (*self.hidden_features, self.out_features),
        ):
            layers.extend(
                [
                    nn.Linear(in_features=before, out_features=after),
                    self.activation(),
                    self.normalization(),
                ]
            )

        layers = layers[:-2]
        layers.append(nn.Dropout(p=drop))

        # Cleaning if we are not using the normalization layer
        layers = list(filter(lambda layer: layer is not None, layers))

        super().__init__(*layers)


class DropPath(nn.Module):
    """Drop paths, also known as Stochastic Depth, per sample.

        When applied in main path of residual blocks.

        Parameters:
        -----------
        drop_prob: float (default=None)
            Drop path probability (should be in range 0-1).

        Notes
        -----
        Code copied and modified from VISSL facebookresearch:
    https://github.com/facebookresearch/vissl/blob/0b5d6a94437bc00baed112ca90c9d78c6ccfbafb/vissl/models/model_helpers.py#L676
        All rights reserved.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    # Utility function to print DropPath module
    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"

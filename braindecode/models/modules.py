# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .functions import drop_path
from ..util import np_to_th


class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


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

        return F.conv2d(
            x, weight=combined_weight, bias=bias, stride=(1, 1)
        )


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with GELU activation and optional dropout.

    Parameters:
    -----------
    in_features: int
        Number of input features.
    hidden_features: int (default=None)
        Number of hidden features, if None, set to in_features.
    out_features: int (default=None)
        Number of output features, if None, set to in_features.
    act_layer: nn.GELU (default)
        Activation function.
    drop: float (default=0.0)
        Dropout rate.
    """

    def __init__(
            self,
            in_features: int,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalConv(nn.Module):
    """
    Temporal Convolutional Module inspired by Visual Transformer.

    In this module we apply the follow steps three times repeatedly
    to the input tensor, reducing the temporal dimension only in the first.
    - Apply a 2D convolution.
    - Apply a GELU activation function.
    - Apply a GroupNorm with 4 groups.

    Parameters:
    -----------
    in_chans: int (default=1)
        Number of input channels.
    out_chans: int (default=8)
        Number of output channels.
    num_groups: int (default=4)
        Number of groups for GroupNorm.
    kernel_size_1: tuple (default=(1, 15))
        Kernel size for the first convolution.
    kernel_size_2: tuple (default=(1, 3))
        Kernel size for the second and third convolutions.
    stride_1: tuple (default=(1, 8))
        Stride for the first convolution.
    padding_1: tuple (default=(0, 7))
        Padding for the first convolution.
    padding_2: tuple (default=(0, 1))
        Padding for the second and third convolutions.
    Returns:
    --------
    x: torch.Tensor
        Output tensor of shape (Batch, NA, Temporal Channel).
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=8,
            num_groups=4,
            kernel_size_1=(1, 15),
            stride_1=(1, 8),
            padding_1=(0, 7),
            kernel_size_2=(1, 3),
            padding_2=(0, 1),
            act_layer=nn.GELU,
    ):
        super().__init__()

        # Here, we use the Rearrange layer from einops to flatten the input
        # tensor to a 2D tensor, so we can apply 2D convolutions.
        self.channel_patch_flatten = Rearrange(
            "Batch chs npat spatch -> Batch () (chs npat) spatch"
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1,
        )
        self.act_layer_1 = act_layer()
        self.norm1 = nn.GroupNorm(num_groups=num_groups,
                                  num_channels=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2,
            padding=padding_2,
        )
        self.act_layer_2 = act_layer()
        self.norm2 = nn.GroupNorm(num_groups=num_groups,
                                  num_channels=out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2,
            padding=padding_2,
        )
        self.norm3 = nn.GroupNorm(num_groups=num_groups,
                                  num_channels=out_channels)
        self.act_layer_3 = act_layer()

        self.transpose_temporal_channel = Rearrange(
            "Batch C NA T -> Batch NA (T C)")

    def forward(self, x):
        """
        Apply 3 steps of 2D convolution, GELU activation function,
        and GroupNorm.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, Channels, n_patchs, size_patch).

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (Batch, NA, Temporal Channel).
        """
        x = self.channel_patch_flatten(x)
        x = self.act_layer_1(self.norm1(self.conv1(x)))
        x = self.act_layer_2(self.norm2(self.conv2(x)))
        x = self.act_layer_3(self.norm3(self.conv3(x)))
        x = self.transpose_temporal_channel(x)
        return x


class PatchEmbed(nn.Module):
    """EEG to Patch Embedding.

    This code is used when we want to apply the patch embedding
    after the codebook layer.

    Parameters:
    -----------
    n_times: int (default=2000)
        Number of temporal components of the input tensor.
    patch_size: int (default=200)
        Size of the patch, default is 1-seconds with 200Hz.
    in_channels: int (default=1)
        Number of input channels for to be used in the convolution.
    embed_dim: int (default=200)
        Number of out_channes to be used in the convolution, here,
        we used the same as patch_size.
    n_codebooks: int (default=62)
        Number of patches to be used in the convolution, here,
        we used the same as n_times // patch_size.
    """

    def __init__(self,
                 n_times=2000,
                 patch_size=200,
                 in_channels=1,
                 embed_dim=200,
                 n_codebooks=62):
        super().__init__()
        num_patches = n_codebooks * (n_times // patch_size)
        self.patch_shape = (1, n_times // patch_size)
        self.EEG_size = n_times
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
        )

        self.merge_transpose = Rearrange(
            "Batch ch (patch spatch) -> Batch (patch spatch) ch")

    def forward(self, x):
        """
        Apply the convolution to the input tensor.
        then merge the output tensor to the desired shape.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, Channels, n_patchs, patch_size).

        Return:
        -------
        x: torch.Tensor
            Output tensor of shape (Batch, n_patchs, patch_size, channels).
        """
        x = self.proj(x)
        x = self.merge_transpose(x)
        return x


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


class Attention(nn.Module):
    """
    Attention with the options of Window-based multi-head self attention (W-MSA).

    This code is strong inspired by:
    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L77

    Basically, the attention module is a linear layer that takes the input
    tensor and returns the output tensor. The input tensor is first passed
    through a linear layer to get the query, key, and value tensors. Then,
    the query tensor is multiplied by the scale factor and the result is
    multiplied by the transpose of the key tensor.

    The flag window_size is used to determine if the attention is
    window-based or not.

    Parameters:
    -----------
    dim: int
        Number of input features.
    num_heads: int (default=8)
        Number of attention heads.
    qkv_bias: bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm: nn.LayerNorm (default=None)
        If not None, apply LayerNorm to the query and key tensors.
    qk_scale: float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    attn_drop: float (default=0.0)
        Dropout rate for the attention weights.
    proj_drop: float (default=0.0)
        Dropout rate for the output tensor.
    window_size: bool (default=None)
        If not None, use window-based multi-head self attention based on Swin Transformer.
    attn_head_dim: int (default=None)
        If not None, use this value as the head_dim. If None, use dim // num_heads.
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=None,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            window_size=None,
            attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                    2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(
                torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                    coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[
                                            0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(
                -1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                                 relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            rel_pos_bias=None,
            return_attention=False,
            return_qkv=False,
    ):
        """
        Apply the attention mechanism to the input tensor.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, N, C).
        rel_pos_bias: torch.Tensor (default=None)
            If not None, add this tensor to the attention weights.
        return_attention: bool (default=False)
            If True, return the attention weights.
        return_qkv: bool (default=False)
            If True, return the query, key, and value tensors together with
            the output tensor.
        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (Batch, N, C).
        qkv: torch.Tensor (optional)
            Query, key, and value tensors of shape
            (Batch, N, 3, num_heads, C // num_heads).
        """
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = nn.functional.linear(input=x, weight=self.qkv.weight,
                                   bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class WindowsAttentionBlock(nn.Module):
    """Attention Block from Vision Transformer with support for
    Window-based Attention.

    Notes: This code is strong inspired by:
    BeiTv2 from Microsoft.

    Parameters:
    -----------
    dim: int
        Number of input features.
    num_heads: int (default=8)
        Number of attention heads.
    mlp_ratio: float (default=4.0)
        Ratio to increase the hidden features from input features in the MLP layer
    qkv_bias: bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm: nn.LayerNorm (default=None)
        If not None, apply LayerNorm to the query and key tensors.
    qk_scale: float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    drop: float (default=0.0)
        Dropout rate for the output tensor.
    attn_drop: float (default=0.0)
        Dropout rate for the attention weights.
    drop_path: float (default=0.0)
        Dropout rate for the output tensor.
    init_values: float (default=None)
        If not None, use this value to initialize the gamma_1 and gamma_2
        parameters.
    act_layer: nn.GELU (default)
        Activation function.
    norm_layer: nn.LayerNorm (default)
        Normalization layer.
    window_size: bool (default=None)
        If not None, use window-based multi-head self attention based on
        Swin Transformer.
    attn_head_dim: int (default=None)
        If not None, use this value as the head_dim. If None,
        the classes use dim // num_heads

    Returns:
    --------
    x: torch.Tensor
        Output tensor of shape (Batch, N, C). [I think]

    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_norm=None,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            init_values=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            window_size=None,
            attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x,
                rel_pos_bias=None,
                return_attention=False,
                return_qkv=False):
        if return_attention:
            return self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True
            )
        if return_qkv:
            y, qkv = self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv
            )
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x),
                                         rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class SegmentPatch(nn.Module):
    """Segment and Patch for EEG data.

    Adapted Patch Embedding inspired in the Visual Transform approach
    to extract the learned segmentor, we expect get the input shape as:
    (Batch, Number of Channels, number of times points).

    We apply a 2D convolution with kernel size of (1, patch_size)
    and a stride of (1, patch_size).

    The results output shape will be:
    (Batch, Number of Channels, Number of patches, patch size).

    This way, we learned a convolution to segment the input shape.

    The number of patches is calculated as the number of samples divided
    by the patch size.

    Parameters:
    -----------
    n_times: int (default=2000)
        Number of temporal components of the input tensor.
    in_chans: int (default=1)
        number of electrods from the EEG signal
    embed_dim: int (default=200)
        Number of n_output to be used in the convolution, here,
        we used the same as patch_size.
    patch_size: int (default=200)
        Size of the patch, default is 1-seconds with 200Hz.
    Returns:
    --------
    x_patched: torch.Tensor
        Output tensor of shape (batch, n_chans, num_patches, embed_dim).
    """

    def __init__(self, n_times=2000, patch_size=200, n_chans=1, embed_dim=200):
        super().__init__()

        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patchs = n_times // patch_size
        self.embed_dim = embed_dim
        self.n_chans = n_chans

        self.patcher = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )
        self.adding_extra_dim = Rearrange(
            pattern="batch nchans temporal -> (batch nchans) 1 temporal")

    def forward(self, x):
        """
        Using an 1D convolution to generate segments of EEG signal.

        Parameters:
        -----------
        X: Tensor
            [batch, n_chans, n_times]

        Returns:
        --------
        X_patch: Tensor
            [batch, n_chans, n_times//patch_size, patch_size]
        """
        batch_size, _, _ = x.shape
        # Input shape: [batch, n_chs, n_times]

        # First, rearrange input to treat the channel dimension 'n_chs' as
        # separate 'dimension' in batch for Conv1d
        # This requires reshaping x to have a height of 1 for each EEG sample.

        x = self.adding_extra_dim(x)

        # Apply the convolution along the temporal dimension
        # Conv2d output shape: [(batch*n_chs), embed_dim, n_patches]
        x = self.patcher(x)

        # Now, rearrange output to get back to a batch-first format,
        # combining embedded patches with channel information
        # Assuming you want [batch, n_chs, n_patches, embed_dim]
        # as output, which keeps channel information
        # This treats each patch embedding as a feature alongside channels
        x = rearrange(
            x,
            pattern="(batch nchans) embed npatchs -> "
                    "batch nchans npatchs embed",
            batch=batch_size,
            nchans=self.n_chans,
        )

        return x

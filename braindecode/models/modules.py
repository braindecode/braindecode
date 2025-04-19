# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

from typing import Callable
import numpy as np
import torch
from einops.layers.torch import Rearrange

from typing import Optional, List, Tuple
from torch.nn.utils.parametrize import register_parametrization
from functools import partial

from mne.filter import create_filter, _check_coefficients
from mne.utils import warn

from torch import Tensor, nn, from_numpy
import torch.nn.functional as F

from torchaudio.functional import fftconvolve, filtfilt


from braindecode.models.functions import (
    drop_path,
    safe_log,
)
from braindecode.util import np_to_th
from braindecode.models.eegminer import GeneralizedGaussianFilter
from braindecode.models.parametrization import MaxNorm, MaxNormParametriza


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
    r"""
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

    padding: Tensor

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
        register_parametrization(self, "weight", MaxNorm(self._max_norm_val, eps))


class LinearWithConstraint(nn.Linear):
    """Linear layer with max-norm constraint on the weights."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_norm = max_norm
        # initialize the weights
        nn.init.xavier_uniform_(self.weight, gain=1)
        nn.init.constant_(self.bias, 0)
        # register the parametrization
        register_parametrization(self, "weight", MaxNormParametriza(self._max_norm))


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with max-norm constraint on the weights."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_norm = max_norm
        # initialize the weights
        nn.init.xavier_uniform_(self.weight, gain=1)

        register_parametrization(self, "weight", MaxNormParametriza(self._max_norm))


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
            out_ch = combined_weight.size(0)
            bias = x.new_zeros(out_ch)  # same device/dtype as input

            # Pull attribute into a local for refinement
            bias_time_opt = self.conv_time.bias
            # Local None‑check refines Optional[Tensor] → Tensor
            if bias_time_opt is not None:
                bias_time = bias_time_opt  # now a real Tensor
                bias += (
                    self.conv_spat.weight.squeeze()
                    .sum(-1)
                    .mm(bias_time.unsqueeze(-1))
                    .squeeze()
                )
            bias_spat_opt = self.conv_spat.bias
            if bias_spat_opt is not None:
                bias += bias_spat_opt

        return F.conv2d(
            x,
            weight=combined_weight,
            bias=bias,
            stride=(1, 1),
        )


class MLP(nn.Sequential):
    r"""Multilayer Perceptron (MLP) with GELU activation and optional dropout.

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric functions

    .. math:: h_{i + 1} = a_{i + 1}(h_i W_{i + 1}^T + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and output feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an
    MLP are its weights and biases :math:`\\phi = \{W_i, b_i | i = 1, \dots, L\}`.

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


class FilterBankLayer(nn.Module):
    """Apply multiple band-pass filters to generate multiview signal representation.

    This layer constructs a bank of signals filtered in specific bands for each channel.
    It uses MNE's `create_filter` function to create the band-specific filters and
    applies them to multi-channel time-series data. Each filter in the bank corresponds to a
    specific frequency band and is applied to all channels of the input data. The filtering is
    performed using FFT-based convolution via the `fftconvolve` function from
    :func:`torchaudio.functional if the method is FIR, and `filtfilt` function from
    :func:`torchaudio.functional if the method is IIR.

    The default configuration creates 9 non-overlapping frequency bands with a 4 Hz bandwidth,
    spanning from 4 Hz to 40 Hz (i.e., 4-8 Hz, 8-12 Hz, ..., 36-40 Hz). This setup is based on the
    reference: *FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface*.

    Parameters
    ----------
    n_chans : int
        Number of channels in the input signal.
    sfreq : int
        Sampling frequency of the input signal in Hz.
    band_filters : Optional[List[Tuple[float, float]]] or int, default=None
        List of frequency bands as (low_freq, high_freq) tuples. Each tuple defines
        the frequency range for one filter in the bank. If not provided, defaults
        to 9 non-overlapping bands with 4 Hz bandwidths spanning from 4 to 40 Hz.
    method : str, default='fir'
        ``'fir'`` will use FIR filtering, ``'iir'`` will use IIR
        forward-backward filtering (via :func:`~scipy.signal.filtfilt`).
        For more details, please check the `MNE Preprocessing Tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html>`_.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

        * **'auto' (default)**: The filter length is chosen based
          on the size of the transition regions (6.6 times the reciprocal
          of the shortest transition band for fir_window='hamming'
          and fir_design="firwin2", and half that for "firwin").
        * **str**: A human-readable time in
          units of "s" or "ms" (e.g., "10s" or "5500ms") will be
          converted to that number of samples if ``phase="zero"``, or
          the shortest power-of-two length at least that duration for
          ``phase="zero-double"``.
        * **int**: Specified length in samples. For fir_design="firwin",
          this should not be used.
    l_trans_bandwidth : Union[str, float, int], default='auto'
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Can be "auto"
        (default) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : Union[str, float, int], default='auto'
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    phase : str, default='zero'
        Phase of the filter.
        When ``method='fir'``, symmetric linear-phase FIR filters are constructed
        with the following behaviors when ``method="fir"``:

        ``"zero"`` (default)
            The delay of this filter is compensated for, making it non-causal.
        ``"minimum"``
            A minimum-phase filter will be constructed by decomposing the zero-phase filter
            into a minimum-phase and all-pass systems, and then retaining only the
            minimum-phase system (of the same length as the original zero-phase filter)
            via :func:`scipy.signal.minimum_phase`.
        ``"zero-double"``
            *This is a legacy option for compatibility with MNE <= 0.13.*
            The filter is applied twice, once forward, and once backward
            (also making it non-causal).
        ``"minimum-half"``
            *This is a legacy option for compatibility with MNE <= 1.6.*
            A minimum-phase filter will be reconstructed from the zero-phase filter with
            half the length of the original filter.

        When ``method='iir'``, ``phase='zero'`` (default) or equivalently ``'zero-double'``
        constructs and applies IIR filter twice, once forward, and once backward (making it
        non-causal) using :func:`~scipy.signal.filtfilt`; ``phase='forward'`` will apply
        the filter once in the forward (causal) direction using
        :func:`~scipy.signal.lfilter`.

           The behavior for ``phase="minimum"`` was fixed to use a filter of the requested
           length and improved suppression.
    iir_params : Optional[dict], default=None
        Dictionary of parameters to use for IIR filtering.
        If ``iir_params=None`` and ``method="iir"``, 4th order Butterworth will be used.
        For more information, see :func:`mne.filter.construct_iir_filter`.
    fir_window : str, default='hamming'
        The window to use in FIR design, can be "hamming" (default),
        "hann" (default in 0.13), or "blackman".
    fir_design : str, default='firwin'
        Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
        or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
        a time-domain design technique that generally gives improved
        attenuation using fewer samples than "firwin2".
    pad : str, default='reflect_limited'
        The type of padding to use. Supports all func:`numpy.pad()` mode options.
        Can also be "reflect_limited", which pads with a reflected version of
        each vector mirrored on the first and last values of the vector,
        followed by zeros. Only used for ``method='fir'``.
    verbose: bool | str | int | None, default=True
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the func:`mne.verbose` for details.
        Should only be passed as a keyword argument.
    """

    def __init__(
        self,
        n_chans: int,
        sfreq: float,
        band_filters: Optional[List[Tuple[float, float]] | int] = None,
        method: str = "fir",
        filter_length: str | float | int = "auto",
        l_trans_bandwidth: str | float | int = "auto",
        h_trans_bandwidth: str | float | int = "auto",
        phase: str = "zero",
        iir_params: Optional[dict] = None,
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        verbose: bool = True,
    ):
        super(FilterBankLayer, self).__init__()

        # The first step here is to check the band_filters
        # We accept as None values.
        if band_filters is None:
            """
            the filter bank is constructed using 9 filters with non-overlapping
            frequency bands, each of 4Hz bandwidth, spanning from 4 to 40 Hz
            (4-8, 8-12, …, 36-40 Hz)

            Based on the reference: FBCNet: A Multi-view Convolutional Neural
            Network for Brain-Computer Interface
            """
            band_filters = [(low, low + 4) for low in range(4, 36 + 1, 4)]
        # We accept as int.
        if isinstance(band_filters, int):
            warn(
                "Creating the filter banks equally divided in the "
                "interval 4Hz to 40Hz with almost equal bandwidths. "
                "If you want a specific interval, "
                "please specify 'band_filters' as a list of tuples.",
                UserWarning,
            )
            start = 4
            end = 40

            total_band_width = end - start  # 4 Hz to 40 Hz

            band_width_calculated = total_band_width / band_filters
            band_filters = [
                (
                    torch.tensor(start + i * band_width_calculated),
                    torch.tensor(start + (i + 1) * band_width_calculated),
                )
                for i in range(band_filters)
            ]

        if not isinstance(band_filters, list):
            raise ValueError(
                "`band_filters` should be a list of tuples if you want to "
                "use them this way."
            )
        else:
            if any(len(bands) != 2 for bands in band_filters):
                raise ValueError(
                    "The band_filters items should be splitable in 2 values."
                )

        # and we accepted as
        self.band_filters = band_filters
        self.n_bands = len(band_filters)
        self.phase = phase
        self.method = method
        self.n_chans = n_chans
        self.sfreq = sfreq
        self.method_iir = True if self.method == "iir" else False

        if self.method_iir:
            if iir_params is None:
                iir_params = dict(output="ba")
            else:
                if "output" in iir_params:
                    if iir_params["output"] == "sos":
                        warn(
                            "It is not possible to use second-order section filtering with Torch. Changing to filter ba",
                            UserWarning,
                        )
                        iir_params["output"] = "ba"

        filts = {}
        for idx, (l_freq, h_freq) in enumerate(band_filters):
            filt = create_filter(
                data=None,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                filter_length=filter_length,
                l_trans_bandwidth=l_trans_bandwidth,
                h_trans_bandwidth=h_trans_bandwidth,
                method=method,
                iir_params=iir_params,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
                verbose=verbose,
            )
            if not self.method_iir:
                filt = from_numpy(filt).float()
                filts[f"band_{idx}"] = {"filt": filt}

            else:
                _check_coefficients((filt["b"], filt["a"]))
                b = torch.tensor(filt["b"], dtype=torch.float64)
                a = torch.tensor(filt["a"], dtype=torch.float64)

                filts[f"band_{idx}"] = {"b": b, "a": a}

        self.filts = nn.ParameterDict(filts)

        if self.method_iir:
            self._apply_filter_func = self._apply_iir
        else:
            self._apply_filter_func = partial(self._apply_fir, n_chans=self.n_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter bank to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, time_points).

        Returns
        -------
        torch.Tensor
            Filtered output tensor of shape (batch_size, n_bands, n_chans, filtered_time_points).
        """
        return torch.cat(
            [self._apply_filter_func(x, p_filt) for p_filt in self.filts.values()],
            dim=1,
        )

    @staticmethod
    def _apply_fir(x, filter: dict, n_chans: int) -> Tensor:
        """
        Apply an FIR filter to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, n_chans, n_times).
        filter : dict
            Dictionary containing IIR filter coefficients.
            - "b": Tensor of numerator coefficients.
        n_chans: int
            Number of channels

        Returns
        -------
        Tensor
            Filtered tensor of shape (batch_size, 1, n_chans, n_times).
        """
        # Expand filter coefficients to match the number of channels
        # Original 'b' shape: (filter_length,)
        # After unsqueeze and repeat: (n_chans, filter_length)
        # After final unsqueeze: (1, n_chans, filter_length)
        filt_expanded = (
            filter["filt"].to(x.device).unsqueeze(0).repeat(n_chans, 1).unsqueeze(0)
        )

        # Perform FFT-based convolution
        # Input x shape: (batch_size, n_chans, n_times)
        # filt_expanded shape: (1, n_chans, filter_length)
        # After convolution: (batch_size, n_chans, n_times)

        filtered = fftconvolve(
            x, filt_expanded, mode="same"
        )  # Shape: (batch_size, nchans, time_points)

        # Add a new dimension for the band
        # Shape after unsqueeze: (batch_size, 1, n_chans, n_times)
        filtered = filtered.unsqueeze(1)
        # returning the filtered
        return filtered

    @staticmethod
    def _apply_iir(x: Tensor, filter: dict) -> Tensor:
        """
        Apply an IIR filter to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, n_chans, n_times).
        filter : dict
            Dictionary containing IIR filter coefficients

            - "b": Tensor of numerator coefficients.
            - "a": Tensor of denominator coefficients.

        Returns
        -------
        Tensor
            Filtered tensor of shape (batch_size, 1, n_chans, n_times).
        """
        # Apply filtering using torchaudio's filtfilt
        filtered = filtfilt(
            x,
            a_coeffs=filter["a"].type_as(x).to(x.device),
            b_coeffs=filter["b"].type_as(x).to(x.device),
            clamp=False,
        )
        # Rearrange dimensions to (batch_size, 1, n_chans, n_times)
        return filtered.unsqueeze(1)


class LogActivation(nn.Module):
    """Logarithm activation function."""

    def __init__(self, epsilon: float = 1e-6, *args, **kwargs):
        """
        Parameters
        ----------
        epsilon : float
            Small float to adjust the activation.
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.epsilon)  # Adding epsilon to prevent log(0)


class SqueezeFinalOutput(nn.Module):
    """

    Removes empty dimension at end and potentially removes empty time
    dimension. It does  not just use squeeze as we never want to remove
    first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    def __init__(self):
        super().__init__()

        self.squeeze = Rearrange("b c t 1 -> b c t")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) drop feature dim
        x = self.squeeze(x)
        # 2) drop time dim if singleton
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


class Square(nn.Module):
    """Element-wise square."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class StatLayer(nn.Module):
    """
    Generic layer to compute a statistical function along a specified dimension.
    Parameters
    ----------
    stat_fn : Callable
        A function like torch.mean, torch.std, etc.
    dim : int
        Dimension along which to apply the function.
    keepdim : bool, default=True
        Whether to keep the reduced dimension.
    clamp_range : tuple(float, float), optional
        Used only for functions requiring clamping (e.g., log variance).
    apply_log : bool, default=False
        Whether to apply log after computation (used for LogVarLayer).
    """

    def __init__(
        self,
        stat_fn: Callable[..., torch.Tensor],
        dim: int,
        keepdim: bool = True,
        clamp_range: Optional[Tuple[float, float]] = None,
        apply_log: bool = False,
    ) -> None:
        super().__init__()
        self.stat_fn = stat_fn
        self.dim = dim
        self.keepdim = keepdim
        self.clamp_range = clamp_range
        self.apply_log = apply_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stat_fn(x, dim=self.dim, keepdim=self.keepdim)
        if self.clamp_range is not None:
            out = torch.clamp(out, min=self.clamp_range[0], max=self.clamp_range[1])
        if self.apply_log:
            out = torch.log(out)
        return out


# make things more simple
def _max_fn(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    return x.max(dim=dim, keepdim=keepdim)[0]


def _power_fn(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    # compute mean of squared values along `dim`
    return torch.mean(x**2, dim=dim, keepdim=keepdim)


MeanLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, torch.mean)
MaxLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, _max_fn)
VarLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, torch.var)
StdLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, torch.std)
LogVarLayer: Callable[[int, bool], StatLayer] = partial(
    StatLayer,
    torch.var,
    clamp_range=(1e-6, 1e6),
    apply_log=True,
)

LogPowerLayer: Callable[[int, bool], StatLayer] = partial(
    StatLayer,
    _power_fn,
    clamp_range=(1e-4, 1e4),
    apply_log=True,
)

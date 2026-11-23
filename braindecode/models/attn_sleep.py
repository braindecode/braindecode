# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)
#
# This implementation derives from https://github.com/emadeldeen24/AttnSleep:
#
# MIT License
#
# Copyright (c) 2020 Emadeldeen Eldele
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from copy import deepcopy
from numbers import Integral

import torch
from torch import nn

from braindecode.functional import scaled_dot_product_attention
from braindecode.models.base import EEGModuleMixin, deprecated_init_flags
from braindecode.modules import CausalConv1d


@deprecated_init_flags(
    identity_head=("return_feats",),
    return_feats="features",
)
class AttnSleep(EEGModuleMixin, nn.Module):
    r"""Sleep Staging Architecture from Eldele et al  (2021) [Eldele2021]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. figure:: https://raw.githubusercontent.com/emadeldeen24/AttnSleep/refs/heads/main/imgs/AttnSleep.png
        :align: center
        :alt: AttnSleep Architecture

    Attention based Neural Net for sleep staging as described in [Eldele2021]_.
    The code for the paper and this model is also available at [1]_.
    Takes single channel EEG as input.
    Feature extraction module based on multi-resolution convolutional neural network (MRCNN)
    and adaptive feature recalibration (AFR).
    The second module is the temporal context encoder (TCE) that leverages a multi-head attention
    mechanism to capture the temporal dependencies among the extracted features.

    Warning - This model was designed for signals of 30 seconds at 100Hz or 125Hz (in which case
    the reference architecture from [1]_ which was validated on SHHS dataset [2]_ will be used)
    to use any other input is likely to make the model perform in unintended ways. Any other
    window length also needs a ``d_model`` of its own, see the parameter below.

    Parameters
    ----------
    n_tce : int
        Number of TCE clones.
    d_model : int
        Input dimension for the TCE.
        Also the input dimension of the first FC layer in the feed forward
        and the output of the second FC layer in the same.
        Increase for higher sampling rate/signal length.
        It should be divisible by n_attn_heads. It must also equal the number
        of time steps returned by the feature extractor: 80 for 30 seconds at
        100 Hz and 100 for 30 seconds at 125 Hz. A construction error reports
        the value needed for other window lengths.
    d_ff : int
        Output dimension of the first FC layer in the feed forward and the
        input dimension of the second FC layer in the same.
    n_attn_heads : int
        Number of attention heads. It should be a factor of d_model
    drop_prob : float
        Dropout rate in the PositionWiseFeedforward layer and the TCE layers.
    after_reduced_cnn_size : int
        Number of output channels produced by the convolution in the AFR module.
    return_feats : bool
        Deprecated, removed in 2.3: call ``model(x, return_features=True)`` instead.
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    activation : nn.Module, default=nn.ReLU
        Activation function class to apply in the AFR block and the TCE
        feed-forward block. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    activation_mrcnn : nn.Module, default=nn.GELU
        Activation function class to apply in the multi-resolution CNN layer.
        Should be a PyTorch activation module class like ``nn.ReLU`` or
        ``nn.GELU``. Default is ``nn.GELU``.
    attn_query_norm : bool
        If ``True``, apply the layer norm of the attention block to the query
        before the attention, as braindecode 1.0 to 2.0 did. If ``False``
        (default), attend with the raw query, as the authors' code does.

        .. versionadded:: 2.1

    References
    ----------
    .. [Eldele2021] E. Eldele et al., "An Attention-Based Deep Learning Approach for Sleep Stage
        Classification With Single-Channel EEG," in IEEE Transactions on Neural Systems and
        Rehabilitation Engineering, vol. 29, pp. 809-818, 2021, doi: 10.1109/TNSRE.2021.3076234.

    .. [1] https://github.com/emadeldeen24/AttnSleep

    .. [2] https://sleepdata.org/datasets/shhs
    """

    _time_axis = "fixed"  # fixed: feature_extractor.1.layers.0.self_attn.linear.weight is sized by n_times
    _channel_axis = (
        "fixed"  # fixed: feature_extractor.0.features1.0.weight is sized by n_chans
    )

    def __init__(
        self,
        sfreq=None,
        n_tce=2,
        d_model=80,
        d_ff=120,
        n_attn_heads=5,
        drop_prob=0.1,
        activation_mrcnn: type[nn.Module] = nn.GELU,
        activation: type[nn.Module] = nn.ReLU,
        input_window_seconds=None,
        n_outputs=None,
        after_reduced_cnn_size=30,
        return_feats=False,
        chs_info=None,
        n_chans=None,
        n_times=None,
        attn_query_norm=False,
    ):
        if (
            sum(value is not None for value in (n_times, sfreq, input_window_seconds))
            < 2
        ):
            raise ValueError(
                "AttnSleep requires at least two of n_times, sfreq, and "
                "input_window_seconds."
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
        if self._n_chans_for_jit not in (None, 1):
            raise ValueError(
                f"AttnSleep takes single-channel input; got n_chans={self._n_chans_for_jit}."
            )

        self.mapping = {
            "fc.weight": "final_layer.weight",
            "fc.bias": "final_layer.bias",
        }

        if not (
            (self.input_window_seconds == 30 and self.sfreq == 100 and d_model == 80)
            or (
                self.input_window_seconds == 30 and self.sfreq == 125 and d_model == 100
            )
        ):
            warnings.warn(
                "This model was designed originally for input windows of 30sec at 100Hz, "
                "with d_model at 80 or at 125Hz, with d_model at 100, to use anything "
                "other than this may cause errors or cause the model to perform in "
                "other ways than intended",
                UserWarning,
            )

        # Coarse-branch kernel size of the MRCNN (6 at 125 Hz, 7 otherwise).
        kernel_size = 6 if self.sfreq == 125 else 7

        feature_length = self._feature_length(self.n_times, kernel_size)
        if feature_length != d_model:
            raise ValueError(
                f"d_model is {d_model} but the feature extractor returns "
                f"{feature_length} time steps for an input of {self.n_times} "
                f"samples at {self.sfreq} Hz. Set d_model={feature_length}, with "
                "an n_attn_heads that divides it."
            )
        mrcnn = _MRCNN(
            after_reduced_cnn_size,
            kernel_size,
            activation=activation_mrcnn,
            activation_se=activation,
        )
        # braindecode 2.0 also saved the AFR layers as ``features.<index>``.
        afr = "feature_extractor.0.AFR.0."
        for index, name in enumerate(_AFR_LAYERS):
            for key in mrcnn.AFR[0].get_submodule(name).state_dict():
                self.mapping[f"{afr}features.{index}.{key}"] = f"{afr}{name}.{key}"
        attn = _MultiHeadedAttention(n_attn_heads, d_model, after_reduced_cnn_size)
        ff = _PositionwiseFeedForward(d_model, d_ff, drop_prob, activation=activation)
        tce = _TCE(
            _EncoderLayer(
                d_model,
                deepcopy(attn),
                deepcopy(ff),
                after_reduced_cnn_size,
                drop_prob,
                attn_query_norm,
            ),
            n_tce,
        )

        self.feature_extractor = nn.Sequential(mrcnn, tce)
        self.len_last_layer = feature_length * after_reduced_cnn_size

        self.final_layer = nn.Linear(self.len_last_layer, self.n_outputs)

    @classmethod
    def _fill_legacy_defaults(cls, kwargs):
        super()._fill_legacy_defaults(kwargs)
        kwargs.setdefault("attn_query_norm", True)

    @staticmethod
    def _feature_length(n_times: int, kernel_size: int) -> int:
        """Closed-form temporal length returned by the MRCNN feature extractor.

        Propagates ``n_times`` through the Conv1d and MaxPool1d layers of both
        MRCNN branches and sums the two branch lengths, matching the time-axis
        concatenation before the length-preserving 1x1 AFR block.
        """
        # Fine branch: (batch, 1, n_times) -> (batch, 128, fine_len)
        fine_len = _branch_output_length(n_times, _MRCNN_FINE_SPECS)
        # Coarse branch: (batch, 1, n_times) -> (batch, 128, coarse_len)
        coarse_len = _branch_output_length(n_times, _mrcnn_coarse_specs(kernel_size))
        # Concatenation along time, preserved by AFR: (batch, after_reduced_cnn_size, fine_len + coarse_len)
        return fine_len + coarse_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        # MRCNN + TCE feature extraction: (batch, 1, n_times) -> (batch, after_reduced_cnn_size, d_model)
        encoded_features = self.feature_extractor(x)
        # Flatten channels and time steps: (batch, after_reduced_cnn_size * d_model)
        encoded_features = encoded_features.contiguous().view(
            encoded_features.shape[0], -1
        )

        # Classification head: (batch, n_outputs)
        return self.final_layer(encoded_features)


class _SELayer(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU):
        super(_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            activation(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channel, length).

        Returns
        -------
        torch.Tensor
            Output tensor after applying the SE recalibration.
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# Layers of ``_SEBasicBlock`` in the order of braindecode 2.0's ``features``.
_AFR_LAYERS = ("conv1", "bn1", "relu", "conv2", "bn2", "se")


class _SEBasicBlock(nn.Module):
    r"""Residual block with squeeze-and-excitation (SE) recalibration, the AFR block.

    Steps, for an input of shape ``(batch, inplanes, n_times)``:

    1. A 1x1 convolution (``conv1``), batch normalization (``bn1``) and activation
       (``relu``).
    2. A 1x1 convolution (``conv2``) and batch normalization (``bn2``).
    3. The SE layer (``se``) rescales the channels.
    4. The input, through a 1x1 convolution and batch normalization when
       ``inplanes != planes``, is added as a residual, then the activation is applied.

    The output has shape ``(batch, planes, n_times)``.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        activation: type[nn.Module] = nn.ReLU,
        *,
        reduction=16,
    ):
        super(_SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = activation()
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = _SELayer(planes, reduction, activation=activation)
        self.downsample = downsample
        self.stride = stride

    def __setstate__(self, state):
        # braindecode 2.0 pickles also hold these layers as ``features``.
        state["_modules"].pop("features", None)
        super().__setstate__(state)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # braindecode 2.0 state dicts also hold these layers as ``features.<index>``.
        for key in [k for k in state_dict if k.startswith(f"{prefix}features.")]:
            del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual SE block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, inplanes, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, planes, n_times).
        """
        residual = x
        # Convolution, normalization and activation: (batch, planes, n_times)
        out = self.relu(self.bn1(self.conv1(x)))
        # Convolution, normalization and SE recalibration: (batch, planes, n_times)
        out = self.se(self.bn2(self.conv2(out)))

        # Match the residual to the output channels: (batch, planes, n_times)
        if self.downsample is not None:
            residual = self.downsample(x)

        # Add the residual, then activate: (batch, planes, n_times)
        out += residual
        out = self.relu(out)

        return out


# Temporal (kernel_size, stride, padding, dilation) specs of the two MRCNN
# branches from Eldele et al. (2021). The fine branch captures high-frequency
# detail; the coarse branch captures low-frequency structure and uses
# kernel_size=6 at 125 Hz and 7 otherwise in its deeper convolutions.
_MRCNN_FINE_SPECS = (
    (50, 6, 24, 1),  # fine conv1
    (8, 2, 4, 1),  # fine max-pool1
    (8, 1, 4, 1),  # fine conv2
    (8, 1, 4, 1),  # fine conv3
    (4, 4, 2, 1),  # fine max-pool2
)
_MRCNN_COARSE_STEM_SPECS = (
    (400, 50, 200, 1),  # coarse conv1
    (4, 2, 2, 1),  # coarse max-pool1
)
_MRCNN_COARSE_DEEP_PADDING = 3
_MRCNN_COARSE_TAIL_POOL = (2, 2, 1, 1)  # coarse max-pool2


def _mrcnn_coarse_specs(kernel_size: int) -> tuple[tuple[int, int, int, int], ...]:
    """Temporal (kernel_size, stride, padding, dilation) specs of the coarse branch."""
    deep_conv = (kernel_size, 1, _MRCNN_COARSE_DEEP_PADDING, 1)
    return (*_MRCNN_COARSE_STEM_SPECS, deep_conv, deep_conv, _MRCNN_COARSE_TAIL_POOL)


def _conv1d_output_length(
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Closed-form output length of a 1-D Conv1d or MaxPool1d layer."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _branch_output_length(
    n_times: int, specs: tuple[tuple[int, int, int, int], ...]
) -> int:
    """Propagate temporal length through one MRCNN branch of Conv1d/MaxPool1d specs."""
    length = int(n_times)
    for kernel_size, stride, padding, dilation in specs:
        length = _conv1d_output_length(length, kernel_size, stride, padding, dilation)
        if length <= 0:
            raise ValueError(
                "Input window has too small `n_times`/`input_window_seconds` for "
                f"AttnSleep's feature extractor: layer output length became "
                f"{length} for input shape (1, 1, {n_times})."
            )
    return length


def _branch_conv(
    in_channels: int, out_channels: int, spec: tuple[int, int, int, int]
) -> nn.Conv1d:
    """Build a bias-free Conv1d from a ``(kernel_size, stride, padding, dilation)`` spec."""
    kernel_size, stride, padding, dilation = spec
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


def _branch_pool(spec: tuple[int, int, int, int]) -> nn.MaxPool1d:
    """Build a MaxPool1d from a ``(kernel_size, stride, padding, dilation)`` spec."""
    kernel_size, stride, padding, dilation = spec
    return nn.MaxPool1d(
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )


class _MRCNN(nn.Module):
    r"""Multi-resolution CNN and adaptive feature recalibration (AFR) block.

    Applies two parallel 1-D convolutional/max-pooling branches (fine and
    coarse) to a single-channel EEG input of shape ``(batch, 1, n_times)``,
    concatenates their outputs along time to ``(batch, 128, L_1 + L_2)``,
    applies dropout, and recalibrates channels through a 1x1 residual SE block
    to produce ``(batch, after_reduced_cnn_size, L_1 + L_2)``.
    """

    def __init__(
        self,
        after_reduced_cnn_size,
        kernel_size=7,
        activation: type[nn.Module] = nn.GELU,
        activation_se: type[nn.Module] = nn.ReLU,
    ):
        super(_MRCNN, self).__init__()
        drate = 0.5
        self.GELU = activation()
        f_conv1, f_pool1, f_conv2, f_conv3, f_pool2 = _MRCNN_FINE_SPECS
        c_conv1, c_pool1, c_conv2, c_conv3, c_pool2 = _mrcnn_coarse_specs(kernel_size)

        self.features1 = nn.Sequential(
            _branch_conv(1, 64, f_conv1),
            nn.BatchNorm1d(64),
            self.GELU,
            _branch_pool(f_pool1),
            nn.Dropout(drate),
            _branch_conv(64, 128, f_conv2),
            nn.BatchNorm1d(128),
            self.GELU,
            _branch_conv(128, 128, f_conv3),
            nn.BatchNorm1d(128),
            self.GELU,
            _branch_pool(f_pool2),
        )

        self.features2 = nn.Sequential(
            _branch_conv(1, 64, c_conv1),
            nn.BatchNorm1d(64),
            self.GELU,
            _branch_pool(c_pool1),
            nn.Dropout(drate),
            _branch_conv(64, 128, c_conv2),
            nn.BatchNorm1d(128),
            self.GELU,
            _branch_conv(128, 128, c_conv3),
            nn.BatchNorm1d(128),
            self.GELU,
            _branch_pool(c_pool2),
        )

        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(
            _SEBasicBlock, after_reduced_cnn_size, 1, activation=activation_se
        )

    def _make_layer(
        self, block, planes, blocks, stride=1, activation: type[nn.Module] = nn.ReLU
    ):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, activation=activation)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fine and coarse temporal branches: (batch, 1, n_times) -> (batch, 128, L_1/L_2)
        x1 = self.features1(x)
        x2 = self.features2(x)
        # Concatenate both resolutions along time: (batch, 128, L_1 + L_2)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        # Adaptive feature recalibration: (batch, after_reduced_cnn_size, L_1 + L_2)
        x_concat = self.AFR(x_concat)
        return x_concat


##########################################################################################


class _MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, after_reduced_cnn_size, dropout=0.1):
        """Take in model size and number of heads."""
        super().__init__()
        if (
            isinstance(h, bool)
            or not isinstance(h, Integral)
            or h <= 0
            or d_model % h != 0
        ):
            raise ValueError(
                "n_attn_heads must be a positive integer that divides d_model, "
                f"got n_attn_heads={h!r} and d_model={d_model}."
            )
        h = int(h)
        self.d_per_head = d_model // h
        self.h = h

        base_conv = CausalConv1d(
            in_channels=after_reduced_cnn_size,
            out_channels=after_reduced_cnn_size,
            kernel_size=7,
            stride=1,
        )
        self.convs = nn.ModuleList([deepcopy(base_conv) for _ in range(3)])

        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value: torch.Tensor) -> torch.Tensor:
        """Implements Multi-head attention."""
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_per_head).transpose(1, 2)
        key = (
            self.convs[1](key)
            .view(nbatches, -1, self.h, self.d_per_head)
            .transpose(1, 2)
        )
        value = (
            self.convs[2](value)
            .view(nbatches, -1, self.h, self.d_per_head)
            .transpose(1, 2)
        )

        # exact: the reference op order, same outputs as its hand-written attention
        x = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout.p if self.training else 0.0,
            exact=True,
        )

        # merge heads and project
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_per_head)

        return self.linear(x)


class _ResidualLayerNormAttn(nn.Module):
    r"""Attention sublayer of the encoder, with a residual connection.

    Steps, for ``x``, ``key`` and ``value`` of shape
    ``(batch, after_reduced_cnn_size, d_model)``:

    1. The query is ``x``, layer-normalized if ``norm_query``.
    2. The multi-head attention reads the query, ``key`` and ``value``.
    3. Dropout, then the residual ``x`` is added.

    The output has the shape of ``x``.
    """

    def __init__(self, size, dropout, fn_attn, norm_query):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fn_attn = fn_attn
        self.norm_query = norm_query

    def __setstate__(self, state):
        # Full models pickled before the raw-query option normalized the
        # query. Set the flag on the instance so TorchScript also sees it.
        state.setdefault("norm_query", True)
        super().__setstate__(state)

    def forward(
        self,
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        # The reference attends with the raw query. ``norm_query`` keeps the
        # braindecode 1.0 to 2.0 behaviour.
        query = self.norm(x) if self.norm_query else x
        # Attention output: (batch, after_reduced_cnn_size, d_model)
        out = self.fn_attn(query, key, value)

        return x + self.dropout(out)  # (batch, after_reduced_cnn_size, d_model)


class _ResidualLayerNormFF(nn.Module):
    def __init__(self, size, dropout, fn_ff):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fn_ff = fn_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        x_norm = self.norm(x)

        out = self.fn_ff(x_norm)

        return x + self.dropout(out)


class _TCE(nn.Module):
    r"""
    Transformer Encoder.

    It is a stack of n layers.
    """

    def __init__(self, layer, n):
        super().__init__()

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])

        self.norm = nn.LayerNorm(layer.size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _EncoderLayer(nn.Module):
    r"""
    An encoder layer.

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by _ResidualLayerNorm.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        after_reduced_cnn_size,
        dropout,
        attn_query_norm,
    ):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.residual_self_attn = _ResidualLayerNormAttn(
            size=size,
            dropout=dropout,
            fn_attn=self_attn,
            norm_query=attn_query_norm,
        )
        self.residual_ff = _ResidualLayerNormFF(
            size=size,
            dropout=dropout,
            fn_ff=feed_forward,
        )

        self.conv = CausalConv1d(
            in_channels=after_reduced_cnn_size,
            out_channels=after_reduced_cnn_size,
            kernel_size=7,
            stride=1,
            dilation=1,
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Transformer Encoder."""
        query = self.conv(x_in)
        # Encoder self-attention
        x = self.residual_self_attn(query, x_in, x_in)
        x_ff = self.residual_ff(x)
        return x_ff


class _PositionwiseFeedForward(nn.Module):
    r"""Positionwise feed-forward network."""

    def __init__(
        self, d_model, d_ff, dropout=0.1, activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activate = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements FFN equation."""
        return self.w_2(self.dropout(self.activate(self.w_1(x))))

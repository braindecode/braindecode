# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

import math
import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from mne.utils import deprecated
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d


class AttnSleep(EEGModuleMixin, nn.Module):
    """Sleep Staging Architecture from Eldele et al. (2021) [Eldele2021]_.

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
    to use any other input is likely to make the model perform in unintended ways.

    Parameters
    ----------
    n_tce : int
        Number of TCE clones.
    d_model : int
        Input dimension for the TCE.
        Also the input dimension of the first FC layer in the feed forward
        and the output of the second FC layer in the same.
        Increase for higher sampling rate/signal length.
        It should be divisible by n_attn_heads
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
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_classes : int
        Alias for `n_outputs`.
    input_size_s : float
        Alias for `input_window_seconds`.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    activation_mrcnn: nn.Module, default=nn.ReLU
        Activation function class to apply in the Mask R-CNN layer.
        Should be a PyTorch activation module class like ``nn.ReLU`` or
        ``nn.GELU``. Default is ``nn.GELU``.

    References
    ----------
    .. [Eldele2021] E. Eldele et al., "An Attention-Based Deep Learning Approach for Sleep Stage
        Classification With Single-Channel EEG," in IEEE Transactions on Neural Systems and
        Rehabilitation Engineering, vol. 29, pp. 809-818, 2021, doi: 10.1109/TNSRE.2021.3076234.

    .. [1] https://github.com/emadeldeen24/AttnSleep

    .. [2] https://sleepdata.org/datasets/shhs
    """

    def __init__(
        self,
        sfreq=None,
        n_tce=2,
        d_model=80,
        d_ff=120,
        n_attn_heads=5,
        drop_prob=0.1,
        activation_mrcnn: nn.Module = nn.GELU,
        activation: nn.Module = nn.ReLU,
        input_window_seconds=None,
        n_outputs=None,
        after_reduced_cnn_size=30,
        return_feats=False,
        chs_info=None,
        n_chans=None,
        n_times=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

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

        # the usual kernel size for the mrcnn, for sfreq 100
        kernel_size = 7

        if self.sfreq == 125:
            kernel_size = 6

        mrcnn = _MRCNN(
            after_reduced_cnn_size,
            kernel_size,
            activation=activation_mrcnn,
            activation_se=activation,
        )
        attn = _MultiHeadedAttention(n_attn_heads, d_model, after_reduced_cnn_size)
        ff = _PositionwiseFeedForward(d_model, d_ff, drop_prob, activation=activation)
        tce = _TCE(
            _EncoderLayer(
                d_model, deepcopy(attn), deepcopy(ff), after_reduced_cnn_size, drop_prob
            ),
            n_tce,
        )

        self.feature_extractor = nn.Sequential(mrcnn, tce)
        self.len_last_layer = self._len_last_layer(self.n_times)
        self.return_feats = return_feats

        # TODO: Add new way to handle return features
        """if return_feats:
            raise ValueError("return_feat == True is not accepted anymore")"""
        if not return_feats:
            self.final_layer = nn.Linear(
                d_model * after_reduced_cnn_size, self.n_outputs
            )

    def _len_last_layer(self, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(torch.Tensor(1, 1, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        encoded_features = self.feature_extractor(x)
        encoded_features = encoded_features.contiguous().view(
            encoded_features.shape[0], -1
        )

        if self.return_feats:
            return encoded_features

        return self.final_layer(encoded_features)


class _SELayer(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU):
        super(_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            activation(inplace=True),
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


class _SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        activation: nn.Module = nn.ReLU,
        *,
        reduction=16,
    ):
        super(_SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = activation(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = _SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.se
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SE layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor after applying the SE recalibration.
        """
        residual = x
        out = self.features(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _MRCNN(nn.Module):
    def __init__(
        self,
        after_reduced_cnn_size,
        kernel_size=7,
        activation: nn.Module = nn.GELU,
        activation_se: nn.Module = nn.ReLU,
    ):
        super(_MRCNN, self).__init__()
        drate = 0.5
        self.GELU = activation()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(
                64, 128, kernel_size=kernel_size, stride=1, bias=False, padding=3
            ),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(
                128, 128, kernel_size=kernel_size, stride=1, bias=False, padding=3
            ),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(
            _SEBasicBlock, after_reduced_cnn_size, 1, activate=activation_se
        )

    def _make_layer(
        self, block, planes, blocks, stride=1, activate: nn.Module = nn.ReLU
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activate=activate))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


##########################################################################################


def _attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implementation of Scaled dot product attention"""
    # d_k - dimension of the query and key vectors
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)  # attention weights
    output = torch.matmul(p_attn, value)  # (B, h, T, d_k)
    return output, p_attn


class _MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, after_reduced_cnn_size, dropout=0.1):
        """Take in model size and number of heads."""
        super().__init__()
        assert d_model % h == 0
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
        """Implements Multi-head attention"""
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

        x_raw, attn_weights = _attention(query, key, value)
        # apply dropout to the *weights*
        attn = self.dropout(attn_weights)
        # recompute the weighted sum with dropped weights
        x = torch.matmul(attn, value)

        # stash the pre‑dropout weights if you need them
        self.attn = attn_weights

        # merge heads and project
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_per_head)

        return self.linear(x)


class _ResidualLayerNormAttn(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout, fn_attn):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fn_attn = fn_attn

    def forward(
        self,
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        x_norm = self.norm(x)

        out = self.fn_attn(x_norm, key, value)

        return x + self.dropout(out)


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
    """
    Transformer Encoder
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
    """
    An encoder layer
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by _ResidualLayerNorm.
    """

    def __init__(self, size, self_attn, feed_forward, after_reduced_cnn_size, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.residual_self_attn = _ResidualLayerNormAttn(
            size=size,
            dropout=dropout,
            fn_attn=self_attn,
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
        """Transformer Encoder"""
        query = self.conv(x_in)
        # Encoder self-attention
        x = self.residual_self_attn(query, x_in, x_in)
        x_ff = self.residual_ff(x)
        return x_ff


class _PositionwiseFeedForward(nn.Module):
    """Positionwise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activate = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements FFN equation."""
        return self.w_2(self.dropout(self.activate(self.w_1(x))))


@deprecated(
    "`SleepStagerEldele2021` was renamed to `AttnSleep` in v1.12 to follow original author's name; this alias will be removed in v1.14."
)
class SleepStagerEldele2021(AttnSleep):
    """Deprecated alias for SleepStagerEldele2021."""

    pass

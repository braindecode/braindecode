# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import Chomp1d, MaxNormLinear


class EEGTCNet(EEGModuleMixin, nn.Module):
    r"""EEGTCNet model from Ingolfsson et al (2020) [ingolfsson2020]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: https://braindecode.org/dev/_static/model/eegtcnet.jpg
        :align: center
        :alt: EEGTCNet Architecture

    Combining EEGNet and TCN blocks.

    Parameters
    ----------
    activation : nn.Module, optional
        Activation function to use. Default is `nn.ELU()`.
    depth_multiplier : int, optional
        Depth multiplier for the depthwise convolution. Default is 2.
    filter_1 : int, optional
        Number of temporal filters in the first convolutional layer. Default is 8.
    kern_length : int, optional
        Length of the temporal kernel in the first convolutional layer. Default is 64.
    drop_prob : float, optional
        Default dropout rate, used for both the EEGNet front-end and the TCN
        block unless overridden by ``drop_prob_eeg`` / ``drop_prob_tcn``.
        Default is 0.5.
    depth : int, optional
        Number of residual blocks in the TCN. Default is 2.
    kernel_size : int, optional
        Size of the temporal convolutional kernel in the TCN. Default is 4.
    filters : int, optional
        Number of filters in the TCN convolutional layers. Default is 12.
    max_norm_const : float
        Maximum L2-norm constraint imposed on weights of the last
        fully-connected layer. Defaults to 0.25.
    drop_prob_eeg : float | None, optional
        Dropout rate for the EEGNet front-end. If ``None`` (default), falls back
        to ``drop_prob``. The source/paper uses ``0.2`` for BCI IV 2a.
    drop_prob_tcn : float | None, optional
        Dropout rate for the TCN block. If ``None`` (default), falls back to
        ``drop_prob``. The source/paper uses ``0.3`` for BCI IV 2a.
    tcn_batch_norm : bool, optional
        If ``True`` (default), insert a :class:`~torch.nn.BatchNorm1d` after each
        TCN convolution and use a bias in the residual ``1x1`` downsample, matching
        the original Keras source. Set to ``False`` to get the TCN block of
        braindecode before 1.6.1, for example to load checkpoints trained with it.
        To also get the outputs of that version, set ``act_after_temporal_bn=True``.
        Default is ``True``.

        .. versionadded:: 1.6.1
    act_after_temporal_bn : bool, optional
        If ``True``, apply ``activation`` between the temporal batch norm and the
        depthwise convolution, as braindecode did before 2.1. The authors' code
        has no activation there. Default is ``False``.

        .. versionadded:: 2.1

    References
    ----------
    .. [ingolfsson2020] Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N.,
        Cavigelli, L., & Benini, L. (2020). EEG-TCNet: An accurate temporal
        convolutional network for embedded motor-imagery brain–machine interfaces.
        https://doi.org/10.48550/arXiv.2006.00622
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: eegnet_tc.depthwise_conv.weight

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        activation: type[nn.Module] = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        kern_length: int = 64,
        drop_prob: float = 0.5,
        depth: int = 2,
        kernel_size: int = 4,
        filters: int = 12,
        max_norm_const: float = 0.25,
        drop_prob_eeg: float | None = None,
        drop_prob_tcn: float | None = None,
        tcn_batch_norm: bool = True,
        act_after_temporal_bn: bool = False,
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

        self.activation = activation
        self.drop_prob = drop_prob
        # Separate dropout rates for the EEGNet front-end and the TCN block;
        # fall back to ``drop_prob`` when not explicitly set.
        self.drop_prob_eeg = drop_prob if drop_prob_eeg is None else drop_prob_eeg
        self.drop_prob_tcn = drop_prob if drop_prob_tcn is None else drop_prob_tcn
        self.tcn_batch_norm = tcn_batch_norm
        self.act_after_temporal_bn = act_after_temporal_bn
        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.kern_length = kern_length
        self.depth = depth
        self.kernel_size = kernel_size
        self.filters = filters
        self.max_norm_const = max_norm_const
        self.filter_2 = self.filter_1 * self.depth_multiplier

        self.arrange_dim_input = Rearrange(
            "batch nchans ntimes -> batch 1 ntimes nchans"
        )
        # EEGNet_TC Block
        self.eegnet_tc = _EEGNetTC(
            n_chans=self.n_chans,
            filter_1=self.filter_1,
            kern_length=self.kern_length,
            depth_multiplier=self.depth_multiplier,
            drop_prob=self.drop_prob_eeg,
            activation=self.activation,
            act_after_temporal_bn=self.act_after_temporal_bn,
        )
        self.arrange_dim_eegnet = Rearrange(
            "batch filter2 rtimes 1 -> batch rtimes filter2"
        )

        # TCN Block
        self.tcn_block = _TCNBlock(
            input_dimension=self.filter_2,
            depth=self.depth,
            kernel_size=self.kernel_size,
            filters=self.filters,
            drop_prob=self.drop_prob_tcn,
            activation=self.activation,
            batch_norm=self.tcn_batch_norm,
        )

        # Classification Block
        self.final_layer = MaxNormLinear(
            in_features=self.filters,
            out_features=self.n_outputs,
            max_norm_val=self.max_norm_const,
        )
        delta2 = 2 * (self.filter_2 // 2) - self.filter_2 + 1
        n_pool1_min = max(1, self.filter_1 - delta2)
        delta1 = 2 * (self.kern_length // 2) - self.kern_length + 1
        self._min_n_times = max(1, n_pool1_min * 8 - delta1)

    @classmethod
    def _fill_legacy_defaults(cls, kwargs):
        """Rebuild configs saved before a key existed as they were.

        Saved configs hold every init argument, so a missing key means an
        older save: ``act_after_temporal_bn`` (added in 2.1) defaults to
        ``True`` and ``tcn_batch_norm`` (added in 1.6.1) to ``False``.
        """
        super()._fill_legacy_defaults(kwargs)
        kwargs.setdefault("act_after_temporal_bn", True)
        kwargs.setdefault("tcn_batch_norm", False)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EEGTCNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        # x shape: (batch_size, n_chans, n_times)
        x = self.arrange_dim_input(x)  # (batch_size, 1, n_times, n_chans)
        x = self.eegnet_tc(x)  # (batch_size, filter, reduced_time, 1)

        x = self.arrange_dim_eegnet(x)  # (batch_size, reduced_time, F2)
        x = self.tcn_block(x)  # (batch_size, time_steps, filters)

        # Select the last time step
        x = x[:, -1, :]  # (batch_size, filters)

        x = self.final_layer(x)  # (batch_size, n_outputs)

        return x


class _EEGNetTC(nn.Module):
    r"""EEGNet Temporal Convolutional Network (TCN) block.

    The main difference from our :class:`EEGNet` (braindecode) implementation is the
    kernel and dimensional order. Because of this, we decided to keep this
    implementation in a future issue; we will re-evaluate if it is necessary
    to maintain this separate implementation.

    Steps, for an input of shape ``(batch, 1, n_times, n_chans)`` (time is the
    height axis); ``time`` is ``n_times`` up to the zero padding:

    1. Temporal convolution and batch norm, then ``activation`` if
       ``act_after_temporal_bn``: ``(batch, filter_1, time, n_chans)``.
    2. Depthwise convolution over the channels, batch norm, ``activation``,
       average pooling by 8 and dropout: ``(batch, filter_2, time // 8, 1)``.
    3. Separable convolution (depthwise over time, then pointwise), batch norm,
       ``activation``, average pooling by ``filter_1`` and dropout:
       ``(batch, filter_2, reduced_time, 1)``.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    filter_1 : int
        Number of temporal filters in the first convolutional layer.
    kern_length : int
        Length of the temporal kernel in the first convolutional layer.
    depth_multiplier : int
        Depth multiplier for the depthwise convolution.
    drop_prob : float
        Dropout rate.
    activation : nn.Module
        Activation function.
    act_after_temporal_bn : bool
        If ``True``, apply ``activation`` after the temporal batch norm.
    """

    def __init__(
        self,
        n_chans: int,
        filter_1: int = 8,
        kern_length: int = 64,
        depth_multiplier: int = 2,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.ELU,
        act_after_temporal_bn: bool = False,
    ):
        super().__init__()
        self.activation = activation()
        self.drop_prob = drop_prob
        self.act_after_temporal_bn = act_after_temporal_bn
        self.n_chans = n_chans
        self.filter_1 = filter_1
        self.filter_2 = self.filter_1 * depth_multiplier

        # First Conv2D Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_1,
            kernel_size=(kern_length, 1),
            padding=(kern_length // 2, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.filter_1)

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.filter_1,
            out_channels=self.filter_2,
            kernel_size=(1, n_chans),
            groups=self.filter_1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.filter_2)
        self.pool1 = nn.AvgPool2d(kernel_size=(8, 1))
        self.drop1 = nn.Dropout(p=drop_prob)

        # Separable Convolution (Depthwise + Pointwise)
        self.separable_conv_depthwise = nn.Conv2d(
            in_channels=self.filter_2,
            out_channels=self.filter_2,
            kernel_size=(self.filter_2, 1),
            groups=self.filter_2,
            padding=(self.filter_2 // 2, 0),
            bias=False,
        )
        self.separable_conv_pointwise = nn.Conv2d(
            in_channels=self.filter_2,
            out_channels=self.filter_2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.filter_2)
        self.pool2 = nn.AvgPool2d(kernel_size=(self.filter_1, 1))
        self.drop2 = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_times, n_chans)
        # Temporal filters, applied along time to each channel.
        x = self.conv1(x)  # (batch, filter_1, time, n_chans)
        x = self.bn1(x)
        if self.act_after_temporal_bn:
            x = self.activation(x)

        # Spatial filters: depthwise convolution that collapses the channels.
        x = self.depthwise_conv(x)  # (batch, filter_2, time, 1)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)  # (batch, filter_2, time // 8, 1)
        x = self.drop1(x)

        # Separable convolution: filters over time, then a mix of the filters.
        x = self.separable_conv_depthwise(x)  # (batch, filter_2, time // 8 + 1, 1)
        x = self.separable_conv_pointwise(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)  # (batch, filter_2, reduced_time, 1)
        x = self.drop2(x)

        return x


class _TCNBlock(nn.Module):
    r"""
    Many differences from our Temporal Block (braindecode) implementation.
    Because of this, we decided to keep this implementation in a future issue;
    we will re-evaluate if it is necessary to maintain this separate
    implementation.


    """

    def __init__(
        self,
        input_dimension: int,
        depth: int,
        kernel_size: int,
        filters: int,
        drop_prob: float,
        activation: type[nn.Module] = nn.ELU,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.activation = activation()
        self.drop_prob = drop_prob
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        # The Keras source uses a bias on the residual 1x1 downsample, paired
        # with the per-conv BatchNorm; the previous braindecode default omitted
        # both. Keeping them on the same flag makes ``batch_norm=False`` reproduce
        # the legacy architecture exactly.
        self.downsample = (
            nn.Conv1d(input_dimension, filters, kernel_size=1, bias=batch_norm)
            if input_dimension != filters
            else None
        )

        for i in range(depth):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            in_dim = input_dimension if i == 0 else filters
            conv_block = nn.Sequential(
                *self._conv_unit(in_dim, dilation, padding),
                *self._conv_unit(filters, dilation, padding),
            )
            self.layers.append(conv_block)

    def _conv_unit(self, in_channels: int, dilation: int, padding: int):
        # Conv -> Chomp -> [BatchNorm] -> activation -> Dropout, as in the
        # original EEG-TCNet Keras implementation.
        unit = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            Chomp1d(padding),
        ]
        if self.batch_norm:
            # Keras BatchNormalization defaults: momentum=0.99 (-> 0.01 in
            # PyTorch convention) and epsilon=1e-3.
            unit.append(nn.BatchNorm1d(self.filters, momentum=0.01, eps=1e-3))
        unit += [self.activation, nn.Dropout(self.drop_prob)]
        return unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, input_dimension)
        x = x.permute(0, 2, 1)  # (batch_size, input_dimension, time_steps)

        res = x if self.downsample is None else self.downsample(x)
        for layer in self.layers:
            out = layer(x)
            out = out + res
            out = self.activation(out)
            res = out  # Update residual
            x = out  # Update input for next layer

        out = out.permute(0, 2, 1)  # (batch_size, time_steps, filters)
        return out

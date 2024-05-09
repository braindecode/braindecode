from __future__ import annotations

import numpy as np
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import Ensure4d
from braindecode.models.modules_attention import (
    GSoP,
    SqueezeAndExcitation,
    FCA,
    EncNet,
    ECA,
    GatherExcite,
    GCT,
    SRM,
    CBAM,
    CAT,
    CATLite,
)


class _FeatureExtractor(nn.Module):
    """
    A module for feature extraction of the data with temporal and spatial
    transformations.

    This module sequentially processes the input through a series of layers:
    rearrangement, temporal convolution, batch normalization, spatial convolution,
    another batch normalization, an ELU non-linearity, average pooling, and dropout.


    Parameters
    ----------
    n_chans : int
        The number of channels in the input data.
    n_temporal_filters : int, optional
        The number of filters to use in the temporal convolution layer. Default is 40.
    temporal_filter_length : int, optional
        The size of each filter in the temporal convolution layer. Default is 25.
    spatial_expansion : int, optional
        The expansion factor of the spatial convolution layer, determining the number
        of output channels relative to the number of temporal filters. Default is 1.
    pool_length : int, optional
        The size of the window for the average pooling operation. Default is 75.
    pool_stride : int, optional
        The stride of the average pooling operation. Default is 15.
    dropout : float, optional
        The dropout rate for regularization. Default is 0.5.
    """

    def __init__(
        self,
        n_chans: int,
        n_temporal_filters: int = 40,
        temporal_filter_length: int = 25,
        spatial_expansion: int = 1,
        pool_length: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.ensure4d = Ensure4d()
        self.rearrange_input = Rearrange("b c t 1 -> b 1 c t")
        self.temporal_conv = nn.Conv2d(
            1,
            n_temporal_filters,
            kernel_size=(1, temporal_filter_length),
            padding=(0, temporal_filter_length // 2),
            bias=False,
        )
        self.intermediate_bn = nn.BatchNorm2d(n_temporal_filters)
        self.spatial_conv = nn.Conv2d(
            n_temporal_filters,
            n_temporal_filters * spatial_expansion,
            kernel_size=(n_chans, 1),
            groups=n_temporal_filters,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(n_temporal_filters * spatial_expansion)
        self.nonlinearity = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ensure4d(x)
        x = self.rearrange_input(x)
        x = self.temporal_conv(x)
        x = self.intermediate_bn(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class _ChannelAttentionBlock(nn.Module):
    """
    A neural network module implementing channel-wise attention mechanisms to enhance
    feature representations by selectively emphasizing important channels and suppressing
    less useful ones. This block integrates convolutional layers, pooling, dropout, and
    an optional attention mechanism that can be customized based on the given mode.

    Parameters
    ----------
    attention_mode : str, optional
        The type of attention mechanism to apply. If `None`, no attention is applied.
        - "se" for Squeeze-and-excitation network
        - "gsop" for Global Second-Order Pooling
        - "fca" for Frequency Channel Attention Network
        - "encnet" for context encoding module
        - "eca" for Efficient channel attention for deep convolutional neural networks
        - "ge" for Gather-Excite
        - "gct" for Gated Channel Transformation
        - "srm" for Style-based Recalibration Module
        - "cbam" for Convolutional Block Attention Module
        - "cat" for Learning to collaborate channel and temporal attention
        from multi-information fusion
        - "catlite" for Learning to collaborate channel attention
        from multi-information fusion (lite version, cat w/o temporal attention)

    in_channels : int, default=16
        The number of input channels to the block.
    temp_filter_length : int, default=15
        The length of the temporal filters in the convolutional layers.
    pool_length : int, default=8
        The length of the window for the average pooling operation.
    pool_stride : int, default=8
        The stride of the average pooling operation.
    dropout : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.
    reduction_rate : int, default=4
        The reduction rate used in the attention mechanism to reduce dimensionality
        and computational complexity.
    use_mlp : bool, default=False
        Flag to indicate whether an MLP (Multi-Layer Perceptron) should be used within
        the attention mechanism for further processing.
    seq_len : int, default=62
        The sequence length, used in certain types of attention mechanisms to process
        temporal dimensions.
    freq_idx : int, default=0
        DCT index used in fca attention mechanism.
    n_codewords : int, default=4
        The number of codewords (clusters) used in attention mechanisms that employ
        quantization or clustering strategies.
    kernel_size : int, default=9
        The kernel size used in certain types of attention mechanisms for convolution
        operations.
    extra_params : bool, default=False
        Flag to indicate whether additional, custom parameters should be passed to
        the attention mechanism.

    Attributes
    ----------
    conv : torch.nn.Sequential
        Sequential model of convolutional layers, batch normalization, and ELU
        activation, designed to process input features.
    pool : torch.nn.AvgPool2d
        Average pooling layer to reduce the dimensionality of the feature maps.
    dropout : torch.nn.Dropout
        Dropout layer for regularization.
    attention_block : torch.nn.Module or None
        The attention mechanism applied to the output of the convolutional layers,
        if `attention_mode` is not None. Otherwise, it's set to None.

    Examples
    --------
    >>> channel_attention_block = _ChannelAttentionBlock(attention_mode='cbam', in_channels=16, reduction_rate=4, kernel_size=7)
    >>> x = torch.randn(1, 16, 64, 64)  # Example input tensor
    >>> output = channel_attention_block(x)
    The output tensor then can be further processed or used as input to another block.

    """

    def __init__(
        self,
        attention_mode: str | None = None,
        in_channels: int = 16,
        temp_filter_length: int = 15,
        pool_length: int = 8,
        pool_stride: int = 8,
        dropout: float = 0.5,
        reduction_rate: int = 4,
        use_mlp: bool = False,
        seq_len: int = 62,
        freq_idx: int = 0,
        n_codewords: int = 4,
        kernel_size: int = 9,
        extra_params: bool = False,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                (1, temp_filter_length),
                padding=(0, temp_filter_length // 2),
                bias=False,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, in_channels, (1, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
        )

        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        if attention_mode is not None:
            self.attention_block = get_attention_block(
                attention_mode,
                ch_dim=in_channels,
                reduction_rate=reduction_rate,
                use_mlp=use_mlp,
                seq_len=seq_len,
                freq_idx=freq_idx,
                n_codewords=n_codewords,
                kernel_size=kernel_size,
                extra_params=extra_params,
            )
        else:
            self.attention_block = None

    def forward(self, x):
        out = self.conv(x)
        if self.attention_block is not None:
            out = self.attention_block(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class AttentionBaseNet(EEGModuleMixin, nn.Module):
    """AttentionBaseNet.

    Neural Network from the paper: EEG motor imagery decoding:
    A framework for comparative analysis with channel attention
    mechanisms

    The paper and original code with more details about the methodological
    choices are available at the [Martin2023]_ and [MartinCode]_.

    The AttentionBaseNet architecture is composed of four modules:
    - Input Block that performs a temporal convolution and a spatial
    convolution.
    - Channel Expansion that modifies the number of channels.
    - An attention block that performs channel attention with several
    options
    - ClassificationHead

    .. versionadded:: 0.9

    Parameters
    ----------

    References
    ----------
    .. [Martin2023] Wimpff, M., Gizzi, L., Zerfowski, J. and Yang, B., 2023.
        EEG motor imagery decoding: A framework for comparative analysis with
        channel attention mechanisms. arXiv preprint arXiv:2310.11198.
    .. [MartinCode] Wimpff, M., Gizzi, L., Zerfowski, J. and Yang, B.
        GitHub https://github.com/martinwimpff/channel-attention (accessed 2024-03-28)
    """

    def __init__(
        self,
        n_times=None,
        n_chans=None,
        n_outputs=None,
        chs_info=None,
        sfreq=None,
        input_window_seconds=None,
        n_temporal_filters: int = 40,
        temp_filter_length_inp: int = 25,
        spatial_expansion: int = 1,
        pool_length_inp: int = 75,
        pool_stride_inp: int = 15,
        dropout_inp: float = 0.5,
        ch_dim: int = 16,
        temp_filter_length: int = 15,
        pool_length: int = 8,
        pool_stride: int = 8,
        dropout: float = 0.5,
        attention_mode: str | None = None,
        reduction_rate: int = 4,
        use_mlp: bool = False,
        freq_idx: int = 0,
        n_codewords: int = 4,
        kernel_size: int = 9,
        extra_params: bool = False,
    ):
        super(AttentionBaseNet, self).__init__()

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        self.input_block = _FeatureExtractor(
            n_chans=self.n_chans,
            n_temporal_filters=n_temporal_filters,
            temporal_filter_length=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length=pool_length_inp,
            pool_stride=pool_stride_inp,
            dropout=dropout_inp,
        )

        self.channel_expansion = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters * spatial_expansion, ch_dim, (1, 1), bias=False
            ),
            nn.BatchNorm2d(ch_dim),
            nn.ELU(),
        )

        seq_lengths = self._calculate_sequence_lengths(
            self.n_times,
            [temp_filter_length_inp, temp_filter_length],
            [pool_length_inp, pool_length],
            [pool_stride_inp, pool_stride],
        )

        self.channel_attention_block = _ChannelAttentionBlock(
            attention_mode=attention_mode,
            in_channels=ch_dim,
            temp_filter_length=temp_filter_length,
            pool_length=pool_length,
            pool_stride=pool_stride,
            dropout=dropout,
            reduction_rate=reduction_rate,
            use_mlp=use_mlp,
            seq_len=seq_lengths[0],
            freq_idx=freq_idx,
            n_codewords=n_codewords,
            kernel_size=kernel_size,
            extra_params=extra_params,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_lengths[-1] * ch_dim, self.n_outputs)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.channel_attention_block(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _calculate_sequence_lengths(
        input_window_samples: int,
        kernel_lengths: list,
        pool_lengths: list,
        pool_strides: list,
    ):
        seq_lengths = []
        out = input_window_samples
        for k, pl, ps in zip(kernel_lengths, pool_lengths, pool_strides):
            out = np.floor(out + 2 * (k // 2) - k + 1)
            out = np.floor((out - pl) / ps + 1)
            seq_lengths.append(int(out))
        return seq_lengths


def get_attention_block(
    attention_mode: str,
    ch_dim: int = 16,
    reduction_rate: int = 4,
    use_mlp: bool = False,
    seq_len: int | None = None,
    freq_idx: int = 0,
    n_codewords: int = 4,
    kernel_size: int = 9,
    extra_params: bool = False,
):
    """
    Util function to the attention block based on the attention mode.

    Parameters
    ----------
    attention_mode: str
        The type of attention mechanism to apply.
    ch_dim: int
        The number of input channels to the block.
    reduction_rate: int
        The reduction rate used in the attention mechanism to reduce
        dimensionality and computational complexity.
        Used in all the methods, except for the
        encnet and eca.
    use_mlp: bool
        Flag to indicate whether an MLP (Multi-Layer Perceptron) should be used
        within the attention mechanism for further processing. Used in the ge
        and srm attention mechanism.
    seq_len: int
        The sequence length, used in certain types of attention mechanisms to
        process temporal dimensions. Used in the ge or fca attention mechanism.
    freq_idx: int
        DCT index used in fca attention mechanism.
    n_codewords: int
        The number of codewords (clusters) used in attention mechanisms
        that employ quantization or clustering strategies, encnet.
    kernel_size: int
        The kernel size used in certain types of attention mechanisms for convolution
        operations, used in the cbam, eca, and cat attention mechanisms.
    extra_params: bool
        Parameter to pass additional parameters to the GatherExcite mechanism.

    Returns
    -------
    nn.Module
        The attention block based on the attention mode.
    """
    if attention_mode == "se":
        return SqueezeAndExcitation(in_channels=ch_dim, reduction_rate=reduction_rate)
    # improving the squeeze module
    elif attention_mode == "gsop":
        return GSoP(in_channels=ch_dim, reduction_rate=reduction_rate)
    elif attention_mode == "fca":
        assert seq_len is not None
        return FCA(
            in_channels=ch_dim,
            seq_len=seq_len,
            reduction_rate=reduction_rate,
            freq_idx=freq_idx,
        )
    elif attention_mode == "encnet":
        return EncNet(in_channels=ch_dim, n_codewords=n_codewords)
    # improving the excitation module
    elif attention_mode == "eca":
        return ECA(in_channels=ch_dim, kernel_size=kernel_size)
    # improving the squeeze and the excitation module
    elif attention_mode == "ge":
        assert seq_len is not None
        return GatherExcite(
            in_channels=ch_dim,
            seq_len=seq_len,
            extra_params=extra_params,
            use_mlp=use_mlp,
            reduction_rate=reduction_rate,
        )
    elif attention_mode == "gct":
        return GCT(in_channels=ch_dim)
    elif attention_mode == "srm":
        return SRM(in_channels=ch_dim, use_mlp=use_mlp, reduction_rate=reduction_rate)
    # temporal and channel attention
    elif attention_mode == "cbam":
        return CBAM(
            in_channels=ch_dim, reduction_rate=reduction_rate, kernel_size=kernel_size
        )
    elif attention_mode == "cat":
        return CAT(
            in_channels=ch_dim, reduction_rate=reduction_rate, kernel_size=kernel_size
        )
    elif attention_mode == "catlite":
        return CATLite(ch_dim, reduction_rate=reduction_rate)
    else:
        raise NotImplementedError

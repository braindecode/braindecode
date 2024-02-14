import numpy as np
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.functions_attention import get_attention_block


class _InputBlock(nn.Module):
    """
    TODO: Add docstring
    """
    def __init__(
            self,
            n_channels: int,
            n_temporal_filters: int = 40,
            temporal_filter_length: int = 25,
            spatial_expansion: int = 1,
            pool_length: int = 75,
            pool_stride: int = 15,
            dropout: float = 0.5
    ):
        super(_InputBlock, self).__init__()
        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.temporal_conv = nn.Conv2d(1, n_temporal_filters,
                                       kernel_size=(1, temporal_filter_length),
                                       padding=(
                                       0, temporal_filter_length // 2),
                                       bias=False)
        self.intermediate_bn = nn.BatchNorm2d(n_temporal_filters)
        self.spatial_conv = nn.Conv2d(n_temporal_filters,
                                      n_temporal_filters * spatial_expansion,
                                      kernel_size=(n_channels, 1),
                                      groups=n_temporal_filters, bias=False)
        self.bn = nn.BatchNorm2d(n_temporal_filters * spatial_expansion)
        self.nonlinearity = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool_length),
                                 stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
    TODO: Add docstring
    """
    def __init__(
            self,
            attention_mode: str = None,
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
            extra_params: bool = False
    ):
        super(_ChannelAttentionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, temp_filter_length),
                      padding=(0, temp_filter_length // 2),
                      bias=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (1, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU())

        self.pool = nn.AvgPool2d((1, pool_length),
                                 stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        if attention_mode is not None:
            self.attention_block = get_attention_block(
                attention_mode, ch_dim=in_channels,
                reduction_rate=reduction_rate,
                use_mlp=use_mlp, seq_len=seq_len, freq_idx=freq_idx,
                n_codewords=n_codewords, kernel_size=kernel_size,
                extra_params=extra_params)
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

    Neural Network from the paper, EEG motor imagery decoding:
    A framework for comparative analysis with channel attention
    mechanisms

    The paper and original code with more details about the methodological
    choices are available at the [Martin2023]_ and [MartinCode]_.
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.channel_attention_block(x)
        x = self.classifier(x)
    The AttentionBaseNet architecture is composed of four modules:
        - Input Block that performa a temporal convolution and a spatial
            convolution.
        - Channel Expansion that increases the number of spatial information.
        - An attention block that performs channel attention with several
        options
        - ClassificationHead

    .. versionadded:: 0.9

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.
    att_depth: int
        Number of self-attention layers.
    att_heads: int
        Number of attention heads.
    att_drop_prob: float
        Dropout rate of the self-attention layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.
    return_features: bool
        If True, the forward method returns the features before the
        last classification layer. Defaults to False.
    n_classes :
        Alias for n_outputs.
    n_channels :
        Alias for n_chans.
    input_window_samples :
        Alias for n_times.
    References
    ----------
    .. [Song2022] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, pp.710-719. https://ieeexplore.ieee.org/document/9991178
    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """
    def __init__(
            self,
            n_times: int,
            n_chans: int,
            n_outputs: int,
            n_temporal_filters: int = 40,
            temp_filter_length_inp: int = 25,
            spatial_expansion: int = 1,
            pool_length_inp: int = 75,
            pool_stride_inp: int = 15,
            dropout_inp: int = 0.5,
            ch_dim: int = 16,
            temp_filter_length: int = 15,
            pool_length: int = 8,
            pool_stride: int = 8,
            dropout: float = 0.5,
            attention_mode: str = None,
            reduction_rate: int = 4,
            use_mlp: bool = False,
            freq_idx: int = 0,
            n_codewords: int = 4,
            kernel_size: int = 9,
            extra_params: bool = False,
            chs_info=None,
            sfreq=None,  # Check if we can replace freq_idx with this
    ):
        super(AttentionBaseNet, self).__init__()

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq

        self.input_block = _InputBlock(
            n_channels=self.n_chans, n_temporal_filters=n_temporal_filters,
            temporal_filter_length=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length=pool_length_inp, pool_stride=pool_stride_inp,
            dropout=dropout_inp)

        self.channel_expansion = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters * spatial_expansion, ch_dim,
                (1, 1), bias=False),
            nn.BatchNorm2d(ch_dim),
            nn.ELU())

        seq_lengths = self._calculate_sequence_lengths(
            self.n_times, [temp_filter_length_inp, temp_filter_length],
            [pool_length_inp, pool_length],
            [pool_stride_inp, pool_stride])

        self.channel_attention_block = _ChannelAttentionBlock(
            attention_mode=attention_mode, in_channels=ch_dim,
            temp_filter_length=temp_filter_length, pool_length=pool_length,
            pool_stride=pool_stride, dropout=dropout,
            reduction_rate=reduction_rate,
            use_mlp=use_mlp, seq_len=seq_lengths[0], freq_idx=freq_idx,
            n_codewords=n_codewords, kernel_size=kernel_size,
            extra_params=extra_params)

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_lengths[-1] * ch_dim, self.n_outputs))

    def forward(self, x):
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.channel_attention_block(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _calculate_sequence_lengths(input_window_samples: int,
                                    kernel_lengths: list,
                                    pool_lengths: list, pool_strides: list):
        seq_lengths = []
        out = input_window_samples
        for (k, pl, ps) in zip(kernel_lengths, pool_lengths, pool_strides):
            out = np.floor(out + 2 * (k // 2) - k + 1)
            out = np.floor((out - pl) / ps + 1)
            seq_lengths.append(int(out))
        return seq_lengths

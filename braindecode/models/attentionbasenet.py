from __future__ import annotations

import math

from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Ensure4d
from braindecode.modules.attention import (
    CAT,
    CBAM,
    ECA,
    FCA,
    GCT,
    SRM,
    CATLite,
    EncNet,
    GatherExcite,
    GSoP,
    SqueezeAndExcitation,
)


class AttentionBaseNet(EEGModuleMixin, nn.Module):
    """AttentionBaseNet from Wimpff M et al. (2023) [Martin2023]_.

    :bdg-success:`Convolution` :bdg-info:`Small Attention`

    .. figure:: https://content.cld.iop.org/journals/1741-2552/21/3/036020/revision2/jnead48b9f2_hr.jpg
        :align: center
        :alt: AttentionBaseNet Architecture
        :width: 640px


    .. rubric:: Architectural Overview

    AttentionBaseNet is a *convolution-first* network with a *channel-attention* stage.
    The end-to-end flow is:

    - (i) :class:`_FeatureExtractor` learns a temporal filter bank and per-filter spatial
      projections (depthwise across electrodes), then condenses time by pooling;
    - (ii) **Channel Expansion** uses a ``1x1`` convolution to set the feature width;
    - (iii) :class:`_ChannelAttentionBlock` refines features via depthwise–pointwise temporal
      convs and an optional channel-attention module (SE/CBAM/ECA/…);
    - (iv) **Classifier** flattens the sequence and applies a linear readout.

    This design mirrors shallow CNN pipelines (EEGNet-style stem) but inserts a pluggable
    attention unit that *re-weights channels* (and optionally temporal positions) before
    classification.


    .. rubric:: Macro Components

    - :class:`_FeatureExtractor` **(Shallow conv stem → condensed feature map)**

        - *Operations.*
        - **Temporal conv** (:class:`torch.nn.Conv2d`) with kernel ``(1, L_t)`` creates a learned
          FIR-like filter bank with ``n_temporal_filters`` maps.
        - **Depthwise spatial conv** (:class:`torch.nn.Conv2d`, ``groups=n_temporal_filters``)
          with kernel ``(n_chans, 1)`` learns per-filter spatial projections over the full montage.
        - **BatchNorm → ELU → AvgPool → Dropout** stabilize and downsample time.
        - Output shape: ``(B, F2, 1, T₁)`` with ``F2 = n_temporal_filters x spatial_expansion``.

    *Interpretability/robustness.* Temporal kernels behave as analyzable FIR filters; the
    depthwise spatial step yields rhythm-specific topographies. Pooling acts as a local
    integrator that reduces variance on short EEG windows.

    - **Channel Expansion**

        - *Operations.*
        - A ``1x1`` conv → BN → activation maps ``F2 → ch_dim`` without changing
          the temporal length ``T₁`` (shape: ``(B, ch_dim, 1, T₁)``).
          This sets the embedding width for the attention block.

    - :class:`_ChannelAttentionBlock` **(temporal refinement + channel attention)**

        - *Operations.*
        - **Depthwise temporal conv** ``(1, L_a)`` (groups=``ch_dim``) + **pointwise ``1x1``**,
          BN and activation → preserves shape ``(B, ch_dim, 1, T₁)`` while refining timing.
        - **Optional attention module** (see *Additional Mechanisms*) applies channel reweighting
          (some variants also apply temporal gating).
        - **AvgPool (1, P₂)** with stride ``(1, S₂)`` and **Dropout** → outputs
          ``(B, ch_dim, 1, T₂)``.

    *Role.* Emphasizes informative channels (and, in certain modes, salient time steps)
    before the classifier; complements the convolutional priors with adaptive re-weighting.

    - **Classifier (aggregation + readout)**

    *Operations.* :class:`torch.nn.Flatten` → :class:`torch.nn.Linear` from
    ``(B, ch_dim·T₂)`` to classes.


    .. rubric:: Convolutional Details

    - **Temporal (where time-domain patterns are learned).**
        Wide kernels in the stem (``(1, L_t)``) act as a learned filter bank for oscillatory
        bands/transients; the attention block’s depthwise temporal conv (``(1, L_a)``) sharpens
        short-term dynamics after downsampling. Pool sizes/strides (``P₁,S₁`` then ``P₂,S₂``)
        set the token rate and effective temporal resolution.

    - **Spatial (how electrodes are processed).**
        A depthwise spatial conv with kernel ``(n_chans, 1)`` spans the full montage to
        learn *per-temporal-filter* spatial projections (no cross-filter mixing at this step),
        mirroring the interpretable spatial stage in shallow CNNs.

    - **Spectral (how frequency content is captured).**
        No explicit Fourier/wavelet transform is used in the stem—spectral selectivity
        emerges from learned temporal kernels. When ``attention_mode="fca"``, a frequency
        channel attention (DCT-based) summarizes frequencies to drive channel weights.


    .. rubric:: Attention / Sequential Modules

    - **Type.** Channel attention chosen by ``attention_mode`` (SE, ECA, CBAM, CAT, GSoP,
        EncNet, GE, GCT, SRM, CATLite). Most operate purely on channels; CBAM/CAT additionally
        include temporal attention.

    - **Shapes.** Input/Output around attention: ``(B, ch_dim, 1, T₁)``. Re-arrangements
        (if any) are internal to the module; the block returns the same shape before pooling.

    - **Role.** Re-weights channels (and optionally time) to highlight informative sources
        and suppress distractors, improving SNR ahead of the linear head.


    .. rubric:: Additional Mechanisms

        - **Attention variants at a glance.**
        - ``"se"``: Squeeze-and-Excitation (global pooling → bottleneck → gates).
        - ``"gsop"``: Global second-order pooling (covariance-aware channel weights).
        - ``"fca"``: Frequency Channel Attention (DCT summary; uses ``seq_len`` and ``freq_idx``).
        - ``"encnet"``: EncNet with learned codewords (uses ``n_codewords``).
        - ``"eca"``: Efficient Channel Attention (local 1-D conv over channel descriptor; uses ``kernel_size``).
        - ``"ge"``: Gather–Excite (context pooling with optional MLP; can use ``extra_params``).
        - ``"gct"``: Gated Channel Transformation (global context normalization + gating).
        - ``"srm"``: Style-based recalibration (mean–std descriptors; optional MLP).
        - ``"cbam"``: Channel then temporal attention (uses ``kernel_size``).
        - ``"cat"`` / ``"catlite"``: Collaborative (channel ± temporal) attention; *lite* omits temporal.
        - **Auto-compatibility on short inputs.**

    If the input duration is too short for the configured kernels/pools, the implementation
    **automatically rescales** temporal lengths/strides downward (with a warning) to keep
    shapes valid and preserve the pipeline semantics.


    .. rubric:: Usage and Configuration

    - ``n_temporal_filters``, ``temporal_filter_length`` and ``spatial_expansion``:
        control the capacity and the number of spatial projections in the stem.
    - ``pool_length_inp``, ``pool_stride_inp`` then ``pool_length``, ``pool_stride``:
        trade temporal resolution for compute; they determine the final sequence length ``T₂``.
    - ``ch_dim``: width after the ``1x1`` expansion and the effective embedding size for attention.
    - ``attention_mode`` + its specific hyperparameters (``reduction_rate``,
        ``kernel_size``, ``seq_len``, ``freq_idx``, ``n_codewords``, ``use_mlp``):
        select and tune the reweighting mechanism.
    - ``drop_prob_inp`` and ``drop_prob_attn``: regularize stem and attention stages.
    - **Training tips.**

    Start with moderate pooling (e.g., ``P₁=75,S₁=15``) and ELU activations; enable attention
    only after the stem learns stable filters. For small datasets, prefer simpler modes
    (``"se"``, ``"eca"``) before heavier ones (``"gsop"``, ``"encnet"``).

    Notes
    -----
    - Sequence length after each stage is computed internally; the final classifier expects
      a flattened ``ch_dim x T₂`` vector.
    - Attention operates on *channel* dimension by design; temporal gating exists only in
      specific variants (CBAM/CAT).
    - The paper and original code with more details about the methodological
      choices are available at the [Martin2023]_ and [MartinCode]_.
    .. versionadded:: 0.9

    Parameters
    ----------
    n_temporal_filters : int, optional
        Number of temporal convolutional filters in the first layer. This defines
        the number of output channels after the temporal convolution.
        Default is 40.
    temp_filter_length : int, default=15
        The length of the temporal filters in the convolutional layers.
    spatial_expansion : int, optional
        Multiplicative factor to expand the spatial dimensions. Used to increase
        the capacity of the model by expanding spatial features. Default is 1.
    pool_length_inp : int, optional
        Length of the pooling window in the input layer. Determines how much
        temporal information is aggregated during pooling. Default is 75.
    pool_stride_inp : int, optional
        Stride of the pooling operation in the input layer. Controls the
        downsampling factor in the temporal dimension. Default is 15.
    drop_prob_inp : float, optional
        Dropout rate applied after the input layer. This is the probability of
        zeroing out elements during training to prevent overfitting.
        Default is 0.5.
    ch_dim : int, optional
        Number of channels in the subsequent convolutional layers. This controls
        the depth of the network after the initial layer. Default is 16.
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
    pool_length : int, default=8
        The length of the window for the average pooling operation.
    pool_stride : int, default=8
        The stride of the average pooling operation.
    drop_prob_attn : float, default=0.5
        The dropout rate for regularization for the attention layer. Values should be between 0 and 1.
    reduction_rate : int, default=4
        The reduction rate used in the attention mechanism to reduce dimensionality
        and computational complexity.
    use_mlp : bool, default=False
        Flag to indicate whether an MLP (Multi-Layer Perceptron) should be used within
        the attention mechanism for further processing.
    freq_idx : int, default=0
        DCT index used in fca attention mechanism.
    n_codewords : int, default=4
        The number of codewords (clusters) used in attention mechanisms that employ
        quantization or clustering strategies.
    kernel_size : int, default=9
        The kernel size used in certain types of attention mechanisms for convolution
        operations.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    extra_params : bool, default=False
        Flag to indicate whether additional, custom parameters should be passed to
        the attention mechanism.

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
        # Module parameters
        n_temporal_filters: int = 40,
        temp_filter_length_inp: int = 25,
        spatial_expansion: int = 1,
        pool_length_inp: int = 75,
        pool_stride_inp: int = 15,
        drop_prob_inp: float = 0.5,
        ch_dim: int = 16,
        temp_filter_length: int = 15,
        pool_length: int = 8,
        pool_stride: int = 8,
        drop_prob_attn: float = 0.5,
        attention_mode: str | None = None,
        reduction_rate: int = 4,
        use_mlp: bool = False,
        freq_idx: int = 0,
        n_codewords: int = 4,
        kernel_size: int = 9,
        activation: nn.Module = nn.ELU,
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

        min_n_times_required = self._get_min_n_times(
            pool_length_inp,
            pool_stride_inp,
            pool_length,
        )

        if self.n_times < min_n_times_required:
            scaling_factor = self.n_times / min_n_times_required
            warn(
                f"n_times ({self.n_times}) is smaller than the minimum required "
                f"({min_n_times_required}) for the current model parameters configuration. "
                "Adjusting parameters to ensure compatibility."
                "Reducing the kernel, pooling, and stride sizes accordingly.\n"
                "Scaling factor: {:.2f}".format(scaling_factor),
                UserWarning,
            )
            # 3. Scale down all temporal parameters proportionally
            # Use max(1, ...) to ensure parameters remain valid
            temp_filter_length_inp = max(
                1, int(temp_filter_length_inp * scaling_factor)
            )
            pool_length_inp = max(1, int(pool_length_inp * scaling_factor))
            pool_stride_inp = max(1, int(pool_stride_inp * scaling_factor))
            temp_filter_length = max(1, int(temp_filter_length * scaling_factor))
            pool_length = max(1, int(pool_length * scaling_factor))
            pool_stride = max(1, int(pool_stride * scaling_factor))

        self.input_block = _FeatureExtractor(
            n_chans=self.n_chans,
            n_temporal_filters=n_temporal_filters,
            temporal_filter_length=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length=pool_length_inp,
            pool_stride=pool_stride_inp,
            drop_prob=drop_prob_inp,
            activation=activation,
        )

        self.channel_expansion = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters * spatial_expansion, ch_dim, (1, 1), bias=False
            ),
            nn.BatchNorm2d(ch_dim),
            activation(),
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
            drop_prob=drop_prob_attn,
            reduction_rate=reduction_rate,
            use_mlp=use_mlp,
            seq_len=seq_lengths[0],
            freq_idx=freq_idx,
            n_codewords=n_codewords,
            kernel_size=kernel_size,
            extra_params=extra_params,
            activation=activation,
        )

        self.final_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_lengths[-1] * ch_dim, self.n_outputs)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.channel_attention_block(x)
        x = self.final_layer(x)
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
            out = math.floor(out + 2 * (k // 2) - k + 1)
            out = math.floor((out - pl) / ps + 1)
            seq_lengths.append(int(out))
        return seq_lengths

    @staticmethod
    def _get_min_n_times(
        pool_length_inp: int,
        pool_stride_inp: int,
        pool_length: int,
    ) -> int:
        """
        Calculates the minimum n_times required for the model to work
        with the given parameters.

        The calculation is based on reversing the pooling operations to
        ensure the input to each is valid.
        """
        # The input to the second pooling layer must be at least its kernel size.
        min_len_for_second_pool = pool_length

        # Reverse the first pooling operation to find the required input size.
        # Formula: min_L_in = Stride * (min_L_out - 1) + Kernel
        min_len = pool_stride_inp * (min_len_for_second_pool - 1) + pool_length_inp
        return min_len


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
    drop_prob : float, optional
        The dropout rate for regularization. Default is 0.5.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    """

    def __init__(
        self,
        n_chans: int,
        n_temporal_filters: int = 40,
        temporal_filter_length: int = 25,
        spatial_expansion: int = 1,
        pool_length: int = 75,
        pool_stride: int = 15,
        drop_prob: float = 0.5,
        activation: nn.Module = nn.ELU,
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
        self.nonlinearity = activation()
        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(drop_prob)

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
    drop_prob : float, default=0.5
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
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

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
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

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
        drop_prob: float = 0.5,
        reduction_rate: int = 4,
        use_mlp: bool = False,
        seq_len: int = 62,
        freq_idx: int = 0,
        n_codewords: int = 4,
        kernel_size: int = 9,
        extra_params: bool = False,
        activation: nn.Module = nn.ELU,
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
            activation(),
        )

        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(drop_prob)

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

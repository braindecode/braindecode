# Authors: Cedric Rommel <cedric.rommel@inria.fr>
#
# License: BSD (3-clause)
import math

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d, Ensure4d, MaxNormLinear


class ATCNet(EEGModuleMixin, nn.Module):
    """ATCNet from Altaheri et al. (2022) [1]_.

    :bdg-success:`Convolution` :bdg-info:`Small Attention`

    .. figure:: https://user-images.githubusercontent.com/25565236/185449791-e8539453-d4fa-41e1-865a-2cf7e91f60ef.png
        :align: center
        :alt: ATCNet Architecture
        :width: 650px

    .. rubric:: Architectural Overview

    ATCNet is a *convolution-first* architecture augmented with a *lightweight attention–TCN*
    sequence module. The end-to-end flow is:

    - (i) :class:`_ConvBlock` learns temporal filter-banks and spatial projections (EEGNet-style),
      downsampling time to a compact feature map;

    - (ii) Sliding Windows carve overlapping temporal windows from this map;

    - (iii) for each window, :class:`_AttentionBlock` applies small multi-head self-attention
      over time, followed by a :class:`_TCNResidualBlock` stack (causal, dilated);

    - (iv) window-level features are aggregated (mean of window logits or concatenation)
      and mapped via a max-norm–constrained linear layer.

    Relative to ViT, ATCNet replaces linear patch projection with learned *temporal–spatial*
    convolutions; it processes *parallel* window encoders (attention→TCN) instead of a deep
    stack; and swaps the MLP head for a TCN suited to 1-D EEG sequences.

    .. rubric:: Macro Components

    - :class:`_ConvBlock` **(Shallow conv stem → feature map)**

        - *Operations.*
        - **Temporal conv** (:class:`torch.nn.Conv2d`) with kernel ``(L_t, 1)`` builds a
            FIR-like filter bank (``F1`` maps).
        - **Depthwise spatial conv** (:class:`torch.nn.Conv2d`, ``groups=F1``) with kernel
          ``(1, n_chans)`` learns per-filter spatial projections (akin to EEGNet’s CSP-like step).
        - **BN → ELU → AvgPool → Dropout** to stabilize and condense activations.
        - **Refining temporal conv** (:class:`torch.nn.Conv2d`) with kernel ``(L_r, 1)`` +
          **BN → ELU → AvgPool → Dropout**.

    The output shape is ``(B, F2, T_c, 1)`` with ``F2 = F1·D`` and ``T_c = T/(P1·P2)``.
    Temporal kernels behave as FIR filters; the depthwise-spatial conv yields frequency-specific
    topographies. Pooling acts as a local integrator, reducing variance and imposing a
    useful inductive bias on short EEG windows.

    - **Sliding-Window Sequencer**

      From the condensed time axis (length ``T_c``), ATCNet forms ``n`` overlapping windows
      of width ``T_w = T_c - n + 1`` (one start per index). Each window produces a sequence
      ``(B, F2, T_w)`` forwarded to its own attention–TCN branch. This creates *parallel*
      encoders over shifted contexts and is key to robustness on nonstationary EEG.

    - :class:`_AttentionBlock` **(small MHA on temporal positions)**

        - *Operations.*
        - Rearrange to ``(B, T_w, F2)``,
        - Normalization :class:`torch.nn.LayerNorm`
        - Custom MultiHeadAttention :class:`_MHA` (``num_heads=H``, per-head dim ``d_h``) + residual add,
        - Dropout :class:`torch.nn.Dropout`
        - Rearrange back to ``(B, F2, T_w)``.


    **Note**: Attention is *local to a window* and purely temporal.

    *Role.* Re-weights evidence across the window, letting the model emphasize informative
    segments (onsets, bursts) before causal convolutions aggregate history.

    - :class:`_TCNResidualBlock` **(causal dilated temporal CNN)**

        - *Operations.*
        - Two :class:`braindecode.modules.CausalConv1d` layers per block with dilation  ``1, 2, 4, …``
        - Across blocks of `torch.nn.ELU` + `torch.nn.BatchNorm1d` + `torch.nn.Dropout`) +
          a residual (identity or 1x1 mapping).
        - The final feature used per window is the *last* causal step ``[..., -1]`` (forecast-style).

    *Role.* Efficient long-range temporal integration with stable gradients; the dilated
    receptive field complements attention’s soft selection.

    - **Aggregation & Classifier**

        - *Operations.*
        - Either (a) map each window feature ``(B, F2)`` to logits via :class:`braindecode.modules.MaxNormLinear`
        and **average** across windows (default, matching official code), or
        - (b) **concatenate** all window features ``(B, n·F2)`` and apply a single :class:`MaxNormLinear`.
        The max-norm constraint regularizes the readout.

    .. rubric:: Convolutional Details

    - **Temporal.** Temporal structure is learned in three places:
        - (1) the stem’s wide ``(L_t, 1)`` conv (learned filter bank),
        - (2) the refining ``(L_r, 1)`` conv after pooling (short-term dynamics), and
        - (3) the TCN’s causal 1-D convolutions with exponentially increasing dilation
          (long-range dependencies). The minimum sequence length required by the TCN stack is
          ``(K_t - 1)·2^{L-1} + 1``; the implementation *auto-scales* kernels/pools/windows
          when inputs are shorter to preserve feasibility.

    - **Spatial.** A depthwise spatial conv spans the **full montage** (kernel ``(1, n_chans)``),
        producing *per-temporal-filter* spatial projections (no cross-filter mixing at this step).
        This mirrors EEGNet’s interpretability: each temporal filter has its own spatial pattern.


    .. rubric:: Attention / Sequential Modules

    - **Type.** Multi-head self-attention with ``H`` heads and per-head dim ``d_h`` implemented
      in :class:`_MHA`, allowing ``embed_dim = H·d_h`` independent of input and output dims.
    - **Shapes.** ``(B, F2, T_w) → (B, T_w, F2) → (B, F2, T_w)``. Attention operates along
      the **temporal** axis within a window; channels/features stay in the embedding dim ``F2``.
    - **Role.** Highlights salient temporal positions prior to causal convolution; small attention
      keeps compute modest while improving context modeling over pooled features.

    .. rubric:: Additional Mechanisms

    - **Parallel encoders over shifted windows.** Improves montage/phase robustness by
      ensembling nearby contexts rather than committing to a single segmentation.
    - **Max-norm classifier.** Enforces weight norm constraints at the readout, a common
      stabilization trick in EEG decoding.
    - **ViT vs. ATCNet (design choices).** Convolutional *nonlinear* projection rather than
      linear patchification; attention followed by **TCN** (not MLP); *parallel* window
      encoders rather than stacked encoders.

    .. rubric:: Usage and Configuration

        - ``conv_block_n_filters (F1)``, ``conv_block_depth_mult (D)`` → capacity of the stem
        (with ``F2 = F1·D`` feeding attention/TCN), dimensions aligned to ``F2``, like :class:`EEGNet`.
        - Pool sizes ``P1,P2`` trade temporal resolution for stability/compute; they set
        ``T_c = T/(P1·P2)`` and thus window width ``T_w``.
        - ``n_windows`` controls the ensemble over shifts (compute ∝ windows).
        - ``att_num_heads``, ``att_head_dim`` set attention capacity; keep ``H·d_h ≈ F2``.
        - ``tcn_depth``, ``tcn_kernel_size`` govern receptive field; larger values demand
        longer inputs (see minimum length above). The implementation warns and *rescales*
        kernels/pools/windows if inputs are too short.
        - **Aggregation choice.** ``concat=False`` (default, average of per-window logits) matches
        the official code; ``concat=True`` mirrors the paper’s concatenation variant.


    Notes
    -----
    - Inputs substantially shorter than the implied minimum length trigger **automatic
      downscaling** of kernels, pools, windows, and TCN kernel size to maintain validity.
    - The attention–TCN sequence operates **per window**; the last causal step is used as the
      window feature, aligning the temporal semantics across windows.

    .. versionadded:: 1.1

        - More detailed documentation of the model.


    Parameters
    ----------
    input_window_seconds : float, optional
        Time length of inputs, in seconds. Defaults to 4.5 s, as in BCI-IV 2a
        dataset.
    sfreq : int, optional
        Sampling frequency of the inputs, in Hz. Default to 250 Hz, as in
        BCI-IV 2a dataset.
    conv_block_n_filters : int
        Number temporal filters in the first convolutional layer of the
        convolutional block, denoted F1 in figure 2 of the paper [1]_. Defaults
        to 16 as in [1]_.
    conv_block_kernel_length_1 : int
        Length of temporal filters in the first convolutional layer of the
        convolutional block, denoted Kc in table 1 of the paper [1]_. Defaults
        to 64 as in [1]_.
    conv_block_kernel_length_2 : int
        Length of temporal filters in the last convolutional layer of the
        convolutional block. Defaults to 16 as in [1]_.
    conv_block_pool_size_1 : int
        Length of first average pooling kernel in the convolutional block.
        Defaults to 8 as in [1]_.
    conv_block_pool_size_2 : int
        Length of first average pooling kernel in the convolutional block,
        denoted P2 in table 1 of the paper [1]_. Defaults to 7 as in [1]_.
    conv_block_depth_mult : int
        Depth multiplier of depthwise convolution in the convolutional block,
        denoted D in table 1 of the paper [1]_. Defaults to 2 as in [1]_.
    conv_block_dropout : float
        Dropout probability used in the convolution block, denoted pc in
        table 1 of the paper [1]_. Defaults to 0.3 as in [1]_.
    n_windows : int
        Number of sliding windows, denoted n in [1]_. Defaults to 5 as in [1]_.
    att_head_dim : int
        Embedding dimension used in each self-attention head, denoted dh in
        table 1 of the paper [1]_. Defaults to 8 as in [1]_.
    att_num_heads : int
        Number of attention heads, denoted H in table 1 of the paper [1]_.
        Defaults to 2 as in [1]_.
    att_dropout : float
        Dropout probability used in the attention block, denoted pa in table 1
        of the paper [1]_. Defaults to 0.5 as in [1]_.
    tcn_depth : int
        Depth of Temporal Convolutional Network block (i.e. number of TCN
        Residual blocks), denoted L in table 1 of the paper [1]_. Defaults to 2
        as in [1]_.
    tcn_kernel_size : int
        Temporal kernel size used in TCN block, denoted Kt in table 1 of the
        paper [1]_. Defaults to 4 as in [1]_.
    tcn_dropout : float
        Dropout probability used in the TCN block, denoted pt in table 1
        of the paper [1]_. Defaults to 0.3 as in [1]_.
    tcn_activation : torch.nn.Module
        Nonlinear activation to use. Defaults to nn.ELU().
    concat : bool
        When ``True``, concatenates each slidding window embedding before
        feeding it to a fully-connected layer, as done in [1]_. When ``False``,
        maps each slidding window to `n_outputs` logits and average them.
        Defaults to ``False`` contrary to what is reported in [1]_, but
        matching what the official code does [2]_.
    max_norm_const : float
        Maximum L2-norm constraint imposed on weights of the last
        fully-connected layer. Defaults to 0.25.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad, M. Alsulaiman (2022).
        *Physics-informed attention temporal convolutional network for EEG-based motor imagery classification.*
        IEEE Transactions on Industrial Informatics. doi:10.1109/TII.2022.3197419.
    .. [2] Official EEG-ATCNet implementation (TensorFlow):
        https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=250.0,
        conv_block_n_filters=16,
        conv_block_kernel_length_1=64,
        conv_block_kernel_length_2=16,
        conv_block_pool_size_1=8,
        conv_block_pool_size_2=7,
        conv_block_depth_mult=2,
        conv_block_dropout=0.3,
        n_windows=5,
        att_head_dim=8,
        att_num_heads=2,
        att_drop_prob=0.5,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_drop_prob=0.3,
        tcn_activation: nn.Module = nn.ELU,
        concat=False,
        max_norm_const=0.25,
        chs_info=None,
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

        # Validate and adjust parameters based on input size

        min_len_tcn = (tcn_kernel_size - 1) * (2 ** (tcn_depth - 1)) + 1
        # Minimum length required to get at least one sliding window
        min_len_sliding = n_windows + min_len_tcn - 1
        # Minimum input size that produces the required feature map length
        min_n_times = min_len_sliding * conv_block_pool_size_1 * conv_block_pool_size_2

        # 2. If the input is shorter, calculate a scaling factor
        if self.n_times < min_n_times:
            scaling_factor = self.n_times / min_n_times
            warn(
                f"n_times ({self.n_times}) is smaller than the minimum required "
                f"({min_n_times}) for the current model parameters configuration. "
                "Adjusting parameters to ensure compatibility."
                "Reducing the kernel, pooling, and stride sizes accordingly."
                "Scaling factor: {:.2f}".format(scaling_factor),
                UserWarning,
            )
            conv_block_kernel_length_1 = max(
                1, int(conv_block_kernel_length_1 * scaling_factor)
            )
            conv_block_kernel_length_2 = max(
                1, int(conv_block_kernel_length_2 * scaling_factor)
            )
            conv_block_pool_size_1 = max(
                1, int(conv_block_pool_size_1 * scaling_factor)
            )
            conv_block_pool_size_2 = max(
                1, int(conv_block_pool_size_2 * scaling_factor)
            )

            # n_windows should be at least 1
            n_windows = max(1, int(n_windows * scaling_factor))

            # tcn_kernel_size must be at least 2 for dilation to work
            tcn_kernel_size = max(2, int(tcn_kernel_size * scaling_factor))

        self.conv_block_n_filters = conv_block_n_filters
        self.conv_block_kernel_length_1 = conv_block_kernel_length_1
        self.conv_block_kernel_length_2 = conv_block_kernel_length_2
        self.conv_block_pool_size_1 = conv_block_pool_size_1
        self.conv_block_pool_size_2 = conv_block_pool_size_2
        self.conv_block_depth_mult = conv_block_depth_mult
        self.conv_block_dropout = conv_block_dropout
        self.n_windows = n_windows
        self.att_head_dim = att_head_dim
        self.att_num_heads = att_num_heads
        self.att_dropout = att_drop_prob
        self.tcn_depth = tcn_depth
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout = tcn_drop_prob
        self.tcn_activation = tcn_activation
        self.concat = concat
        self.max_norm_const = max_norm_const
        self.tcn_n_filters = int(self.conv_block_depth_mult * self.conv_block_n_filters)
        map = dict()
        for w in range(self.n_windows):
            map[f"max_norm_linears.[{w}].weight"] = f"final_layer.[{w}].weight"
            map[f"max_norm_linears.[{w}].bias"] = f"final_layer.[{w}].bias"
        self.mapping = map

        # Check later if we want to keep the Ensure4d. Not sure if we can
        # remove it or replace it with eipsum.
        self.ensuredims = Ensure4d()
        self.dimshuffle = Rearrange("batch C T 1 -> batch 1 T C")

        self.conv_block = _ConvBlock(
            n_channels=self.n_chans,  # input shape: (batch_size, 1, T, C)
            n_filters=conv_block_n_filters,
            kernel_length_1=conv_block_kernel_length_1,
            kernel_length_2=conv_block_kernel_length_2,
            pool_size_1=conv_block_pool_size_1,
            pool_size_2=conv_block_pool_size_2,
            depth_mult=conv_block_depth_mult,
            dropout=conv_block_dropout,
        )

        self.F2 = int(conv_block_depth_mult * conv_block_n_filters)
        self.Tc = int(self.n_times / (conv_block_pool_size_1 * conv_block_pool_size_2))
        self.Tw = self.Tc - self.n_windows + 1

        self.attention_blocks = nn.ModuleList(
            [
                _AttentionBlock(
                    in_shape=self.F2,
                    head_dim=self.att_head_dim,
                    num_heads=att_num_heads,
                    dropout=att_drop_prob,
                )
                for _ in range(self.n_windows)
            ]
        )

        self.temporal_conv_nets = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        _TCNResidualBlock(
                            in_channels=self.F2 if i == 0 else self.tcn_n_filters,
                            kernel_size=self.tcn_kernel_size,
                            n_filters=self.tcn_n_filters,
                            dropout=self.tcn_dropout,
                            activation=self.tcn_activation,
                            dilation=2**i,
                        )
                        for i in range(self.tcn_depth)
                    ]
                )
                for _ in range(self.n_windows)
            ]
        )

        if self.concat:
            self.final_layer = nn.ModuleList(
                [
                    MaxNormLinear(
                        in_features=self.tcn_n_filters * self.n_windows,
                        out_features=self.n_outputs,
                        max_norm_val=self.max_norm_const,
                    )
                ]
            )
        else:
            self.final_layer = nn.ModuleList(
                [
                    MaxNormLinear(
                        in_features=self.tcn_n_filters,
                        out_features=self.n_outputs,
                        max_norm_val=self.max_norm_const,
                    )
                    for _ in range(self.n_windows)
                ]
            )

        self.out_fun = nn.Identity()

    def forward(self, X):
        # Dimension: (batch_size, C, T)
        X = self.ensuredims(X)
        # Dimension: (batch_size, C, T, 1)
        X = self.dimshuffle(X)
        # Dimension: (batch_size, 1, T, C)

        # ----- Sliding window -----
        conv_feat = self.conv_block(X)
        # Dimension: (batch_size, F2, Tc, 1)
        conv_feat = conv_feat.view(-1, self.F2, self.Tc)
        # Dimension: (batch_size, F2, Tc)

        # ----- Sliding window -----
        sw_concat: list[torch.Tensor] = []  # to store sliding window outputs
        # for w in range(self.n_windows):
        for idx, (attention, tcn_module, final_layer) in enumerate(
            zip(self.attention_blocks, self.temporal_conv_nets, self.final_layer)
        ):
            conv_feat_w = conv_feat[..., idx : idx + self.Tw]
            # Dimension: (batch_size, F2, Tw)

            # ----- Attention block -----
            att_feat = attention(conv_feat_w)
            # Dimension: (batch_size, F2, Tw)

            # ----- Temporal convolutional network (TCN) -----
            tcn_feat = tcn_module(att_feat)[..., -1]
            # Dimension: (batch_size, F2)

            # Outputs of sliding window can be either averaged after being
            # mapped by dense layer or concatenated then mapped by a dense
            # layer
            if not self.concat:
                tcn_feat = final_layer(tcn_feat)

            sw_concat.append(tcn_feat)

        # ----- Aggregation and prediction -----
        if self.concat:
            sw_concat_agg = torch.cat(sw_concat, dim=1)
            sw_concat_agg = self.final_layer[0](sw_concat_agg)
        else:
            if len(sw_concat) > 1:  # more than one window
                sw_concat_agg = torch.stack(sw_concat, dim=0)
                sw_concat_agg = torch.mean(sw_concat_agg, dim=0)
            else:  # one window (# windows = 1)
                sw_concat_agg = sw_concat[0]

        return self.out_fun(sw_concat_agg)


class _ConvBlock(nn.Module):
    """Convolutional block proposed in ATCNet [1]_, inspired by the EEGNet
    architecture [2]_.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
           S. M., Hung, C. P., & Lance, B. J. (2018).
           EEGNet: A Compact Convolutional Network for EEG-based
           Brain-Computer Interfaces.
           arXiv preprint arXiv:1611.08024.
    """

    def __init__(
        self,
        n_channels,
        n_filters=16,
        kernel_length_1=64,
        kernel_length_2=16,
        pool_size_1=8,
        pool_size_2=7,
        depth_mult=2,
        dropout=0.3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(kernel_length_1, 1),
            padding="same",
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(num_features=n_filters, eps=1e-4)

        n_depth_kernels = n_filters * depth_mult
        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_depth_kernels,
            groups=n_filters,
            kernel_size=(1, n_channels),
            padding="valid",
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(num_features=n_depth_kernels, eps=1e-4)

        self.activation2 = nn.ELU()

        self.pool2 = nn.AvgPool2d(kernel_size=(pool_size_1, 1))

        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = nn.Conv2d(
            in_channels=n_depth_kernels,
            out_channels=n_depth_kernels,
            kernel_size=(kernel_length_2, 1),
            padding="same",
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(num_features=n_depth_kernels, eps=1e-4)

        self.activation3 = nn.ELU()

        self.pool3 = nn.AvgPool2d(kernel_size=(pool_size_2, 1))

        self.drop3 = nn.Dropout2d(dropout)

    def forward(self, X):
        # ----- Temporal convolution -----
        # Dimension: (batch_size, 1, T, C)
        X = self.conv1(X)
        X = self.bn1(X)
        # Dimension: (batch_size, F1, T, C)

        # ----- Depthwise channels convolution -----
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.activation2(X)
        # Dimension: (batch_size, F1*D, T, 1)
        X = self.pool2(X)
        X = self.drop2(X)
        # Dimension: (batch_size, F1*D, T/P1, 1)

        # ----- "Spatial" convolution -----
        X = self.conv3(X)
        X = self.bn3(X)
        X = self.activation3(X)
        # Dimension: (batch_size, F1*D, T/P1, 1)
        X = self.pool3(X)
        X = self.drop3(X)
        # Dimension: (batch_size, F1*D, T/(P1*P2), 1)

        return X


class _AttentionBlock(nn.Module):
    """Multi Head self Attention (MHA) block used in ATCNet [1]_, inspired from
    [2]_.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] Vaswani, A. et al., "Attention is all you need",
           in Advances in neural information processing systems, 2017.
    """

    def __init__(
        self,
        in_shape=32,
        head_dim=8,
        num_heads=2,
        dropout=0.5,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.head_dim = head_dim
        self.num_heads = num_heads

        # Puts time dimension at -2 and feature dim at -1
        self.dimshuffle = Rearrange("batch C T -> batch T C")

        # Layer normalization
        self.ln = nn.LayerNorm(normalized_shape=in_shape, eps=1e-6)

        # Multi-head self-attention layer
        # (We had to reimplement it since the original code is in tensorflow,
        # where it is possible to have an embedding dimension different than
        # the input and output dimensions, which is not possible in pytorch.)
        self.mha = _MHA(
            input_dim=in_shape,
            head_dim=head_dim,
            output_dim=in_shape,
            num_heads=num_heads,
            dropout=dropout,
        )

        # XXX: This line in the official code is weird, as there is already
        # dropout in the MultiheadAttention layer. They also don't mention
        # any additional dropout between the attention block and TCN in the
        # paper. We are adding it here however to follo so we are removing this
        # for now.
        self.drop = nn.Dropout(0.3)

    def forward(self, X):
        # Dimension: (batch_size, F2, Tw)
        X = self.dimshuffle(X)
        # Dimension: (batch_size, Tw, F2)

        # ----- Layer norm -----
        out = self.ln(X)

        # ----- Self-Attention -----
        out = self.mha(out, out, out)
        # Dimension: (batch_size, Tw, F2)

        # XXX In the paper fig. 1, it is drawn that layer normalization is
        # performed before the skip connection, while it is done afterwards
        # in the official code. Here we follow the code.

        # ----- Skip connection -----
        out = X + self.drop(out)

        # Move back to shape (batch_size, F2, Tw) from the beginning
        return self.dimshuffle(out)


class _TCNResidualBlock(nn.Module):
    """Modified TCN Residual block as proposed in [1]_. Inspired from
    Temporal Convolutional Networks (TCN) [2]_.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] Bai, S., Kolter, J. Z., & Koltun, V.
           "An empirical evaluation of generic convolutional and recurrent
           networks for sequence modeling", 2018.
    """

    def __init__(
        self,
        in_channels,
        kernel_size=4,
        n_filters=32,
        dropout=0.3,
        activation: nn.Module = nn.ELU,
        dilation=1,
    ):
        super().__init__()
        self.activation = activation()
        self.dilation = dilation
        self.dropout = dropout
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        nn.init.kaiming_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm1d(n_filters)

        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        nn.init.kaiming_uniform_(self.conv2.weight)

        self.bn2 = nn.BatchNorm1d(n_filters)

        self.drop2 = nn.Dropout(dropout)

        # Reshape the input for the residual connection when necessary
        if in_channels != n_filters:
            self.reshaping_conv = nn.Conv1d(
                in_channels=in_channels,  # Specify input channels
                out_channels=n_filters,  # Specify output channels
                kernel_size=1,
                padding="same",
            )
        else:
            self.reshaping_conv = nn.Identity()

    def forward(self, X):
        # Dimension: (batch_size, F2, Tw)
        # ----- Double dilated convolutions -----
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.drop2(out)

        X = self.reshaping_conv(X)

        # ----- Residual connection -----
        out = X + out

        return self.activation(out)


class _MHA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Multi-head Attention

        The difference between this module and torch.nn.MultiheadAttention is
        that this module supports embedding dimensions different then input
        and output ones. It also does not support sequences of different
        length.

        Parameters
        ----------
        input_dim : int
            Dimension of query, key and value inputs.
        head_dim : int
            Dimension of embed query, key and value in each head,
            before computing attention.
        output_dim : int
            Output dimension.
        num_heads : int
            Number of heads in the multi-head architecture.
        dropout : float, optional
            Dropout probability on output weights. Default: 0.0 (no dropout).
        """

        super(_MHA, self).__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        # typical choice for the split dimension of the heads
        self.embed_dim = head_dim * num_heads

        # embeddings for multi-head projections
        self.fc_q = nn.Linear(input_dim, self.embed_dim)
        self.fc_k = nn.Linear(input_dim, self.embed_dim)
        self.fc_v = nn.Linear(input_dim, self.embed_dim)

        # output mapping
        self.fc_o = nn.Linear(self.embed_dim, output_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Compute MHA(Q, K, V)

        Parameters
        ----------
        Q: torch.Tensor of size (batch_size, seq_len, input_dim)
            Input query (Q) sequence.
        K: torch.Tensor of size (batch_size, seq_len, input_dim)
            Input key (K) sequence.
        V: torch.Tensor of size (batch_size, seq_len, input_dim)
            Input value (V) sequence.

        Returns
        -------
        O: torch.Tensor of size (batch_size, seq_len, output_dim)
            Output MHA(Q, K, V)
        """
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == self.input_dim

        batch_size, _, _ = Q.shape

        # embedding for multi-head projections (masked or not)
        Q = self.fc_q(Q)  # (B, S, D)
        K, V = self.fc_k(K), self.fc_v(V)  # (B, S, D)

        # Split into num_head vectors (num_heads * batch_size, n/m, head_dim)
        Q_ = torch.cat(Q.split(self.head_dim, -1), 0)  # (B', S, D')
        K_ = torch.cat(K.split(self.head_dim, -1), 0)  # (B', S, D')
        V_ = torch.cat(V.split(self.head_dim, -1), 0)  # (B', S, D')

        # Attention weights of size (num_heads * batch_size, n, m):
        # measures how similar each pair of Q and K is.
        W = torch.softmax(
            Q_.bmm(K_.transpose(-2, -1)) / math.sqrt(self.head_dim),
            -1,  # (B', D', S)
        )  # (B', N, M)

        # Multihead output (batch_size, seq_len, dim):
        # weighted sum of V where a value gets more weight if its corresponding
        # key has larger dot product with the query.
        H = torch.cat(
            (W.bmm(V_)).split(  # (B', S, S)  # (B', S, D')
                batch_size, 0
            ),  # [(B, S, D')] * num_heads
            -1,
        )  # (B, S, D)

        out = self.fc_o(H)

        return self.dropout(out)

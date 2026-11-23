# Authors: Jiquan Wang et al. (original implementation)
#          ptlnextdoor <67443596+ptlnextdoor@users.noreply.github.com>
#          (braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/wjq-learning/CareSleepNet (MIT).
"""CareSleepNet: a hybrid CNN-transformer sleep stager over EEG and EOG."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import LearnedPositionalEncoding, TimmMlp


class CareSleepNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""CareSleepNet from Wang et al. (2024) [caresleepnet2024]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    A sequence-to-sequence sleep stager: it reads ``n_epochs`` consecutive sleep
    epochs of EEG and EOG and returns one prediction per epoch::

        (batch, n_chans, n_times) -> split into n_epochs epochs -> EEG and EOG
        epoch encoders -> (batch, n_epochs, D) each -> cross-modal blocks,
        summed -> positional encoding -> sequence encoder -> linear classifier
        per epoch -> (batch, n_outputs, n_epochs)

    with ``D = encoder_widths[-1]``, 512 by default.

    .. rubric:: Macro Components

    ``CareSleepNet.eeg_encoder`` and ``CareSleepNet.eog_encoder``
        **Operations.** One stage per entry of the ``encoder_*`` tuples, four by
        default. A stage is a multi-scale dilated convolution block (one
        convolution per entry of ``msd_dilations``, 1, 2 and 4 by default, and a
        1x1 shortcut, summed and layer-normalized) followed by max pooling; every
        stage but the first is then followed by a post-norm transformer layer
        over time. The average over time ends the encoder. **Role.** One
        ``D``-dimensional vector per epoch, from the EEG channels and from the
        EOG channels.

    ``CareSleepNet.eog_to_eeg`` and ``CareSleepNet.eeg_to_eog``
        **Operations.** Pre-norm blocks of multi-head attention and an MLP: the
        queries come from one signal, the keys, values and residual from the
        other. The two outputs are summed. **Role.** Cross-modal fusion.

    ``CareSleepNet.positional_encoding`` and ``CareSleepNet.sequence_encoder``
        **Operations.** A learned positional encoding over the epochs and one
        pre-norm transformer layer. **Role.** Context across the epochs.

    ``CareSleepNet.sequence_norm`` and ``CareSleepNet.final_layer``
        **Operations.** A layer norm and a linear classifier, applied to each
        epoch. **Role.** One row of stage logits per epoch.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** dilated convolutions and transformer layers within each
      epoch, then a transformer over the epochs of the sequence.
    - **Spatial:** EEG and EOG channels are encoded apart, each mixed by the
      first convolution of its encoder, then fused by cross-modal attention.
    - **Spectral:** learned by the multi-scale dilated convolutions.

    .. rubric:: Additional Mechanisms

    - The input holds the epochs one after the other in time,
      ``n_times = n_epochs * epoch_samples``, so each epoch spans
      ``n_times // n_epochs`` samples. The last ``n_eog_chans`` channels are
      EOG and the others EEG.
    - The transformer layers of the epoch encoders are BERT layers (post-norm,
      ``eps=transformer_norm_eps``, no dropout between the GELU and the second
      linear layer); the reference's all-ones attention mask is a no-op and is
      left out. The initialization follows the reference.

    .. important::

        The reference code [caresleepnetcode]_ takes signals resampled to
        100 Hz (ISRUC also band-passed at 0.3-40 Hz), cut into sequences of 20
        epochs of 30 s, with 5 classes (W, N1, N2, N3, REM) and, on ISRUC, 6
        EEG and 2 EOG channels. No trained weights were released.

    `License <https://github.com/wjq-learning/CareSleepNet/blob/386e149f7eac8d93207972268f8bc82ea37361c1/LICENSE>`_

    Parameters
    ----------
    n_epochs : int, default=20
        Number of sleep epochs in one input sequence; ``n_times`` must be a
        multiple of it. The reference uses 20 epochs of 30 s at 100 Hz.
    n_eog_chans : int, default=2
        Number of EOG channels, at the end of the channel axis. The reference
        uses 6 EEG and 2 EOG channels on ISRUC.
    n_heads : int, default=8
        Attention heads of the cross-modal and sequence blocks. They must divide
        ``encoder_widths[-1]``.
    encoder_widths : sequence of int, default=(64, 128, 256, 512)
        Output channels of each stage of the epoch encoder. Its length sets the
        number of stages, and the four tuples below have one entry per stage. The
        last entry is also the width of the epoch vectors, of the cross-modal and
        sequence blocks and of the classifier input (512 in the reference).
    encoder_kernel_sizes : sequence of int, default=(49, 9, 9, 9)
        Kernel size of the multi-scale convolutions of each stage. It must be
        odd, so that the padding ``(kernel_size - 1) // 2`` times the dilation
        keeps every branch the length of the strided shortcut.
    encoder_strides : sequence of int, default=(12, 1, 1, 1)
        Stride of the multi-scale convolutions of each stage.
    encoder_pool_kernel_sizes : sequence of int, default=(9, 2, 2, 2)
        Kernel size of the max pooling after each stage, with padding
        ``(kernel_size - 1) // 2``.
    encoder_pool_strides : sequence of int, default=(2, 2, 2, 2)
        Stride of the max pooling after each stage.
    msd_dilations : sequence of int, default=(1, 2, 4)
        Dilation of each convolution branch of a multi-scale block, the same in
        every stage.
    transformer_n_heads : int, default=8
        Attention heads of the epoch encoder's transformer layers. The reference
        sets them in a BERT config of their own, apart from the ``n_heads`` of
        the other blocks. They must divide every entry of ``encoder_widths[1:]``.
    transformer_mlp_ratio : float, default=4.0
        Width of the feed-forward layer of the epoch encoder's transformer
        layers over their width (``intermediate_size`` in the reference). The
        cross-modal and sequence blocks have a feed-forward layer as wide as
        themselves, ``encoder_widths[-1]``.
    drop_prob : float, default=0.1
        Dropout rate of every dropout layer; every dropout rate of the
        reference is 0.1.
    activation : type[nn.Module], default=nn.GELU
        Activation of every convolution branch and transformer MLP.
    norm_eps : float, default=1e-6
        Epsilon of the layer norms outside the epoch encoder's transformer
        layers: those of the multi-scale blocks, the cross-modal blocks and the
        sequence encoder (``eps=1e-6`` in the reference).
    transformer_norm_eps : float, default=1e-12
        Epsilon of the layer norms of the epoch encoder's transformer layers
        (``BertLayerNorm`` in the reference).
    init_std : float, default=0.02
        Std of the normal initialization of the weights of the epoch encoder's
        transformer layers (``initializer_range`` in the reference) and of the
        positional encoding.
    bias_init_std : float, default=1e-6
        Std of the normal initialization of the biases of the MLPs of the
        cross-modal and sequence blocks.

    References
    ----------
    .. [caresleepnet2024] Wang, J., Zhao, S., Jiang, H., Zhou, Y., Yu, Z.,
       Li, T., Li, S., & Pan, G. (2024). CareSleepNet: A Hybrid Deep Learning
       Network for Automatic Sleep Staging. IEEE Journal of Biomedical and
       Health Informatics, 28(12), 7392-7405.
       https://doi.org/10.1109/JBHI.2024.3426939
    .. [caresleepnetcode] Wang, J. CareSleepNet source code.
       https://github.com/wjq-learning/CareSleepNet/tree/386e149f7eac8d93207972268f8bc82ea37361c1
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: eeg_encoder.0.branches.0.0.weight

    def __init__(
        self,
        # Signal-related parameters.
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # Model parameters.
        n_epochs: int = 20,
        n_eog_chans: int = 2,
        n_heads: int = 8,
        encoder_widths: Sequence[int] = (64, 128, 256, 512),
        encoder_kernel_sizes: Sequence[int] = (49, 9, 9, 9),
        encoder_strides: Sequence[int] = (12, 1, 1, 1),
        encoder_pool_kernel_sizes: Sequence[int] = (9, 2, 2, 2),
        encoder_pool_strides: Sequence[int] = (2, 2, 2, 2),
        msd_dilations: Sequence[int] = (1, 2, 4),
        transformer_n_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        norm_eps: float = 1e-6,
        transformer_norm_eps: float = 1e-12,
        init_std: float = 0.02,
        bias_init_std: float = 1e-6,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if not 0 < n_eog_chans < self.n_chans:
            raise ValueError(
                f"n_eog_chans must be between 1 and n_chans - 1 = {self.n_chans - 1}, "
                f"got {n_eog_chans}."
            )
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be at least 1, got {n_epochs}.")
        if self.n_times % n_epochs:
            raise ValueError(
                f"n_times ({self.n_times}) must be a positive multiple of n_epochs ({n_epochs})."
            )
        # The tuples of the epoch encoder have one entry per stage.
        stage_tuples = (
            encoder_widths,
            encoder_kernel_sizes,
            encoder_strides,
            encoder_pool_kernel_sizes,
            encoder_pool_strides,
        )
        if len({len(stage) for stage in stage_tuples}) != 1:
            raise ValueError(
                "encoder_widths, encoder_kernel_sizes, encoder_strides, "
                "encoder_pool_kernel_sizes and encoder_pool_strides must have one "
                f"entry per stage, got lengths {[len(stage) for stage in stage_tuples]}."
            )
        if any(kernel_size % 2 == 0 for kernel_size in encoder_kernel_sizes):
            raise ValueError(
                f"encoder_kernel_sizes must be odd, got {tuple(encoder_kernel_sizes)}: "
                "an even kernel leaves the dilated branches shorter than the shortcut."
            )
        if not encoder_widths:
            raise ValueError("The epoch encoder needs at least one stage.")
        if n_heads < 1 or encoder_widths[-1] % n_heads:
            raise ValueError(
                f"n_heads ({n_heads}) must divide encoder_widths[-1] "
                f"({encoder_widths[-1]})."
            )
        if transformer_n_heads < 1 or any(
            width % transformer_n_heads for width in encoder_widths[1:]
        ):
            raise ValueError(
                f"transformer_n_heads ({transformer_n_heads}) must divide every "
                f"entry of encoder_widths[1:], got {tuple(encoder_widths[1:])}."
            )
        # Shortest epoch the epoch encoder accepts: walk its stages backwards from
        # one time step. A layer that gives n steps needs, at its input,
        # (n - 1) * stride + 1 steps if it is a multi-scale block (odd kernel and
        # "same" padding: it turns L steps into (L - 1) // stride + 1), and
        # (n - 1) * stride + kernel - 2 * padding steps if it is a max pooling.
        min_epoch_samples = 1
        for stride, pool_kernel_size, pool_stride in zip(
            reversed(encoder_strides),
            reversed(encoder_pool_kernel_sizes),
            reversed(encoder_pool_strides),
        ):
            pool_padding = (pool_kernel_size - 1) // 2
            min_epoch_samples = (
                (min_epoch_samples - 1) * pool_stride
                + pool_kernel_size
                - 2 * pool_padding
            )
            min_epoch_samples = (min_epoch_samples - 1) * stride + 1
        if self.n_times // n_epochs < min_epoch_samples:
            raise ValueError(
                f"Each sleep epoch must have at least {min_epoch_samples} samples "
                "for the epoch encoder, got n_times // n_epochs = "
                f"{self.n_times // n_epochs}."
            )
        self.n_epochs = n_epochs
        self.n_eog_chans = n_eog_chans
        self._n_times_step = n_epochs
        self.on_non_divisible = "crop"
        self._min_n_times = n_epochs * min_epoch_samples
        # Read by _init_parameters, which reset_parameters calls again.
        self.init_std = init_std
        self.bias_init_std = bias_init_std
        dim = encoder_widths[-1]  # width of the epoch vectors and of the sequence model

        # The epoch encoders see one epoch at a time: fold the epochs into the
        # batch, then unfold their vectors into a sequence of epochs.
        self.epochs_to_batch = Rearrange(
            "batch chans (epoch time) -> (batch epoch) chans time",
            epoch=n_epochs,
        )
        self.batch_to_epochs = Rearrange(
            "(batch epoch) dim -> batch epoch dim", epoch=n_epochs
        )
        # The classifier gives one row per epoch; the output has them last.
        self.epochs_last = Rearrange("batch epoch out -> batch out epoch")

        epoch_encoder = partial(
            _epoch_encoder,
            widths=encoder_widths,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            pool_kernel_sizes=encoder_pool_kernel_sizes,
            pool_strides=encoder_pool_strides,
            dilations=msd_dilations,
            n_heads=transformer_n_heads,
            mlp_ratio=transformer_mlp_ratio,
            norm_eps=norm_eps,
            transformer_norm_eps=transformer_norm_eps,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.eeg_encoder = epoch_encoder(self.n_chans - n_eog_chans)
        self.eog_encoder = epoch_encoder(n_eog_chans)
        # The cross-modal and sequence blocks have a feed-forward width equal to
        # their own, dim (the reference's ``mlp_dim=512`` for ``hidden_dim=512``).
        self.eog_to_eeg = _CrossModalBlock(
            dim, n_heads, dim, norm_eps, drop_prob, activation
        )
        self.eeg_to_eog = _CrossModalBlock(
            dim, n_heads, dim, norm_eps, drop_prob, activation
        )
        self.positional_encoding = LearnedPositionalEncoding(n_epochs, dim, drop_prob)
        self.sequence_encoder = nn.TransformerEncoderLayer(
            dim,
            n_heads,
            dim,
            drop_prob,
            activation=activation(),
            layer_norm_eps=norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.sequence_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.final_layer = nn.Linear(dim, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialization of the reference: BERT in the epoch encoders,
        torchvision's ViT in the cross-modal and sequence encoders."""
        for encoder in (self.eeg_encoder, self.eog_encoder):
            for layer in encoder:
                if isinstance(layer, nn.TransformerEncoderLayer):
                    for linear in (
                        layer.self_attn.out_proj,
                        layer.linear1,
                        layer.linear2,
                    ):
                        nn.init.normal_(linear.weight, std=self.init_std)
                        nn.init.zeros_(linear.bias)
                    nn.init.normal_(layer.self_attn.in_proj_weight, std=self.init_std)
        vit_linears = (
            self.eog_to_eeg.mlp.fc1,
            self.eog_to_eeg.mlp.fc2,
            self.eeg_to_eog.mlp.fc1,
            self.eeg_to_eog.mlp.fc2,
            self.sequence_encoder.linear1,
            self.sequence_encoder.linear2,
        )
        for linear in vit_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.normal_(linear.bias, std=self.bias_init_std)
        nn.init.normal_(self.positional_encoding.encoding, std=self.init_std)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the per-epoch classifier for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a sleep stage for each epoch of a sequence.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, n_chans, n_times)``: the epochs one after the other in
            time, the EEG channels first and the last ``n_eog_chans`` EOG.

        Returns
        -------
        torch.Tensor
            ``(batch, n_outputs, n_epochs)``.
        """
        # x: (batch, n_chans, n_times), the epochs one after the other in time
        x = self.epochs_to_batch(x)  # (batch * n_epochs, n_chans, epoch_samples)
        # The EEG and EOG channels, each with its own epoch encoder that gives
        # one vector per epoch.
        eeg, eog = x.split([self.n_chans - self.n_eog_chans, self.n_eog_chans], dim=1)
        # (batch, n_epochs, D) each, with D = encoder_widths[-1]
        eeg = self.batch_to_epochs(self.eeg_encoder(eeg))
        eog = self.batch_to_epochs(self.eog_encoder(eog))
        # Each signal attends to the other; the two views are summed:
        # (batch, n_epochs, D)
        x = self.eog_to_eeg(eeg, eog) + self.eeg_to_eog(eog, eeg)
        # Context across the epochs of the sequence: (batch, n_epochs, D)
        x = self.sequence_encoder(self.positional_encoding(x))
        # One row of logits per epoch: (batch, n_epochs, n_outputs)
        x = self.final_layer(self.sequence_norm(x))
        return self.epochs_last(x)  # (batch, n_outputs, n_epochs)


def _epoch_encoder(
    in_chans: int,
    *,
    widths: Sequence[int],
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    pool_kernel_sizes: Sequence[int],
    pool_strides: Sequence[int],
    dilations: Sequence[int],
    n_heads: int,
    mlp_ratio: float,
    norm_eps: float,
    transformer_norm_eps: float,
    drop_prob: float,
    activation: type[nn.Module],
) -> nn.Sequential:
    """Encode each sleep epoch of one signal type into a vector.

    ``(batch, in_chans, epoch_samples)`` to ``(batch, widths[-1])``. One stage per
    entry of the tuples: a multi-scale dilated convolution (``_MSDConv``) that
    widens the channels to ``widths[stage]``, then a max pooling; after every
    stage but the first, a BERT layer over the time steps. The average over time
    ends the encoder. A stage takes the time axis from ``T`` to
    ``ceil(T / stride)`` steps with its convolution, then to
    ``(T + 2p - k) // s + 1`` with a pooling of kernel ``k``, stride ``s`` and
    padding ``p = (k - 1) // 2``.
    """
    msd_conv = partial(
        _MSDConv,
        dilations=dilations,
        norm_eps=norm_eps,
        drop_prob=drop_prob,
        activation=activation,
    )
    # the BERT layers take (batch, time, channels), the convolutions and the
    # pooling (batch, channels, time)
    to_tokens = Rearrange("batch chans time -> batch time chans")
    to_channels = Rearrange("batch time chans -> batch chans time")
    layers: list[nn.Module] = []
    stages = zip(widths, kernel_sizes, strides, pool_kernel_sizes, pool_strides)
    for stage, (width, kernel, stride, pool_kernel, pool_stride) in enumerate(stages):
        layers += [
            # (batch, width, ceil(time / stride))
            msd_conv(in_chans, width, kernel, stride),
            # (batch, width, time after the pooling), with the padding of "same"
            nn.MaxPool1d(pool_kernel, pool_stride, (pool_kernel - 1) // 2),
        ]
        if stage > 0:
            # (batch, time, width), a BERT layer, then back to (batch, width, time)
            bert = _bert_layer(
                width, n_heads, mlp_ratio, transformer_norm_eps, drop_prob, activation
            )
            layers += [to_tokens, bert, to_channels]
        in_chans = width
    # average over time: (batch, widths[-1], 1), then (batch, widths[-1])
    layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
    return nn.Sequential(*layers)


def _bert_layer(
    dim: int,
    n_heads: int,
    mlp_ratio: float,
    norm_eps: float,
    drop_prob: float,
    activation: type[nn.Module],
) -> nn.TransformerEncoderLayer:
    """BERT layer of the reference (post-norm, ``n_heads`` heads, feed-forward
    width ``mlp_ratio * dim``), from ``(batch, time, dim)`` to the same shape."""
    layer = nn.TransformerEncoderLayer(
        dim,
        n_heads,
        int(dim * mlp_ratio),
        drop_prob,
        activation=activation(),
        layer_norm_eps=norm_eps,
        batch_first=True,
    )
    layer.dropout = nn.Identity()  # BERT has no dropout after the GELU
    return layer


class _MSDConv(nn.Module):
    """Multi-scale dilated convolution.

    Steps: one dilated convolution per entry of ``dilations`` (1, 2, 4 in the
    reference) with batch norm and activation, plus a 1x1 shortcut with batch norm,
    are summed; then dropout and a layer norm over the channels. Each convolution
    pads ``padding`` times its dilation, with ``padding = (kernel_size - 1) // 2``,
    so that, for an odd kernel, every branch has the length of the strided
    shortcut: ``(batch, in_chans, time)`` to
    ``(batch, out_chans, ceil(time / stride))``.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        dilations: Sequence[int],
        norm_eps: float,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        # the padding of "same" for an odd kernel, before the dilation
        padding = (kernel_size - 1) // 2
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, 1, stride, bias=False),
            nn.BatchNorm1d(out_chans),
        )
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    in_chans,
                    out_chans,
                    kernel_size,
                    stride,
                    padding * dilation,
                    dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(out_chans),
                activation(),
            )
            for dilation in dilations
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(out_chans, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # strided 1x1 shortcut: (batch, in_chans, time) -> (batch, out_chans, new_time)
        out = self.shortcut(x)
        for branch in self.branches:
            # add each dilated branch, of the same shape as the shortcut
            out = out + branch(x)
        out = self.dropout(out)
        # the layer norm takes the channels last: (batch, new_time, out_chans),
        # then back to (batch, out_chans, new_time)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class _CrossModalBlock(nn.Module):
    """Pre-norm block whose queries come from the other signal (EEG or EOG).

    Unlike a transformer decoder block it has no self-attention, and the
    residual is the attended sequence ``x``, not the queries; no shared block
    does that.

    Steps: dropout and layer norm of ``x`` and of ``query``; multi-head
    attention of the queries over ``x``, added to ``x``; an MLP on a layer-normed
    ``x``, added to ``x``; a last layer norm. ``x`` and ``query`` are
    ``(batch, n_epochs, dim)`` and so is the output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        norm_eps: float,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.drop_x = nn.Dropout(drop_prob)
        self.drop_query = nn.Dropout(drop_prob)
        self.norm_x = nn.LayerNorm(dim, eps=norm_eps)
        self.norm_query = nn.LayerNorm(dim, eps=norm_eps)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=drop_prob, batch_first=True
        )
        self.drop_attention = nn.Dropout(drop_prob)
        self.norm_mlp = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = TimmMlp(dim, mlp_dim, act_layer=activation, drop=drop_prob)
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_epochs, dim), the keys, values and residual
        x = self.drop_x(x)
        # query: (batch, n_epochs, dim), from the other signal
        query = self.norm_query(self.drop_query(query))
        x_normed = self.norm_x(x)
        # each epoch of the query signal attends over the epochs of x:
        # (batch, n_epochs, dim)
        attended, _ = self.attention(query, x_normed, x_normed, need_weights=False)
        x = x + self.drop_attention(attended)
        # MLP block: (batch, n_epochs, dim)
        x = x + self.mlp(self.norm_mlp(x))
        return self.norm_out(x)

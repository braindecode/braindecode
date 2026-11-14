# Authors: Panagiotis Tsinganos et al. (original implementation)
#          Arslan Shafique <81501094+arslan-shafique@users.noreply.github.com>
#          OpenAI Codex (Braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/DSIP-UPatras/ICASSP2019_TCN
# at commit cd28909ccf3696b835446cbec2f15eb60610a2af.
"""TCCNet for surface electromyography gesture recognition."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin


class TCCNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""Temporal convolutional network for sEMG gesture recognition.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`
    :bdg-info:`Attention/Transformer`

    TCCNet was introduced by Tsinganos et al. for subject-specific gesture
    recognition on NinaPro DB1 [tccnet2019]_. It applies residual causal
    convolutions with exponentially increasing dilation, then summarizes the
    temporal representation with either average-over-time (``"aot"``) or
    learned context attention (``"att"``).

    This implementation follows the authors' MIT-licensed Keras source pinned
    at commit ``cd28909ccf3696b835446cbec2f15eb60610a2af`` [tccnetcode]_.
    Braindecode inputs are transposed internally from
    ``(batch, n_chans, n_times)`` to the source layout
    ``(batch, n_times, n_chans)``. The source's final softmax is intentionally
    omitted because Braindecode models return logits.

    The paper experiments use 10-channel NinaPro DB1 sEMG sampled at 100 Hz.
    Each channel is rectified and low-pass filtered by a first-order Butterworth
    filter at 1 Hz. A sample is then min-max normalized across its unpadded
    values. Repetitions 1, 3, 4, 6, 8, 9 and 10 form the training set; 2, 5
    and 7 form the validation set. Variable-length gestures are pre-padded with
    ``-10`` in the reference pipeline, which can be reproduced by setting
    ``mask_value=-10.0``.

    Parameters
    ----------
    filters : sequence of sequence of int, default=((32, 32), (32, 32), (64, 64), (128, 128))
        Output filters of each causal convolution, one inner sequence per
        residual block. It sets the number of blocks and of convolutions per
        block.
    kernel_size : int, default=3
        Temporal kernel size of every causal convolution.
    drop_prob : float, default=0.5
        Channel-wise dropout probability after each activated convolution,
        equivalent to Keras ``SpatialDropout1D``.
    activation : type[nn.Module], default=nn.ReLU
        Activation class applied after each causal convolution.
    classification_head : {"aot", "att"}, default="aot"
        Temporal aggregation head. ``"aot"`` averages valid time steps;
        ``"att"`` uses the context-vector attention from the reference source.
    mask_value : float | None, default=None
        A time step for which every channel equals this value is treated as
        padding. Use ``-10.0`` for the paper's variable-length preprocessing.
        ``None`` disables masking, matching the reference constructor default.

    Notes
    -----
    The Keras source applies L2 regularization to convolution and classifier
    kernels. In PyTorch this should be configured as optimizer weight decay;
    it is not part of the forward architecture.

    The published repository's 300-sample configuration uses
    ``filters=((32, 64), (64, 128))``. Its variable-length configuration uses
    ``filters=((32, 32), (32, 32), (64, 64), (128,))``. Both set
    ``drop_prob=0.05`` and train for 30 epochs with Adam (learning rate 0.01,
    epsilon 0.001), L2 coefficient 0.0005, time and magnitude warping of 0.2,
    and additive noise at 25 dB SNR.

    References
    ----------
    .. [tccnet2019] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., &
       Skodras, A. (2019). Improved Gesture Recognition Based on sEMG Signals
       and TCN. ICASSP 2019, 1169-1173.
       https://doi.org/10.1109/ICASSP.2019.8683239
    .. [tccnetcode] Tsinganos, P. et al. ICASSP2019_TCN source code.
       https://github.com/DSIP-UPatras/ICASSP2019_TCN/tree/cd28909ccf3696b835446cbec2f15eb60610a2af
    """

    def __init__(
        self,
        # Signal-related parameters.
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters.
        filters: Sequence[Sequence[int]] = (
            (32, 32),
            (32, 32),
            (64, 64),
            (128, 128),
        ),
        kernel_size: int = 3,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.ReLU,
        classification_head: str = "aot",
        mask_value: float | None = None,
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

        if classification_head not in ("aot", "att"):
            raise ValueError(
                f"classification_head must be 'aot' or 'att', got {classification_head!r}."
            )
        self.mask_value = mask_value

        blocks = []
        in_channels = self.n_chans
        dilation = 1
        for block_filters in filters:
            # The dilation doubles at every convolution, across blocks too.
            dilations = [dilation * 2**i for i in range(len(block_filters))]
            dilation *= 2 ** len(block_filters)
            blocks.append(
                _TCCResidualBlock(
                    in_channels,
                    block_filters,
                    kernel_size,
                    dilations,
                    drop_prob,
                    activation,
                )
            )
            in_channels = block_filters[-1]
        self.feature_extractor = nn.ModuleList(blocks)

        self.attention = (
            _TCCContextAttention(in_channels) if classification_head == "att" else None
        )
        self.final_layer = nn.Linear(in_channels, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Apply the source Glorot initialization to the classifier."""
        nn.init.xavier_normal_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits for a batch of sEMG windows."""
        valid_mask = None
        if self.mask_value is not None:
            valid_mask = torch.any(x != self.mask_value, dim=1)

        features = x
        for block in self.feature_extractor:
            features = block(features, valid_mask)
        features = features.transpose(1, 2)

        if self.attention is not None:
            pooled = self.attention(features, valid_mask)
        elif valid_mask is None:
            pooled = features.mean(dim=1)
        else:
            weights = valid_mask.unsqueeze(-1).to(features.dtype)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return self.final_layer(pooled)


class _CausalConv1d(nn.Module):
    """Keras-compatible causal convolution used by the pinned TCCNet source."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the source Glorot initialization and zero bias."""
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(1), 0)
        left_padding = (self.kernel_size - 1) * self.dilation
        convolved = F.conv1d(
            F.pad(x, (left_padding, 0)),
            self.weight,
            dilation=self.dilation,
        )
        # The source CausalConv1D only adds its bias where the bias-free
        # convolution is non-zero, preserving fully masked time steps.
        nonzero = convolved != 0
        return (convolved + self.bias.view(1, -1, 1)) * nonzero


class _TCCResidualBlock(nn.Module):
    """Faithful TCCNet residual block whose dilation continues across blocks."""

    def __init__(
        self,
        in_channels: int,
        filters: Sequence[int],
        kernel_size: int,
        dilations: Sequence[int],
        drop_prob: float,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.projection = _CausalConv1d(in_channels, filters[-1], kernel_size=1)
        self.convolutions = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for out_channels, dilation in zip(filters, dilations):
            self.convolutions.append(
                _CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            self.activations.append(activation())
            self.dropouts.append(nn.Dropout1d(drop_prob))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        residual = self.projection(x, valid_mask)
        for convolution, activation, dropout in zip(
            self.convolutions,
            self.activations,
            self.dropouts,
            strict=True,
        ):
            x = dropout(activation(convolution(x, valid_mask)))
        return x + residual


class _TCCContextAttention(nn.Module):
    """TCCNet's single-context attention with the source parameter layout."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_features))
        self.context = nn.Parameter(torch.empty(n_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the initializers used by the pinned Keras source."""
        nn.init.xavier_uniform_(self.weight)
        bound = math.sqrt(3.0 / self.context.numel())
        nn.init.uniform_(self.context, -bound, bound)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        scores = torch.tanh(x @ self.weight) @ self.context
        attention = torch.exp(scores)
        if valid_mask is not None:
            attention = attention * valid_mask.to(attention.dtype)
        # ``keras.backend.epsilon()`` is 1e-7 in the pinned source.
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-7)
        return torch.sum(x * attention.unsqueeze(-1), dim=1)

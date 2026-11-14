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
from torch import nn

from braindecode.functional import masked_mean
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d


class TCCNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""TCCNet from Tsinganos et al. (2019) [tccnet2019]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`
    :bdg-info:`Attention/Transformer`

    A temporal convolutional network that classifies a whole sEMG gesture.
    Residual blocks of causal convolutions, whose dilation doubles at every
    layer and keeps doubling across blocks, map the gesture to a sequence of
    features. An average over time (``"aot"``) or a context-vector attention
    (``"att"``) pools that sequence into one vector for a linear classifier.

    The paper compares four models that differ in receptive field (RF) and
    head (poster Table 1; parameters counted for 10 channels and 53 classes):

    ============  ==================================  ======  ==========
    Model         ``filters``                         Convs   Parameters
    ============  ==================================  ======  ==========
    AoT, 300 ms   ``((32, 64), (64, 128))``           4       60,117
    AoT, 2500 ms  ``((32, 32), (32, 32), (64, 64),    7       72,245
                  (128,))``
    Att, 300 ms   ``((32, 64), (64, 128))``           4       76,629
    Att, 2500 ms  ``((32, 32), (32, 32), (64, 64),    7       88,757
                  (128,))``
    ============  ==================================  ======  ==========

    The defaults build Att, 2500 ms, the best model of the paper (89.76 %
    top-1). Pass ``classification_head="aot"`` and/or
    ``filters=((32, 64), (64, 128))`` for the other three. With
    ``kernel_size=3``, the RF is 31 samples (310 ms at 100 Hz) for the
    4-convolution models and 255 samples (2550 ms) for the 7-convolution ones.

    Inputs are ``(batch, n_chans, n_times)`` and may have any length. The
    softmax of the reference is left out, because braindecode models return
    logits.

    Parameters
    ----------
    filters : sequence of sequence of int, default=((32, 32), (32, 32), (64, 64), (128,))
        Output filters of each causal convolution, one inner sequence per
        residual block. It sets the number of blocks and of convolutions per
        block.
    kernel_size : int, default=3
        Temporal kernel size of every causal convolution.
    drop_prob : float, default=0.05
        Channel-wise dropout after each activated convolution, as Keras
        ``SpatialDropout1D``.
    activation : type[nn.Module], default=nn.ReLU
        Activation after each causal convolution.
    classification_head : {"att", "aot"}, default="att"
        ``"att"``: context-vector attention; ``"aot"``: average over the
        valid time steps.
    mask_value : float | None, default=-10.0
        A time step where every channel equals this value is padding: it is
        zeroed before every convolution and left out of the pooling, as the
        reference's Keras ``Masking`` layer does. ``None`` turns masking off.

    Notes
    -----
    Protocol of the reference code [tccnetcode]_, for all four models: one
    model per subject of NinaPro DB1 (27 subjects, 10 sEMG channels at
    100 Hz, 53 classes including rest). Each channel is rectified and
    low-pass filtered by a 1 Hz first-order Butterworth filter applied
    forward-backward (``filtfilt``). Each whole gesture repetition is one
    sample: the variable-length gestures of a batch are pre-padded with
    ``-10`` to the longest one and masked. Repetitions 1, 3, 4, 6, 8, 9
    and 10 train; 2, 5 and 7 test. Training uses Adam (learning rate 0.01,
    epsilon 0.001) for 30 epochs, batch size 128, L2 0.0005 on convolution
    and classifier kernels (use optimizer weight decay in PyTorch), class-
    balanced sample weights, and 10 augmented copies of each training
    gesture (time and magnitude warping 0.2, noise at 25 dB SNR). The
    generator's min-max normalisation is off in the released configurations.

    The masked average over time equals the reference only for pre-padding,
    the reference pipeline: the reference sums every time step, padded ones
    included, and divides by the number of valid steps, while this port
    always leaves padded steps out; on a fully padded input it gives NaN, as
    the reference does. The bias of each convolution is added only where the
    convolution output is non-zero, as in the reference ``CausalConv1D``;
    this only matters for bit-exact agreement with it. The attention
    subtracts the row maximum before the exponential: the same values as
    the reference, which overflows to NaN once the context vector grows
    large. Initialisation follows Keras 2.2.4: truncated Glorot normal
    kernels, zero biases, Glorot uniform attention weights.

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
        filters: Sequence[Sequence[int]] = ((32, 32), (32, 32), (64, 64), (128,)),
        kernel_size: int = 3,
        drop_prob: float = 0.05,
        activation: type[nn.Module] = nn.ReLU,
        classification_head: str = "att",
        mask_value: float | None = -10.0,
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
        """Keras 2.2.4 initialisation of the reference (see Notes)."""
        for module in self.modules():
            if isinstance(module, (CausalConv1d, nn.Linear)):
                _keras_glorot_normal_(module.weight)
            if isinstance(module, (_MaskedCausalConv1d, nn.Linear)):
                nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classifier for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        _keras_glorot_normal_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of sEMG gestures."""
        valid_mask = None
        if self.mask_value is not None:
            # (batch, n_chans, n_times) -> (batch, n_times), True = real step
            valid_mask = torch.any(x != self.mask_value, dim=1)

        features = x
        for block in self.feature_extractor:
            # (batch, n_chans | filters, n_times) -> (batch, filters, n_times)
            features = block(features, valid_mask)
        # (batch, filters, n_times) -> (batch, n_times, filters)
        features = features.transpose(1, 2)

        if self.attention is not None:
            pooled = self.attention(features, valid_mask)
        else:
            pooled = masked_mean(features, valid_mask, [1])
        # (batch, filters) -> (batch, n_outputs)
        return self.final_layer(pooled)


def _keras_glorot_normal_(weight: torch.Tensor) -> None:
    """Keras 2.2.4 ``glorot_normal``: a normal truncated at 2 sigma, rescaled
    by 1 / 0.8796 (the std of a standard normal truncated at 2)."""
    receptive_field = weight[0][0].numel()
    fan_in = weight.shape[1] * receptive_field
    fan_out = weight.shape[0] * receptive_field
    std = math.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


class _MaskedCausalConv1d(nn.Module):
    """Reference ``CausalConv1D``: zero the padded input steps, then add the
    bias only where the bias-free causal convolution is non-zero."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=False
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(1), 0)
        # (batch, in_channels, n_times) -> (batch, out_channels, n_times)
        out = self.conv(x)
        return (out + self.bias.view(1, -1, 1)) * (out != 0)


class _TCCResidualBlock(nn.Module):
    """TCN residual block of the reference; private because braindecode.modules
    has no shared TCN residual block (ATCNet, TCFormer and TCN keep their own)."""

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
        # Kernel-size-1 projection of the block input onto the last filters.
        self.projection = _MaskedCausalConv1d(in_channels, filters[-1], 1, 1)
        self.convolutions = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for out_channels, dilation in zip(filters, dilations):
            self.convolutions.append(
                _MaskedCausalConv1d(in_channels, out_channels, kernel_size, dilation)
            )
            self.activations.append(activation())
            self.dropouts.append(nn.Dropout1d(drop_prob))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        # (batch, in_channels, n_times) -> (batch, filters[-1], n_times)
        residual = self.projection(x, valid_mask)
        for convolution, activation, dropout in zip(
            self.convolutions, self.activations, self.dropouts
        ):
            x = dropout(activation(convolution(x, valid_mask)))
        return x + residual


class _TCCContextAttention(nn.Module):
    """``AttentionWithContext(bias=False)`` of the reference; private because
    braindecode.modules has no context-vector attention pooling."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_features))
        self.context = nn.Parameter(torch.empty(n_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Keras 2.2.4 ``glorot_uniform``; for the 1-D context vector Keras
        takes fan_in = fan_out = sqrt(n_features)."""
        nn.init.xavier_uniform_(self.weight)
        bound = math.sqrt(3.0 / math.sqrt(self.context.numel()))
        nn.init.uniform_(self.context, -bound, bound)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        # (batch, n_times, n_features) -> (batch, n_times)
        scores = torch.tanh(x @ self.weight) @ self.context
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        # exp(s - m) / (sum exp(s - m) + eps * exp(-m)) is the reference's
        # exp(s) / (sum exp(s) + eps) without its overflow; eps = 1e-7 is
        # keras.backend.epsilon(). m >= 0 also covers fully padded rows.
        shift = scores.amax(dim=1, keepdim=True).clamp_min(0.0).detach()
        attention = torch.exp(scores - shift)
        attention = attention / (
            attention.sum(dim=1, keepdim=True) + 1e-7 * torch.exp(-shift)
        )
        # (batch, n_times, n_features) -> (batch, n_features)
        return torch.sum(x * attention.unsqueeze(-1), dim=1)

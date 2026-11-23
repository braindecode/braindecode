# Authors: Panagiotis Tsinganos et al. (original implementation)
#          Arslan Shafique <81501094+arslan-shafique@users.noreply.github.com>
#
# License: MIT
# Adapted from https://github.com/DSIP-UPatras/ICASSP2019_TCN (MIT).
# Its MeanOverTime and AttentionWithContext layers come from
# https://github.com/cbaziotis/keras-utilities (MIT).
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

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/tccnet.png
       :align: center
       :alt: TCCNet architecture

       "Proposed model" panel of the authors' ICASSP 2019 poster (MIT, from
       [tccnetcode]_): dilated causal convolutions, read out by an average
       over time (AoT) or an attention (Att) head. Braindecode returns the
       logits, without the softmax.

    A temporal convolutional network that classifies a whole sEMG gesture.
    The reference trains one model per subject of NinaPro DB1 (27 subjects,
    10 sEMG channels at 100 Hz, 53 classes including rest).

    .. rubric:: Architecture Overview

    Residual blocks of causal convolutions map the gesture to a sequence of
    features, and a pooling over time reduces it to one vector for a linear
    classifier. Inputs may have any length::

        (batch, n_chans, n_times) -> residual blocks
        -> (batch, filters[-1][-1], n_times) -> pooling over the valid steps
        -> (batch, filters[-1][-1]) -> final_layer -> (batch, n_outputs)

    The defaults build the best model of the paper: the attention head and a
    receptive field of 255 samples (2550 ms at 100 Hz). Pass
    ``classification_head="aot"`` and/or ``filters=((32, 64), (64, 128))``
    (31 samples, 310 ms) for its other three models. With 10 channels and 53
    classes the four models have 60,117 (AoT, 310 ms), 72,245 (AoT, 2550 ms),
    76,629 (Att, 310 ms) and 88,757 (Att, 2550 ms) parameters; the authors'
    poster lists them as 60K, 70K, 75K and 85K, and names the receptive fields
    300 and 2500 ms.

    .. rubric:: Macro Components

    ``TCCNet.feature_extractor``
        **Operations.** One residual block per inner sequence of ``filters``:
        its causal convolutions, each followed by the activation and a
        channel-wise dropout, plus a kernel-size-1 projection of the block
        input that is added to the last convolution. The dilation grows by the
        factor ``dilation_growth`` at every convolution, across blocks too; it
        doubles by default, so ``n`` convolutions of kernel size ``k`` see
        :math:`1 + (k - 1)(2^n - 1)` samples. **Role.** One feature vector per
        time step, computed from the past only.

    ``TCCNet.attention``
        **Operations.** With ``classification_head="att"``: a score
        ``tanh(h W) u`` for each time step ``h``, a softmax of the scores over
        the valid steps, and the sum of the steps weighted by it. It is
        ``None`` with ``"aot"``, which averages the valid steps instead.
        **Role.** Pools the sequence into one vector per gesture.

    ``TCCNet.final_layer``
        **Operations.** One :class:`~torch.nn.Linear` on the pooled vector.
        **Role.** The logits, without the softmax of the reference.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** causal convolutions with dilations 1, 2, 4, ... by default:
      each step sees only the past, and the receptive field grows exponentially.
    - **Spatial:** every convolution mixes all its input channels; there is no
      per-channel branch.
    - **Spectral:** not modelled. The reference rectifies each channel and
      low-pass filters it, so the network reads an amplitude envelope.

    .. rubric:: Additional Mechanisms

    - **Masking.** A time step where every channel equals ``mask_value`` is
      padding: every convolution reads it as zero and the pooling leaves it
      out, as the reference's Keras ``Masking`` layer does. As in its
      ``CausalConv1D``, a convolution adds its bias only where its output is
      non-zero.
    - **Attention.** The softmax over the scores ``s`` is the reference's
      ``exp(s) / (sum(exp(s)) + eps)``, computed as
      ``exp(s - m) / (sum(exp(s - m)) + eps * exp(-m))`` with ``m`` the row
      maximum, at least 0 so that a fully padded gesture stays finite. The
      values are the same, without the reference's overflow to NaN once the
      context vector grows large.
    - **Reference parity.** The reference sums every time step, padded ones
      included, and divides by the number of valid steps; this port leaves
      padded steps out. The two agree for pre-padding with an activation that
      maps 0 to 0, such as the reference's ReLU. The bias rule changes the
      result only where a convolution gives exactly zero at a valid step.
      Initialisation follows Keras 2.2.4: truncated Glorot normal kernels,
      zero biases, Glorot uniform attention weights.

    .. important::

        The reference code [tccnetcode]_ rectifies each channel, low-passes it
        at 1 Hz (first-order Butterworth, ``filtfilt``) and pre-pads the
        gestures of a batch with ``-10``. The masked average over time equals
        its average for that padding only; a fully padded gesture gives NaN
        with ``"aot"``, as there. The min-max normalisation of the reference's
        generator is off in its released configurations.

    `License <https://github.com/DSIP-UPatras/ICASSP2019_TCN/blob/cd28909ccf3696b835446cbec2f15eb60610a2af/LICENSE>`_

    Parameters
    ----------
    filters : sequence of sequence of int, default=((32, 32), (32, 32), (64, 64), (128,))
        Output filters of each causal convolution, one inner sequence per
        residual block. It sets the number of blocks and of convolutions per
        block.
    kernel_size : int, default=3
        Temporal kernel size of every causal convolution.
    dilation_growth : int, default=2
        Factor by which the dilation grows at every causal convolution, across
        blocks too: 1, 2, 4, ... for 2.
    drop_prob : float, default=0.05
        Channel-wise dropout after each activated convolution, as Keras
        ``SpatialDropout1D``.
    activation : type[nn.Module], default=nn.ReLU
        Activation after each causal convolution.
    classification_head : {"att", "aot"}, default="att"
        ``"att"``: context-vector attention; ``"aot"``: average over the
        valid time steps.
    mask_value : float | None, default=-10.0
        Value of every channel at a padded time step. ``None`` turns masking
        off.
    eps : float, default=1e-7
        Epsilon of the attention normalisation (``keras.backend.epsilon()`` in
        the reference). Not used with ``classification_head="aot"``.

    References
    ----------
    .. [tccnet2019] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., &
       Skodras, A. (2019). Improved Gesture Recognition Based on sEMG Signals
       and TCN. ICASSP 2019, 1169-1173.
       https://doi.org/10.1109/ICASSP.2019.8683239
    .. [tccnetcode] Tsinganos, P. et al. ICASSP2019_TCN source code.
       https://github.com/DSIP-UPatras/ICASSP2019_TCN/tree/cd28909ccf3696b835446cbec2f15eb60610a2af
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: feature_extractor.0.convolutions.0.conv.weight

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
        filters: Sequence[Sequence[int]] = ((32, 32), (32, 32), (64, 64), (128,)),
        kernel_size: int = 3,
        dilation_growth: int = 2,
        drop_prob: float = 0.05,
        activation: type[nn.Module] = nn.ReLU,
        classification_head: str = "att",
        mask_value: float | None = -10.0,
        eps: float = 1e-7,
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
            # The dilation grows by dilation_growth at every convolution, across
            # blocks too.
            dilations = [
                dilation * dilation_growth**i for i in range(len(block_filters))
            ]
            dilation *= dilation_growth ** len(block_filters)
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
            _TCCContextAttention(in_channels, eps)
            if classification_head == "att"
            else None
        )
        self.final_layer = nn.Linear(in_channels, self.n_outputs)
        self._min_n_times = 1
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Keras 2.2.4 initialisation of the reference: truncated Glorot normal
        kernels and zero biases (the attention has its own, in ``reset_parameters``)."""
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
        """Return logits for a batch of sEMG gestures.

        Parameters
        ----------
        x : torch.Tensor
            sEMG of shape ``(batch, n_chans, n_times)``, of any length. Padded
            steps hold ``mask_value`` in every channel.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_outputs)``.
        """
        valid_mask = None
        if self.mask_value is not None:
            # True at the real steps, padding holds mask_value in every channel:
            # (batch, n_chans, n_times) -> (batch, n_times)
            valid_mask = torch.any(x != self.mask_value, dim=1)

        features = x
        for block in self.feature_extractor:
            # causal convolutions with a residual connection:
            # (batch, n_chans | filters, n_times) -> (batch, filters, n_times)
            features = block(features, valid_mask)
        # time steps first, as the pooling reads them: (batch, n_times, filters)
        features = features.transpose(1, 2)

        # one vector per gesture from its valid steps: (batch, filters)
        if self.attention is not None:
            pooled = self.attention(features, valid_mask)
        else:
            pooled = masked_mean(features, valid_mask, [1])
        # (batch, filters) -> (batch, n_outputs)
        return self.final_layer(pooled)


def _keras_glorot_normal_(weight: torch.Tensor) -> None:
    """Keras 2.2.4 ``glorot_normal``: a normal truncated at 2 sigma, rescaled
    by 1 / the std of a standard normal truncated at +-2 sigma."""
    # Std of a standard normal truncated at +-2 sigma, which Keras 2.2.4's
    # ``glorot_normal`` divides by so that the truncated draw has the Glorot std.
    trunc_normal_2sigma_std = 0.87962566103423978
    receptive_field = weight[0][0].numel()
    fan_in = weight.shape[1] * receptive_field
    fan_out = weight.shape[0] * receptive_field
    std = math.sqrt(2.0 / (fan_in + fan_out)) / trunc_normal_2sigma_std
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


class _MaskedCausalConv1d(nn.Module):
    """Reference ``CausalConv1D``.

    Steps: zero the padded input steps, apply a bias-free causal convolution,
    then add the bias only where that output is non-zero. ``x`` is
    ``(batch, in_channels, n_times)`` and ``valid_mask`` ``(batch, n_times)`` or
    ``None``; the output is ``(batch, out_channels, n_times)``.
    """

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
            # padded steps count as zeros: (batch, in_channels, n_times)
            x = x.masked_fill(~valid_mask.unsqueeze(1), 0)
        # (batch, in_channels, n_times) -> (batch, out_channels, n_times)
        out = self.conv(x)
        # the bias only where the convolution is non-zero (the reference's
        # ``CausalConv1D``): (batch, out_channels, n_times)
        return (out + self.bias.view(1, -1, 1)) * (out != 0)


class _TCCResidualBlock(nn.Module):
    """TCN residual block of the reference; private because braindecode.modules
    has no shared TCN residual block (ATCNet, TCFormer and TCN keep their own).

    Steps: a kernel-size-1 projection of the input to ``filters[-1]`` channels;
    then, for each entry of ``filters``, a causal convolution with its dilation,
    the activation and a channel-wise dropout; the projection is added to the
    last output. ``x`` is ``(batch, in_channels, n_times)`` and ``valid_mask``
    ``(batch, n_times)`` or ``None``; the output is
    ``(batch, filters[-1], n_times)``.
    """

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
        # shortcut at the width of the last convolution:
        # (batch, in_channels, n_times) -> (batch, filters[-1], n_times)
        residual = self.projection(x, valid_mask)
        for convolution, activation, dropout in zip(
            self.convolutions, self.activations, self.dropouts
        ):
            # dilated causal convolution, activation, channel-wise dropout:
            # (batch, channels, n_times) -> (batch, out_channels, n_times)
            x = dropout(activation(convolution(x, valid_mask)))
        # add the shortcut: (batch, filters[-1], n_times)
        return x + residual


class _TCCContextAttention(nn.Module):
    """``AttentionWithContext(bias=False)`` of the reference; private because
    braindecode.modules has no context-vector attention pooling.

    Steps: a score ``tanh(x W) u`` per time step, a softmax of the scores over
    the valid steps (with the epsilon of Keras, see ``forward``), and the sum of
    the steps weighted by it. ``x`` is ``(batch, n_times, n_features)`` and
    ``valid_mask`` ``(batch, n_times)`` or ``None``; the output is
    ``(batch, n_features)``.
    """

    def __init__(self, n_features: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
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
        # one score per time step: (batch, n_times, n_features) -> (batch, n_times)
        scores = torch.tanh(x @ self.weight) @ self.context
        if valid_mask is not None:
            # padded steps get no weight: (batch, n_times)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        # exp(s - m) / (sum exp(s - m) + eps * exp(-m)) is the reference's
        # exp(s) / (sum exp(s) + eps) without its overflow; eps is
        # keras.backend.epsilon(), 1e-7. m >= 0 also covers fully padded rows.
        shift = scores.amax(dim=1, keepdim=True).clamp_min(0.0).detach()
        attention = torch.exp(scores - shift)
        # weights that sum to 1 over the valid steps: (batch, n_times)
        attention = attention / (
            attention.sum(dim=1, keepdim=True) + self.eps * torch.exp(-shift)
        )
        # weighted sum over time: (batch, n_times, n_features) -> (batch, n_features)
        return torch.sum(x * attention.unsqueeze(-1), dim=1)

# Authors: Kerui Wu <wuk9@rpi.edu>
#          Ziyue Zhao <zhaoz10@rpi.edu>
#          Bülent Yener <yener@cs.rpi.edu>
#          Raghav Rathi <44058192+raghav-rathi@users.noreply.github.com> (braindecode implementation)
#
# License: MIT
# Reference implementation: https://github.com/keruiwu/SeizureTransformer (MIT).

from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.functional import sinusoidal_positional_encoding
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one


class SeizureTransformer(EEGModuleMixin, nn.Module, license="mit"):
    r"""SeizureTransformer from Wu et al. (2025) [Wu2025]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/seizure_transformer_arch.png
       :align: center
       :alt: SeizureTransformer architecture
       :width: 1000px

       SeizureTransformer with its default configuration, reproduced from
       [Wu2025]_ (MIT).

    .. rubric:: Architecture Overview

    SeizureTransformer labels every time sample of an EEG window instead of the
    window as a whole, so seizure onsets and offsets come straight out of the
    network without sliding-window inference [Wu2025]_. It won the 2025
    seizure detection challenge, scored with the SzCORE framework [Dan2025]_.
    The network is U-shaped:

    1. A convolutional encoder halves the time axis at each level while it
       widens the features. With the defaults, a 60 s window at 256 Hz becomes
       480 time steps of 512 features.
    2. Residual convolution blocks refine these features.
    3. Sinusoidal positions are added and a Transformer encoder relates every
       time step to the whole window. Its output is added back to the
       residual-block output.
    4. A convolutional decoder upsamples back to the input length and adds the
       encoder feature map of the same resolution at every level.
    5. A final convolution gives ``n_outputs`` logits for every time sample.

    .. rubric:: Macro Components

    ``SeizureTransformer.encoder``
        **Operations.** At each level, a "same"-padded
        :class:`~torch.nn.Conv1d` with kernel ``encoder_kernel_sizes[i]`` and
        ``n_filters[i]`` output channels, followed by ``activation``. The
        result is kept for the decoder, then ``SeizureTransformer.pool``
        halves the time axis (max pooling, rounding odd lengths up).

        **Role.** Down-sample the long input and extract local features at
        increasingly coarse time scales.

    ``SeizureTransformer.res_blocks``
        **Operations.** Pre-activation residual blocks, one per entry of
        ``res_kernel_sizes``: twice batch normalisation, ``activation_res``,
        channel dropout (:class:`~torch.nn.Dropout1d`) and a "same"-padded
        convolution, plus an identity shortcut.

        **Role.** Refine the down-sampled features with small kernels before
        global attention, as in EQTransformer [Mousavi2020]_.

    ``SeizureTransformer.transformer``
        **Operations.** ``num_layers`` post-norm
        :class:`~torch.nn.TransformerEncoderLayer` blocks with ``num_heads``
        heads and a ``dim_feedforward`` feed-forward width, applied after fixed
        sinusoidal positions and dropout.

        **Role.** Capture dependencies across the whole window and give the
        model most of its capacity: about 25 of its 38 million parameters with
        the defaults.

    ``SeizureTransformer.decoder``
        **Operations.** At each level, ``SeizureTransformer.upsample`` doubles
        the time axis (nearest neighbour), the result is cropped to the length
        of the matching encoder level, then a "same"-padded convolution with
        kernel ``decoder_kernel_sizes[i]`` and ``activation`` is applied and
        the encoder feature map is added.

        **Role.** Restore the input resolution for per-sample predictions.

    ``SeizureTransformer.final_layer``
        **Operations.** A :class:`~torch.nn.Conv1d` with ``final_kernel_size``
        taps from ``n_filters[0]`` channels to ``n_outputs``.

        **Role.** Map each time sample to its logits. The output has shape
        ``(batch, n_outputs, n_times)``.

    .. rubric:: Usage

    The model returns logits. With ``n_outputs=1``, :func:`torch.sigmoid`
    gives the seizure probability of each sample, and the paper trains with
    binary cross-entropy [Wu2025]_. The authors' competition model uses the
    defaults and expects this input, which the model does not prepare itself:

    - 19 channels in the order Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Cz, Pz,
      Fp2, F4, C4, P4, O2, F8, T4, T6 (average reference, as in the SzCORE
      BIDS datasets such as :class:`~braindecode.datasets.SIENA`);
    - each channel z-scored over the whole recording, then resampled to
      256 Hz;
    - non-overlapping 60 s windows (``n_times=15360``), the last one padded
      with zeros;
    - in each window, a causal third-order Butterworth band-pass from 0.5 to
      120 Hz, then IIR notch filters at 1 Hz and 60 Hz (quality factor 30).

    The authors turn the probabilities into seizure events with a 0.8
    threshold, a binary opening then a binary closing with a 5-sample
    structuring element, and the removal of events shorter than 2 s.

    .. rubric:: Differences from the Reference Implementation

    - The output is logits with a channel axis, ``(batch, n_outputs,
      n_times)``, instead of sigmoid probabilities of shape
      ``(batch, n_times)``.
    - A single ``drop_prob`` sets the dropout of the residual blocks, the
      positions and the Transformer (all 0.1 in the reference).
    - Odd lengths are handled by rounding the pooled length up and cropping in
      the decoder, so any input of at most ``n_times`` samples gives one
      prediction per sample. The positional table is computed instead of
      stored. The reference checkpoint also stores an unused copy of the
      Transformer layer that is not part of this model.

    .. versionadded:: 1.8.2

    Parameters
    ----------
    n_filters : tuple of int, default=(32, 64, 128, 256, 512)
        Output channels of the encoder levels. The decoder mirrors them, and
        the last value is the Transformer width.
    encoder_kernel_sizes : tuple of int, default=(11, 9, 7, 7, 5)
        Kernel size of the convolution at each encoder level, from the input
        down.
    decoder_kernel_sizes : tuple of int, default=(3, 5, 5, 7, 7)
        Kernel size of the convolution at each decoder level, from the deepest
        level up.
    res_kernel_sizes : tuple of int, default=(3, 3, 3, 3, 2, 3, 2)
        Kernel sizes of the residual blocks, one block per entry.
    final_kernel_size : int, default=11
        Kernel size of the output convolution.
    num_layers : int, default=8
        Number of Transformer encoder layers.
    num_heads : int, default=4
        Number of attention heads. Must divide ``n_filters[-1]``.
    dim_feedforward : int, default=2048
        Hidden width of the Transformer feed-forward blocks.
    drop_prob : float, default=0.1
        Dropout probability of the residual blocks, the positional encoding
        and the Transformer layers.
    activation : type[nn.Module], default=nn.ELU
        Non-linearity after the encoder and decoder convolutions.
    activation_res : type[nn.Module], default=nn.ReLU
        Non-linearity inside the residual blocks.

    References
    ----------
    .. [Wu2025] Wu, K., Zhao, Z., & Yener, B. (2025). Large EEG-U-Transformer
       for time-step level detection without pre-training. arXiv preprint
       arXiv:2504.00336. Code: https://github.com/keruiwu/SeizureTransformer
       (MIT).
    .. [Dan2025] Dan, J., Pale, U., Amirshahi, A., Cappelletti, W., Ingolfsson,
       T. M., Wang, X., Cossettini, A., Bernini, A., Benini, L., Beniczky, S.,
       Atienza, D., & Ryvlin, P. (2025). SzCORE: Seizure Community Open-Source
       Research Evaluation framework for the validation of
       electroencephalography-based automated seizure detection algorithms.
       Epilepsia, 66(S3), 14-24. https://doi.org/10.1111/epi.18113
    .. [Mousavi2020] Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y.,
       & Beroza, G. C. (2020). Earthquake transformer: an attentive
       deep-learning model for simultaneous earthquake detection and phase
       picking. Nature Communications, 11, 3952.
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # model-specific parameters
        n_filters: tuple[int, ...] = (32, 64, 128, 256, 512),
        encoder_kernel_sizes: tuple[int, ...] = (11, 9, 7, 7, 5),
        decoder_kernel_sizes: tuple[int, ...] = (3, 5, 5, 7, 7),
        res_kernel_sizes: tuple[int, ...] = (3, 3, 3, 3, 2, 3, 2),
        final_kernel_size: int = 11,
        num_layers: int = 8,
        num_heads: int = 4,
        dim_feedforward: int = 2048,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.ELU,
        activation_res: type[nn.Module] = nn.ReLU,
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

        if not len(n_filters) == len(encoder_kernel_sizes) == len(decoder_kernel_sizes):
            raise ValueError(
                "n_filters, encoder_kernel_sizes and decoder_kernel_sizes must "
                f"have the same length, got {len(n_filters)}, "
                f"{len(encoder_kernel_sizes)} and {len(decoder_kernel_sizes)}."
            )
        embed_dim = n_filters[-1]
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide the Transformer width "
                f"n_filters[-1] ({embed_dim})."
            )

        # Longest input the positional table covers; shorter inputs also work.
        self.max_n_times = self.n_times
        n_steps = self.max_n_times
        for _ in n_filters:
            n_steps = (n_steps + 1) // 2

        self.encoder = nn.ModuleList(
            _ConvBlock(in_channels, out_channels, kernel_size, activation)
            for in_channels, out_channels, kernel_size in zip(
                (self.n_chans, *n_filters[:-1]), n_filters, encoder_kernel_sizes
            )
        )
        # ceil_mode keeps the last sample of an odd-length level.
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.res_blocks = nn.Sequential(
            *(
                _ResidualConvBlock(embed_dim, kernel_size, drop_prob, activation_res)
                for kernel_size in res_kernel_sizes
            )
        )
        self.to_steps = Rearrange("batch features time -> batch time features")
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(n_steps, embed_dim),
            persistent=False,
        )
        self.positional_dropout = nn.Dropout(drop_prob)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=drop_prob,
                batch_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.to_features = Rearrange("batch time features -> batch features time")
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        decoder_filters = n_filters[::-1]
        self.decoder = nn.ModuleList(
            _ConvBlock(in_channels, out_channels, kernel_size, activation)
            for in_channels, out_channels, kernel_size in zip(
                (embed_dim, *decoder_filters[:-1]),
                decoder_filters,
                decoder_kernel_sizes,
            )
        )
        self.final_layer = nn.Conv1d(
            n_filters[0], self.n_outputs, final_kernel_size, padding="same"
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the output convolution for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        old_head = self.final_layer
        head = nn.Conv1d(
            old_head.in_channels, n_outputs, old_head.kernel_size[0], padding="same"
        )
        self.final_layer = head.to(old_head.weight)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits for every time sample.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``. Inputs shorter than
            the configured ``n_times`` are accepted.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_outputs, n_times)``.
        """
        if x.shape[-1] > self.max_n_times:
            raise ValueError(
                f"SeizureTransformer was built for at most {self.max_n_times} "
                f"time samples (n_times), but received {x.shape[-1]}."
            )
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        features = self.res_blocks(x)
        steps = self.to_steps(features)
        steps = steps + self.positional_encoding[: steps.shape[1]]
        steps = self.transformer(self.positional_dropout(steps))
        x = self.to_features(steps) + features

        for level, block in enumerate(self.decoder):
            skip = skips[len(skips) - 1 - level]
            x = self.upsample(x)[..., : skip.shape[-1]]
            x = block(x) + skip
        return self.final_layer(x)


class _ConvBlock(nn.Module):
    """Length-preserving convolution followed by a non-linearity.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size. An even size is padded on the right.
    activation : type[nn.Module]
        Non-linearity applied after the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.pad = _same_padding(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(self.pad(x)))


class _ResidualConvBlock(nn.Module):
    """Pre-activation residual block of two length-preserving convolutions.

    Parameters
    ----------
    n_channels : int
        Number of input and output channels.
    kernel_size : int
        Kernel size of both convolutions. An even size is padded on the right.
    drop_prob : float
        Probability of dropping a whole channel before each convolution.
    activation : type[nn.Module]
        Non-linearity applied after each batch normalisation.
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.pad = _same_padding(kernel_size)
        self.norm1 = nn.BatchNorm1d(n_channels, eps=1e-3)
        self.activation1 = activation()
        self.dropout1 = nn.Dropout1d(drop_prob)
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size)
        self.norm2 = nn.BatchNorm1d(n_channels, eps=1e-3)
        self.activation2 = activation()
        self.dropout2 = nn.Dropout1d(drop_prob)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.activation1(self.norm1(x)))
        out = self.conv1(self.pad(out))
        out = self.dropout2(self.activation2(self.norm2(out)))
        out = self.conv2(self.pad(out))
        return x + out


def _same_padding(kernel_size: int) -> nn.ConstantPad1d:
    """Zero padding that keeps the length through a stride-1 convolution.

    An even kernel gets its extra zero on the right, as in TensorFlow's
    ``"same"`` padding used by the reference implementation.
    """
    return nn.ConstantPad1d(((kernel_size - 1) // 2, kernel_size // 2), 0.0)

# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)


import numpy as np
import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin


class USleep(EEGModuleMixin, nn.Module):
    """
    Sleep staging architecture from Perslev et al. (2021) [1]_.

    :bdg-success:`Convolution`

    .. figure:: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-021-00440-5/MediaObjects/41746_2021_440_Fig2_HTML.png
        :align: center
        :alt: USleep Architecture

        Figure: U-Sleep consists of an encoder (left) which encodes the input signals into dense feature representations, a decoder (middle) which projects
        the learned features into the input space to generate a dense sleep stage representation, and finally a specially designed segment
        classifier (right) which generates sleep stages at a chosen temporal resolution.

    .. rubric:: Architectural Overview

    U-Sleep is a **fully convolutional**, feed-forward encoder-decoder with a *segment classifier* head for
    time-series **segmentation** (sleep staging). It maps multi-channel PSG (EEG+EOG) to a *dense, high-frequency*
    per-sample representation, then aggregates it into fixed-length stage labels (e.g., 30 s). The network
    processes arbitrarily long inputs in **one forward pass** (resampling to 128 Hz), allowing whole-night
    hypnograms in seconds.

    - (i). :class:`_EncoderBlock` extracts progressively deeper temporal features at lower resolution;
    - (ii). :class:`_Decoder` upsamples and fuses encoder features via U-Net-style skips to recover a per-sample stage map;
    - (iii). Segment Classifier mean-pools over the target epoch length and applies two pointwise convs to yield
      per-epoch probabilities. Integrates into the USleep class.

    .. rubric:: Macro Components

    - Encoder :class:`_EncoderBlock` **(multi-scale temporal feature extractor; downsampling x2 per block)**

        - *Operations.*
        - **Conv1d** (:class:`torch.nn.Conv1d`) with kernel ``9`` (stride ``1``, no dilation)
        - **ELU** (:class:`torch.nn.ELU`)
        - **Batch Norm** (:class:`torch.nn.BatchNorm1d`)
        - **Max Pool 1d**, :class:`torch.nn.MaxPool1d` (``kernel=2, stride=2``).

        Filters grow with depth by a factor of ``sqrt(2)`` (start ``c_1=5``); each block exposes a **skip**
        (pre-pooling activation) to the matching decoder block.
        *Role.* Slow, uniform downsampling preserves early information while expanding the effective temporal
        context over minutes—foundational for robust cross-cohort staging.

    The number of filters grows with depth (capacity scaling); each block also exposes a **skip** (pre-pool)
    to the matching decoder block.

    **Rationale.**
        - Slow, uniform downsampling (x2 each level) preserves information in early layers while expanding the temporal receptive field over the minutes.

    - Decoder :class:`_DecoderBlock`  **(progressive upsampling + skip fusion to high-frequency map, 12 blocks; upsampling x2 per block)**

        - *Operations.*
        - **Nearest-neighbor upsample**, :class:`nn.Upsample` (x2)
        - **Convolution2d** (k=2), :class:`torch.nn.Conv2d`
        - ELU, :class:`torch.nn.ELU`
        - Batch Norm, :class:`torch.nn.BatchNorm2d`
        - **Concatenate** with the encoder skip at the same temporal scale, :function:`torch.cat`
        - **Convolution**, :class:`torch.nn.Conv2d`
        - ELU, :class:`torch.nn.ELU`
        - Batch Norm, :class:`torch.nn.BatchNorm2d`.

    **Output**: A multi-class, **high-frequency** per-sample representation aligned to the input rate (128 Hz).

    - **Segment Classifier incorporate into :class:`braindecode.models.USleep` (aggregation to fixed epochs)**

        - *Operations.*
        - **Mean-pool**, :class:`torch.nn.AvgPool2d` per class with kernel = epoch length *i* and stride *i*
        - **1x1 conv**, :class:`torch.nn.Conv2d`
        - ELU, :class:`torch.nn.ELU`
        - **1x1 conv**, :class:`torch.nn.Conv2d` with ``(T, K)`` (epochs x stages).

    **Role**: Learns a **non-linear** weighted combination over each 30-s window (unlike U-Time's linear combiner).

    .. rubric:: Convolutional Details

    - **Temporal (where time-domain patterns are learned).**
    All convolutions are **1-D along time**; depth (12 levels) plus pooling yields an extensive receptive field
    (reported sensitivity to ±6.75 min around each epoch; theoretical field ≈ 9.6 min at the deepest layer).
    The decoder restores sample-level resolution before epoch aggregation.

    - **Spatial (how channels are processed).**
    Convolutions mix across the *channel* dimension jointly with time (no separate spatial operator). The system
    is **montage-agnostic** (any reasonable EEG/EOG pair) and was trained across diverse cohorts/protocols,
    supporting robustness to channel placement and hardware differences.

    - **Spectral (how frequency content is captured).**
    No explicit Fourier/wavelet transform is used; the **stack of temporal convolutions** acts as a learned
    filter bank whose effective bandwidth grows with depth. The high-frequency decoder output (128 Hz)
    retains fine temporal detail for the segment classifier.


    .. rubric:: Attention / Sequential Modules

    U-Sleep contains **no attention or recurrent units**; it is a *pure* feed-forward, fully convolutional
    segmentation network inspired by U-Net/U-Time, favoring training stability and cross-dataset portability.


    .. rubric:: Additional Mechanisms

    - **U-Net lineage with task-specific head.** U-Sleep extends U-Time by being **deeper** (12 vs. 4 levels),
      switching ReLU→**ELU**, using uniform pooling (2) at all depths, and replacing the linear combiner with a
      **two-layer** pointwise head—improving capacity and resilience across datasets.
    - **Arbitrary-length inference.** Thanks to full convolutionality and tiling-free design, entire nights can be
      staged in a single pass on commodity hardware. Inputs shorter than ≈ 17.5 min may reduce performance by
      limiting long-range context.
    - **Complexity scaling (alpha).** Filter counts can be adjusted by a global **complexity factor** to trade accuracy
      and memory (as described in the paper's topology table).


    .. rubric:: Usage and Configuration

    - **Practice.** Resample PSG to **128 Hz** and provide at least two channels (one EEG, one EOG). Choose epoch
      length *i* (often 30 s); ensure windows long enough to exploit the model's receptive field (e.g., training on
      ≥ 17.5 min chunks).


    Parameters
    ----------
    n_chans : int
        Number of EEG or EOG channels. Set to 2 in [1]_ (1 EEG, 1 EOG).
    sfreq : float
        EEG sampling frequency. Set to 128 in [1]_.
    depth : int
        Number of conv blocks in encoding layer (number of 2x2 max pools).
        Note: each block halves the spatial dimensions of the features.
    n_time_filters : int
        Initial number of convolutional filters. Set to 5 in [1]_.
    complexity_factor : float
        Multiplicative factor for the number of channels at each layer of the U-Net.
        Set to 2 in [1]_.
    with_skip_connection : bool
        If True, use skip connections in decoder blocks.
    n_outputs : int
        Number of outputs/classes. Set to 5.
    input_window_seconds : float
        Size of the input, in seconds. Set to 30 in [1]_.
    time_conv_size_s : float
        Size of the temporal convolution kernel, in seconds. Set to 9 / 128 in
        [1]_.
    ensure_odd_conv_size : bool
        If True and the size of the convolutional kernel is an even number, one
        will be added to it to ensure it is odd, so that the decoder blocks can
        work. This can be useful when using different sampling rates from 128
        or 100 Hz.
    activation : nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    References
    ----------
    .. [1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.
       U-Sleep: resilient high-frequency sleep staging. *npj Digit. Med.* 4, 72 (2021).
       https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """

    def __init__(
        self,
        n_chans=None,
        sfreq=None,
        depth=12,
        n_time_filters=5,
        complexity_factor=1.67,
        with_skip_connection=True,
        n_outputs=5,
        input_window_seconds=None,
        time_conv_size_s=9 / 128,
        ensure_odd_conv_size=False,
        activation: nn.Module = nn.ELU,
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

        self.mapping = {
            "clf.3.weight": "final_layer.0.weight",
            "clf.3.bias": "final_layer.0.bias",
            "clf.5.weight": "final_layer.2.weight",
            "clf.5.bias": "final_layer.2.bias",
        }

        max_pool_size = 2  # Hardcoded to avoid dimensional errors
        time_conv_size = int(np.round(time_conv_size_s * self.sfreq))
        if time_conv_size % 2 == 0:
            if ensure_odd_conv_size:
                time_conv_size += 1
            else:
                raise ValueError(
                    "time_conv_size must be an odd number to accommodate the "
                    "upsampling step in the decoder blocks."
                )

        channels = [self.n_chans]
        n_filters = n_time_filters
        for _ in range(depth + 1):
            channels.append(int(n_filters * np.sqrt(complexity_factor)))
            n_filters = int(n_filters * np.sqrt(2))
        self.channels = channels

        # Instantiate encoder
        self.encoder_blocks = nn.ModuleList(
            _EncoderBlock(
                in_channels=channels[idx],
                out_channels=channels[idx + 1],
                kernel_size=time_conv_size,
                downsample=max_pool_size,
                activation=activation,
            )
            for idx in range(depth)
        )

        # Instantiate bottom (channels increase, temporal dim stays the same)
        self.bottom = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=time_conv_size,
                padding=(time_conv_size - 1) // 2,
            ),  # preserves dimension
            activation(),
            nn.BatchNorm1d(num_features=channels[-1]),
        )

        # Instantiate decoder
        channels_reverse = channels[::-1]
        self.decoder_blocks = nn.ModuleList(
            _DecoderBlock(
                in_channels=channels_reverse[idx],
                out_channels=channels_reverse[idx + 1],
                kernel_size=time_conv_size,
                upsample=max_pool_size,
                with_skip_connection=with_skip_connection,
                activation=activation,
            )
            for idx in range(depth)
        )

        # The temporal dimension remains unchanged
        # (except through the AvgPooling which collapses it to 1)
        # The spatial dimension is preserved from the end of the UNet, and is mapped to n_classes

        self.clf = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # output is (B, C, 1, S * T)
            nn.Tanh(),
            nn.AvgPool1d(self.n_times),  # output is (B, C, S)
        )

        self.final_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=self.n_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # output is (B, n_classes, S)
            activation(),
            nn.Conv1d(
                in_channels=self.n_outputs,
                out_channels=self.n_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Identity(),
            # output is (B, n_classes, S)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """If input x has shape (B, S, C, T), return y_pred of shape (B, n_classes, S).
        If input x has shape (B, C, T), return y_pred of shape (B, n_classes).
        """
        # reshape input
        if x.ndim == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)

        # encoder
        residuals = []
        for down in self.encoder_blocks:
            x, res = down(x)
            residuals.append(res)

        # bottom
        x = self.bottom(x)

        # decoder
        num_blocks = len(self.decoder_blocks)  # statically known
        for idx, dec in enumerate(self.decoder_blocks):
            # pick the matching residual in reverse order
            res = residuals[num_blocks - 1 - idx]
            x = dec(x, res)

        # classifier
        x = self.clf(x)
        y_pred = self.final_layer(x)  # (B, n_classes, seq_length)

        if y_pred.shape[-1] == 1:  # seq_length of 1
            y_pred = y_pred[:, :, 0]

        return y_pred


class _EncoderBlock(nn.Module):
    """Encoding block for a timeseries x of shape (B, C, T)."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        kernel_size=9,
        downsample=2,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.block_prepool = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.BatchNorm1d(num_features=out_channels),
        )

        self.pad = nn.ConstantPad1d(padding=1, value=0.0)
        self.maxpool = nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = self.pad(x)
        x = self.maxpool(x)
        return x, residual


class _DecoderBlock(nn.Module):
    """Decoding block for a timeseries x of shape (B, C, T)."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        kernel_size=9,
        upsample=2,
        with_skip_connection=True,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                padding="same",
            ),
            activation(),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.block_postskip = nn.Sequential(
            nn.Conv1d(
                in_channels=(
                    2 * out_channels if with_skip_connection else out_channels
                ),
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = self._crop_tensors_to_match(
                x, residual, axis=-1
            )  # in case of mismatch
            x = torch.cat([x, residual], dim=1)  # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x

    @staticmethod
    def _crop_tensors_to_match(
        x1: torch.Tensor, x2: torch.Tensor, axis: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Crops two tensors to their lowest-common-dimension along an axis."""
        dim_cropped = min(x1.shape[axis], x2.shape[axis])

        x1_cropped = torch.index_select(
            x1, dim=axis, index=torch.arange(dim_cropped).to(device=x1.device)
        )
        x2_cropped = torch.index_select(
            x2, dim=axis, index=torch.arange(dim_cropped).to(device=x1.device)
        )
        return x1_cropped, x2_cropped

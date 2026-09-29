# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Yilin Liu, Shijia Zhang, Mahanth Gowda (original NeuroPose)
#
# License: BSD (3-clause)
# Reimplementation for research/benchmarking; original work is
# Liu et al., WWW 2021 / IEEE IoT-J 2022.
"""``NeuroPose``: CNN encoder-decoder with ResNet bottleneck."""

from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one


class NeuroPose(EEGModuleMixin, nn.Module):
    r"""NeuroPose from Liu et al (2021) [liu2021neuropose]_.

    :bdg-success:`Convolution`

    .. figure:: https://braindecode.org/dev/_static/model/neuropose.png
       :align: center
       :alt: NeuroPose encoder / ResNet / decoder architecture

       Figure 6 of Liu et al. (2021), CC BY. The three stages map onto
       :class:`NeuroPose`'s ``encoder``, ``resnet`` and ``decoder``.


    Convolutional encoder-decoder with a residual bottleneck mapping
    wearable sEMG windows to continuous finger joint angles. The defaults
    reproduce the configuration emg2pose [salter2024]_ used for its
    published NeuroPose baseline, which widens Liu et al.'s original
    pooling schedule for a device with 10x the sampling rate and 2x the
    spatial resolution.

    .. rubric:: Architecture Overview

    ``(B, n_chans, T)`` → add a feature axis, giving ``(B, 1, T, n_chans)``
    → three Conv2d-BN-ReLU-Dropout-MaxPool encoder stages downsampling
    (time x electrodes) by (10, 2), (8, 2), (4, 4) → ``n_res_blocks``
    residual blocks → three mirrored Conv-BN-ReLU-Dropout-Upsample decoder
    stages restoring the temporal axis → flatten the feature and electrode
    axes → linear head → ``(B, T, n_outputs)``.

    .. rubric:: Macro Components

    ``NeuroPose.encoder``
        **Operations.** Three Conv2d-BN-ReLU-Dropout-MaxPool blocks with
        widths ``encoder_channels`` and ``padding="same"`` 3x2 kernels,
        pooling (time x electrodes) by ``encoder_pool_sizes``.
        **Role.** Compresses the window along time while mixing adjacent
        electrodes, until the electrode axis is a single position.

    ``NeuroPose.resnet`` (residual bottleneck)
        **Operations.** ``n_res_blocks`` blocks computing ``x + f(x)``,
        where ``f`` is ``n_convs_per_block`` Conv-BN-ReLU-Dropout groups at
        constant width. No activation follows the sum.
        **Role.** The paper's key accuracy lever: deeper feature
        extraction without convergence loss ({3, 5, 7} swept upstream).

    ``NeuroPose.decoder`` / ``NeuroPose.final_layer``
        **Operations.** Three Conv-BN-ReLU-Dropout-Upsample(nearest)
        stages invert the encoder's pooling; ``final_layer`` maps the
        flattened feature/electrode axes to ``n_outputs`` per frame.
        **Role.** Dense per-frame joint-angle regression.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** The pooling/upsampling pyramid captures temporal structure.
    - **Channels/space:** Small 2-D kernels mix neighboring electrodes.
    - **Frequency:** Spectral content is learned from the raw waveform;
      unlike a decimating front end, no band is discarded up front.

    .. rubric:: Additional Mechanisms

    The paper's anatomical constraints (bounded activations over joint
    ranges), group-wise MSE losses and the temporal-smoothness penalty
    are training-side concerns outside this module; its BN-freezing
    transfer-learning recipe applies unchanged (freeze all but BatchNorm
    layers, fine-tune on ~90 s of target-user data).

    .. rubric:: Relationship to emg2pose's NeuroPose baseline

    With default arguments this class is operation-equivalent to
    emg2pose's ``network/neuropose.yaml``: the same block widths, kernels,
    pooling and upsampling schedule, the same residual depth, and the same
    6,354,903 parameters at ``n_chans=16, n_outputs=20``.

    One deviation remains, and it is a superset rather than a difference:
    upstream requires the window length to be divisible by the total
    temporal pooling factor (320 by default), having no way to recover the
    samples that flooring discards. This class interpolates the decoded
    sequence back to ``T``, which is exactly the identity when ``T`` is
    divisible by that factor and well-defined when it is not.

    .. rubric:: Pre-trained weights

    Meta's released ``regression_neuropose.ckpt`` is rehosted with these
    parameter names, so no conversion step is needed::

        NeuroPose.from_pretrained("braindecode/NeuroPose-emg2pose")

    The rehosted weights reproduce the reference implementation exactly.
    They stay under emg2pose's CC BY-NC-SA 4.0 license; only the code here is
    BSD-3.

    Liu et al.'s original 200 Hz Myo configuration remains reachable by
    passing the original schedule::

        NeuroPose(
            n_chans=8, n_outputs=16, n_times=1_000, sfreq=200.0,
            encoder_pool_sizes=((5, 2), (4, 2), (2, 2)),
            decoder_upsample_sizes=((5, 4), (4, 2), (2, 2)),
            n_res_blocks=3, n_convs_per_block=2,
        )

    .. versionadded:: 1.8

    Parameters
    ----------
    encoder_channels : tuple of int, optional
        Output widths of the encoder stages. The default is
        ``(32, 128, 256)`` (emg2pose).
    encoder_pool_sizes : tuple of tuple of int, optional
        Per-stage max-pool factors as ``(time, electrodes)``. The default
        is ``((10, 2), (8, 2), (4, 4))``; Liu et al. use
        ``((5, 2), (4, 2), (2, 2))`` at 200 Hz.
    decoder_channels : tuple of int, optional
        Output widths of the decoder stages. The default is
        ``(128, 32, 1)``.
    decoder_upsample_sizes : tuple of tuple of int, optional
        Per-stage nearest-neighbour upsampling factors as
        ``(time, electrodes)``. The default is
        ``((10, 4), (8, 4), (4, 2))``.
    n_res_blocks : int, optional
        Residual blocks between encoder and decoder. The default is ``5``.
    n_convs_per_block : int, optional
        Convolutions inside each residual block. The default is ``3``.
    kernel_size : tuple of int, optional
        Convolution kernel as ``(time, electrodes)``, applied with
        ``padding="same"`` throughout. The default is ``(3, 2)``.
    activation : type[nn.Module], optional
        Activation class used throughout. The default is ``nn.ReLU``.
    drop_prob : float, optional
        Dropout probability inside every block. The default is ``0.05``
        (paper value).

    References
    ----------
    .. [liu2021neuropose] Liu, Zhang, Gowda (2021). NeuroPose: 3D Hand
       Pose Tracking using EMG Wearables. The Web Conference 2021 /
       IEEE Internet of Things Journal 46(1), 2022.
       doi:10.1145/3442381.3449890
    .. [salter2024] Salter, Warren, Schlager, Spurr, Han, Bhasin, Cai,
       Walkington, Bolarinwa, Wang, Wang, Danielson, Merel, Pnevmatikakis,
       Marshall (2024). emg2pose: A Large and Diverse Benchmark for
       Surface Electromyographic Hand Pose Estimation. NeurIPS Datasets
       and Benchmarks. arXiv:2412.02725
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
        encoder_channels: tuple[int, ...] = (32, 128, 256),
        encoder_pool_sizes: tuple[tuple[int, int], ...] = ((10, 2), (8, 2), (4, 4)),
        decoder_channels: tuple[int, ...] = (128, 32, 1),
        decoder_upsample_sizes: tuple[tuple[int, int], ...] = (
            (10, 4),
            (8, 4),
            (4, 2),
        ),
        n_res_blocks: int = 5,
        n_convs_per_block: int = 3,
        kernel_size: tuple[int, int] = (3, 2),
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.05,
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

        if len(encoder_channels) != len(encoder_pool_sizes):
            raise ValueError(
                "encoder_channels and encoder_pool_sizes must have the same "
                f"length; got {len(encoder_channels)} and {len(encoder_pool_sizes)}."
            )
        if len(decoder_channels) != len(decoder_upsample_sizes):
            raise ValueError(
                "decoder_channels and decoder_upsample_sizes must have the same "
                f"length; got {len(decoder_channels)} and "
                f"{len(decoder_upsample_sizes)}."
            )
        if n_res_blocks < 0:
            raise ValueError(f"n_res_blocks must be >= 0; got {n_res_blocks}.")
        if n_convs_per_block < 1:
            raise ValueError(
                f"n_convs_per_block must be >= 1; got {n_convs_per_block}."
            )

        self.encoder = nn.Sequential(
            *[
                _EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    activation=activation,
                    drop_prob=drop_prob,
                )
                for in_channels, out_channels, pool_size in zip(
                    (1, *encoder_channels[:-1]), encoder_channels, encoder_pool_sizes
                )
            ]
        )
        self.resnet = nn.Sequential(
            *[
                _ResBlock(
                    channels=encoder_channels[-1],
                    kernel_size=kernel_size,
                    n_convs=n_convs_per_block,
                    activation=activation,
                    drop_prob=drop_prob,
                )
                for _ in range(n_res_blocks)
            ]
        )
        self.decoder = nn.Sequential(
            *[
                _DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    upsample_size=upsample_size,
                    activation=activation,
                    drop_prob=drop_prob,
                )
                for in_channels, out_channels, upsample_size in zip(
                    (encoder_channels[-1], *decoder_channels[:-1]),
                    decoder_channels,
                    decoder_upsample_sizes,
                )
            ]
        )

        # Electrode axis: max-pool floors it through the encoder, nearest
        # upsampling multiplies it back through the decoder.
        n_electrodes = self.n_chans
        for _, pool_electrodes in encoder_pool_sizes:
            n_electrodes //= pool_electrodes
        if n_electrodes < 1:
            raise ValueError(
                f"n_chans={self.n_chans} is too small for the encoder's electrode "
                "pooling; it must stay >= 1 after division by "
                f"{[pool for _, pool in encoder_pool_sizes]}."
            )
        for _, upsample_electrodes in decoder_upsample_sizes:
            n_electrodes *= upsample_electrodes

        self.input_to_encoder = Rearrange(
            "batch nchans ntimes -> batch 1 ntimes nchans"
        )
        self.decoder_to_sequence = Rearrange(
            "batch features ntimes nchans -> batch ntimes (features nchans)"
        )
        self.sequence_to_channels = Rearrange(
            "batch ntimes njoints -> batch njoints ntimes"
        )
        self.channels_to_sequence = Rearrange(
            "batch njoints ntimes -> batch ntimes njoints"
        )
        self.final_layer = nn.Linear(
            in_features=decoder_channels[-1] * n_electrodes,
            out_features=self.n_outputs,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize trainable layers with the Braindecode model convention."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm),
        ):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the output layer for a new output dimensionality."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(
            in_features=old.in_features, out_features=n_outputs
        ).to(device=old.weight.device, dtype=old.weight.dtype)
        self._init_weights(self.final_layer)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_times_in = x.shape[-1]
        encoder_input = self.input_to_encoder(x)
        encoded = self.encoder(encoder_input)
        bottleneck = self.resnet(encoded)
        decoded = self.decoder(bottleneck)
        decoded_sequence = self.decoder_to_sequence(decoded)
        pose = self.final_layer(decoded_sequence)

        # Upstream needs n_times divisible by the total temporal pooling
        # factor; this restores whatever flooring dropped, and is the identity
        # when it does divide.
        pose_channels = self.sequence_to_channels(pose)
        resized_pose_channels = torch.nn.functional.interpolate(
            pose_channels,
            size=n_times_in,
            mode="linear",
            align_corners=False,
        )
        resized_pose = self.channels_to_sequence(resized_pose_channels)
        return resized_pose


class _EncoderBlock(nn.Module):
    """Conv-BN-activation-dropout-maxpool over (time, electrodes)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        pool_size: tuple[int, int],
        activation: type[nn.Module],
        drop_prob: float,
    ) -> None:
        super().__init__()
        # ``network`` mirrors the upstream block layout.
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_channels),
            activation(),
            nn.Dropout(p=drop_prob),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.network(x)
        return encoded


class _ResBlock(nn.Module):
    """``x + f(x)`` over ``n_convs`` conv-BN-activation-dropout groups."""

    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int],
        n_convs: int,
        activation: type[nn.Module],
        drop_prob: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                nn.BatchNorm2d(num_features=channels),
                activation(),
                nn.Dropout(p=drop_prob),
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No activation after the sum, following the reference implementation.
        residual = self.network(x)
        output = x + residual
        return output


class _DecoderBlock(nn.Module):
    """Conv-BN-activation-dropout-upsample over (time, electrodes)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        upsample_size: tuple[int, int],
        activation: type[nn.Module],
        drop_prob: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_channels),
            activation(),
            nn.Dropout(p=drop_prob),
            nn.Upsample(
                scale_factor=(float(upsample_size[0]), float(upsample_size[1])),
                mode="nearest",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.network(x)
        return decoded

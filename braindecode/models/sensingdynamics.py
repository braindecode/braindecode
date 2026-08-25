# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Raul C. Sîmpetru, Alessandro Del Vecchio (original SensingDynamics)
#
# License: BSD (3-clause)
# Reimplementation for research/benchmarking; the original works are
# bioRxiv 2022.07.29.502064 (CC BY-NC-ND) and IEEE TBME 2024.
"""``SensingDynamicsNet``: 3D-CNN on high-density sEMG grids."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class SensingDynamicsNet(EEGModuleMixin, nn.Module):
    r"""SensingDynamicsNet from Sîmpetru et al (2022) [simpetru2022]_.

    :bdg-success:`Convolution` :bdg-dark-line:`Channel`

    Three-dimensional convolutional network over grid-structured
    high-density sEMG (depth × grids × electrodes × time), mapping
    windows of raw forearm/wrist signals to dense hand-pose frames.
    The original setup uses 320 electrodes (5 grids × 64 channels,
    raw ⊕ 20 Hz low-pass depth copies); this port keeps that geometry
    parameterized so it also runs on emg2pose wristbands
    (``n_grids=1``, ``elec_per_grid=16``).

    .. rubric:: Architecture Overview

    ``(B, C, T)`` → internal envelope copy (fixed low-pass stand-in for
    the paper's filtered depth slice) → reshape to
    ``(B, 2, grids, elec, T)`` → temporal MUAP detector Conv3d
    (1,1,31)/stride (1,1,8) → InstanceNorm + GELU → circular padding →
    grid-mixing Conv3d spanning all grids, dilation on time → adaptive
    pool to ``n_frames`` positions → per-frame MLP head → linear
    interpolation to ``(B, T, n_outputs)``.

    .. rubric:: Macro Components

    ``SensingDynamicsNet.stem`` (temporal MUAP detector)
        **Operations.** Conv3d kernel (1, 1, 31), stride (1, 1, 8)
        (~15 ms window / ~4 ms shift at 2 kHz), InstanceNorm3d + GELU;
        searches single-electrode traces for action-potential overlap
        patterns.
        **Role.** The paper's first stage: per-electrode spike-train
        feature extraction.

    ``SensingDynamicsNet.grid_conv`` (grid mixer)
        **Operations.** Circular padding over electrode/time dims
        preserving recording synchronicity, then Conv3d with kernel
        spanning every grid at once and temporal dilation 2.
        **Role.** Encodes inter-grid muscle synergies into a compact
        representation.

    ``SensingDynamicsNet.head_ff`` / ``SensingDynamicsNet.final_layer``
        **Operations.** Per-frame MLP (``mlp_hidden`` twice, GELU,
        dropout ``drop_prob``) followed by a linear projection to
        ``n_outputs``; outputs are interpolated to the input rate.
        **Role.** Dense pose regression on the retained frame grid.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** A strided detector and dilated grid mixer encode temporal
      structure.
    - **Channels/space:** Explicit grid/electrode convolutions model the sensor
      geometry.
    - **Frequency:** Raw and fixed low-pass depth slices provide complementary
      spectral views.

    .. rubric:: Additional Mechanisms

    The real-time variant's design choices are used here
    ([simpetru2023]_): InstanceNorm instead of BatchNorm and
    GELU instead of SMU, keeping normalization batch-independent for
    streaming use. Subject-specific transfer learning and the
    queue-based prediction smoother remain outside this module.

    .. versionadded:: 1.8

    Parameters
    ----------
    n_grids : int, optional
        Number of electrode grids. The default is ``1``.
    elec_per_grid : int | None, optional
        Electrodes per grid; inferred as ``n_chans / n_grids`` when
        omitted.
    temporal_channels : int, optional
        Width of the MUAP-detector stage. The default is ``256``
        (paper value; reduce for wristband-scale inputs).
    mid_channels : int, optional
        Width after the grid mixer. The default is ``32``.
    mlp_hidden : int, optional
        Hidden width of the per-frame MLP. The default is ``512``.
    n_frames : int, optional
        Retained temporal positions before upsampling. The default is
        ``24``.
    activation : type[nn.Module], optional
        Activation class used throughout. The default is ``nn.GELU``.
    drop_prob : float, optional
        Dropout inside the per-frame head. The default is ``0.25``.

    References
    ----------
    .. [simpetru2022] Sîmpetru, Arkudas, Braun, Osswald, Oliveira,
       Eskofier, Kinfe, Del Vecchio (2022). Sensing the Full Dynamics
       of the Human Hand with a Neural Interface and Deep Learning.
       bioRxiv 2022.07.29.502064; IEEE TBME 71(6), 2024.
       doi:10.1101/2022.07.29.502064
    .. [simpetru2023] Sîmpetru, März, Del Vecchio (2023). Proportional
       and Simultaneous Real-Time Control of the Full Human Hand From
       High-Density Electromyography. doi:10.36227/techrxiv.21904335
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
        n_grids: int = 1,
        elec_per_grid: int | None = None,
        temporal_channels: int = 256,
        mid_channels: int = 32,
        mlp_hidden: int = 512,
        n_frames: int = 24,
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.25,
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

        self.n_grids = int(n_grids)
        if self.n_grids <= 0:
            raise ValueError(f"n_grids must be positive; got {n_grids}.")
        if elec_per_grid is None:
            elec_per_grid = self.n_chans // self.n_grids
        self.elec_per_grid = int(elec_per_grid)
        if self.elec_per_grid <= 0:
            raise ValueError(f"elec_per_grid must be positive; got {elec_per_grid}.")
        if self.elec_per_grid * self.n_grids != self.n_chans:
            raise ValueError("n_chans must equal n_grids * elec_per_grid")
        self.n_frames = max(1, int(n_frames))

        # Fixed-envelope stand-in for the paper's 20 Hz-filtered depth
        # copy: an average filter of ≈50 ms.
        k = max(3, int(round(float(self.sfreq) * 0.05)) | 1)
        self.register_buffer("env_kernel", torch.ones(1, 1, k) / k, persistent=False)
        self.flatten_channels = Rearrange("b c t -> (b c) 1 t")
        self.restore_channels = Rearrange("(b c) 1 t -> b c t", c=self.n_chans)
        self.channels_to_grid = Rearrange(
            "b d (g e) t -> b d g e t",
            g=self.n_grids,
            e=self.elec_per_grid,
        )

        pad_e = min(2, max(0, self.elec_per_grid - 1))
        self.pad_e = pad_e
        self.pad_t = 16
        self.stem = nn.Sequential(
            nn.Conv3d(2, temporal_channels, (1, 1, 31), stride=(1, 1, 8)),
            nn.InstanceNorm3d(temporal_channels),
            activation(),
        )
        self.grid_conv = nn.Sequential(
            nn.Conv3d(
                temporal_channels,
                mid_channels,
                # Paper convention: dilation (1, 2, 1) makes the electrode
                # kernel cover nearly the full padded axis sparsely (effective
                # size = 2*k_e - 1; upstream 32 -> 63 over 64 electrodes).
                kernel_size=(
                    self.n_grids,
                    min(32, (self.elec_per_grid + 2 * pad_e + 1) // 2),
                    18,
                ),
                dilation=(1, 2, 1),
            ),
            nn.InstanceNorm3d(mid_channels),
            activation(),
        )
        self.head_ff = nn.Sequential(
            nn.Linear(mid_channels, mlp_hidden),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_hidden, mlp_hidden),
            activation(),
        )
        self.grid_to_sequence = Rearrange("b c g e t -> b t (c g e)")
        self.sequence_to_channels = Rearrange("b t j -> b j t")
        self.channels_to_sequence = Rearrange("b j t -> b t j")
        self.final_layer = nn.Linear(mlp_hidden, self.n_outputs)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm3d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the output layer for a new output dimensionality."""
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        self.final_layer.apply(self._init_weights)
        self._n_outputs = n_outputs
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        env = self.restore_channels(
            F.conv1d(
                self.flatten_channels(x),
                self.env_kernel,
                padding=self.env_kernel.shape[-1] // 2,
            )
        )[..., :t]
        xin = self.channels_to_grid(torch.stack([x, env], dim=1))
        h = F.pad(
            xin,
            (self.pad_t, self.pad_t, self.pad_e, self.pad_e, 0, 0),
            mode="circular",
        )
        h = self.grid_conv(self.stem(h))  # (B, D, 1, e', k)
        h = F.adaptive_avg_pool3d(h, (1, 1, self.n_frames))
        h = self.grid_to_sequence(h)
        out = self.final_layer(self.head_ff(h))  # (B, K, J)
        return self.channels_to_sequence(
            F.interpolate(
                self.sequence_to_channels(out),
                size=t,
                mode="linear",
                align_corners=False,
            )
        )

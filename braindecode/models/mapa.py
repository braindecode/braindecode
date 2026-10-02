# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/bentang18/MAPA (Apache-2.0).
# Copyright 2026 Ben Tang, Zachary Spalding, and Gregory B. Cogan
"""MAPA: masked autoencoding of iEEG with anatomical priors.

Reimplementation of MAPA (Tang, Spalding & Cogan, 2026), "Pretraining for
Sample-Efficient Neural Interfaces". The architecture is transcribed from the
authors' reference implementation (https://github.com/bentang18/MAPA), which is
released under the Apache 2.0 license and itself follows V-JEPA 2 for its
initialization and pre-norm transformer blocks.

Original Authors: Tang, Spalding & Cogan, Duke University
Braindecode Adaptation: Julien Gadonneix
"""

from __future__ import annotations

import math
import re
import warnings
from numbers import Integral
from typing import NamedTuple, cast

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import rescale_parameter, rotate_pairs
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock

# The reference LayerNorm eps, from ``models/attention.py``.
_LN_EPS = 1e-6
# Depth and head dimension are held fixed across the released widths, so the
# head count follows from ``d_model`` alone.
_DEPTH = 12
_HEAD_DIM = 64
# Blocks whose output carries a deep-supervision norm, following V-JEPA 2.1.
_SUP_TAPS: tuple[int, ...] = (3, 6, 9, 12)

# The frontend is defined at the reference's 2048 Hz, where a hop of 64 samples
# puts every band on the same 32 Hz frame clock.
_SAMPLE_RATE = 2048
_FRAME_RATE = 32
_HOP = _SAMPLE_RATE // _FRAME_RATE
# Name, FFT length, first and last rfft bin (inclusive), and the decimation
# stride on the shared frame clock, in the order slow, mid, fast. At 2048 Hz the
# retained bins span 2-14 Hz, 16-56 Hz and 64-160 Hz.
_BANDS: tuple[tuple[str, int, int, int, int], ...] = (
    ("slow", 1024, 1, 7, 8),
    ("mid", 256, 2, 7, 2),
    ("fast", 128, 4, 10, 1),
)
_BAND_BINS: tuple[int, ...] = tuple(k1 - k0 + 1 for _, _, k0, k1, _ in _BANDS)
# The slow band is the coarsest, so a window must hold a whole number of its
# tokens.
_FRAME_QUANTUM = max(stride for *_, stride in _BANDS)
# The published "Guard 3" caps on the normalized inputs, per band.
_INPUT_CLIP_Z: tuple[float, float, float] = (15.0, 15.0, 20.0)

# Robust z-score constants, from ``data/normalize.py``.
_MAD_TO_SIGMA = 1.4826
_SIGMA_FLOOR = 1e-6

# Rotary bases: ordinal contact numbers are dense, time slots are not.
_ROPE_BASE_CONTACT = 8.0
_ROPE_BASE_TIME = 64.0

# V-JEPA 2 initialization, and the near-zero std the reference gives to
# everything that is *added* to the residual stream.
_INIT_STD = 0.02
_ADDITIVE_INIT_STD = 1e-6

# The 31 DKT cortical parcels: the 34 Desikan-Killiany gyral labels minus the
# three (bankssts, frontalpole, temporalpole) whose boundaries the DKT protocol
# could not define reliably and reassigned to their neighbours.
_DKT_CORTICAL_PARCELS: tuple[str, ...] = (
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "transversetemporal",
)
# The six aseg structures the atlas keeps, which DKT leaves untouched.
_DKT_SUBCORTICAL_STRUCTURES: tuple[str, ...] = (
    "Hippocampus",
    "Amygdala",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Thalamus-Proper",
)

#: FreeSurfer DKT region names, in the slot order of MAPA's region table.
#:
#: The 62 hemisphere-qualified cortical parcels come first, then the 12
#: subcortical structures, which is the order that indexes the released region
#: embedding. Slot ``len(MAPA_DKT_REGIONS)`` is the reserved slot given to a
#: contact that falls outside every region.
MAPA_DKT_REGIONS: tuple[str, ...] = tuple(
    f"ctx-{hemisphere}-{parcel}"
    for hemisphere in ("lh", "rh")
    for parcel in _DKT_CORTICAL_PARCELS
) + tuple(
    f"{hemisphere}-{structure}"
    for hemisphere in ("Left", "Right")
    for structure in _DKT_SUBCORTICAL_STRUCTURES
)

_N_REGIONS = len(MAPA_DKT_REGIONS) + 1
_UNASSIGNED_REGION = len(MAPA_DKT_REGIONS)

# Clinical electrode labels are an array name followed by a contact number.
_LABEL_PATTERN = re.compile(r"^(.*?)(\d+)$")


class _TokenLayout(NamedTuple):
    """Where each token sits, for one montage and one window length."""

    gather_idx: torch.Tensor
    key_mask: torch.Tensor
    token_region: torch.Tensor
    scatter_idx: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    k_full: int


class MAPA(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""MAPA from Tang, Spalding and Cogan (2026) [Tang2026]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`
    :bdg-dark-line:`Channel`

    .. versionadded:: 1.8.2

    .. rubric:: Architecture Overview

    MAPA is a masked-autoencoder foundation model for intracranial EEG whose
    only knowledge of an electrode is anatomical: the atlas region of each
    contact and its position along the implanted array. Coordinates, montage
    size and channel order never reach the model, so one pretrained encoder
    reads subjects it has never seen [Tang2026]_. It works in four stages:

    1. Turn each channel into slow, mid and fast magnitude spectrograms on a
       shared 32 Hz frame clock, robust z-scored and clipped.
    2. Embed each ``(contact, band, time)`` token with its band's linear layer,
       a per-band vector and the embedding of its contact's DKT region.
    3. Apply twelve pre-norm transformer blocks whose attention spans contacts
       and time jointly but never crosses from one array to another.
    4. Pool the normed outputs of blocks 3, 6, 9 and 12 and classify them.

    .. rubric:: Macro Components

    ``MAPA.frontend``
        **Operations.** Hann STFTs of FFT length 1024, 256 and 128 with a
        shared hop of 64 samples; an inclusive bin slice per band; a robust
        z-score per contact and bin; the published caps; a decimation by 8, 2
        and 1.

        **Role.** Produce the inputs the released encoder consumes. Given a
        spectrogram normalized upstream, it only clips and decimates.

    ``MAPA.stem``
        **Operations.** One linear layer per band maps its bins to ``d_model``,
        and a learned per-band vector is added.

        **Role.** Embed tokens while keeping the bands distinguishable. There is
        no frequency embedding and no per-band normalization, which would
        restore the :math:`1/f` dominance the robust z-score removes.

    ``MAPA.encoder``
        **Operations.** Add the learned region embedding once, then apply twelve
        blocks of within-array self-attention, with a two-axis rotary encoding,
        and a feed-forward block of ratio ``mlp_ratio``. With ``deep_sup``,
        blocks 3, 6, 9 and 12 each get their own LayerNorm and are
        concatenated.

        **Role.** Mix contacts and time inside each array.

    ``MAPA.final_layer``
        **Operations.** Mean-pool or flatten the tokens, then apply a linear
        layer.

        **Role.** Adapt the frozen encoder to classification. It is not part of
        the pretraining.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** a 32 Hz frame clock on which the slow, mid and fast bands
      carry one token every 8, 2 and 1 frames. Tokens on the same slot share a
      rotary phase.
    - **Spatial:** the clinical contact number along the array, as one rotary
      axis, plus the additive region embedding. Attention is block-diagonal
      over arrays, and no coordinates are used.
    - **Spectral:** the three STFT bands, spanning 2-14, 16-56 and 64-160 Hz at
      2048 Hz, are the input representation.

    .. rubric:: Additional Mechanisms

    *Electrode metadata.* The array and the contact number are read off the
    clinical label (``"LA7"`` is contact 7 of array ``LA``), from
    ``contact_labels`` or else the ``chs_info`` names. Contact numbers are kept
    verbatim, gaps included, and a label without a trailing number is rejected.
    With neither, the channels form one array numbered from 1, which is almost
    certainly not the real montage. Regions must be exact names of
    :data:`MAPA_DKT_REGIONS` and default to the reserved unassigned slot.

    *One model, many subjects.* The constructor's montage is only the default.
    To read another recording, pass :meth:`forward` the ``sensor_indices`` that
    :meth:`sensor_indices` builds from its labels and regions; all samples of a
    batch share them. The token layout is cached until the montage or the
    window length changes. Only ``pooling="flatten"`` is tied to one montage
    and one window length.

    *Sampling rate and window length.* The bands are FFT bin slices at 2048 Hz,
    so resample to 2048 Hz. A window yields ``1 + n_times // 64`` frames,
    truncated to a multiple of 8, which needs at least 448 samples. With
    ``normalization="session"`` the input is the spectrogram itself
    (``sfreq=32``, ``n_times`` in frames): the 20 retained bins, slow then mid
    then fast, robust z-scored per contact and bin over the whole recording.

    *Pretrained weights.* ``MAPA.stem`` and ``MAPA.encoder`` keep the reference
    module names (the feed-forward ``fc1``/``fc2`` are renamed on load through
    ``mapping``), so a released ``checkpoint["model"]`` (``mapa_vits384`` or
    one of its three spatial ablations, ``region_embed`` and ``space_rope``
    set to ``False`` alone or together) loads with
    :meth:`~torch.nn.Module.load_state_dict` under ``strict=False``, leaving
    only ``final_layer`` uninitialized. The default configuration has the
    released 21,335,424 parameters, excluding the head.

    .. note::
        Differences from the reference implementation:

        - Attention runs on arrays padded to a common size, with the padding
          masked, rather than on one ragged sequence; the outputs are the same.
        - ``normalization="window"`` fits the robust z-score on each window
          rather than on the whole recording. Only ``normalization="session"``
          reproduces the reference inputs.
        - The pooled features are the normed four-tap concatenation, whereas
          the paper's frozen evaluation reads block 12 before that norm.
        - The pretraining objective and decoder, the anatomical localization
          and the artifact detectors ("Guard 1" and "Guard 2") are out of
          scope; the input clipping ("Guard 3") is kept.

    Parameters
    ----------
    contact_labels : list of str, optional
        Clinical label of each channel, such as ``"LA7"``. Defaults to the
        ``chs_info`` channel names, then to a single array numbered from 1.
    regions : list of str or int or None, optional
        DKT region of each channel, as an exact name from
        :data:`MAPA_DKT_REGIONS` or its integer slot; ``None`` selects the
        reserved unassigned slot. Defaults to unassigned everywhere.
    d_model : int
        Token embedding dimension, a multiple of the head dimension 64.
        Default 384, the released ``mapa_vits384``.
    mlp_ratio : int
        Hidden dimension of the feed-forward blocks, as a multiple of
        ``d_model``.
    region_embed : bool
        Whether the region embedding is used. ``False`` is the paper's
        ``no_region`` ablation.
    space_rope : bool
        Whether the rotary encoding carries the contact number. ``False`` is
        the paper's ``no_relpos`` ablation.
    deep_sup : bool
        Whether the encoder returns the four normed deep-supervision taps,
        concatenated to width ``4 * d_model``, or one terminal LayerNorm.
    pooling : {"mean", "flatten"}
        ``"mean"`` averages the tokens, so the head fits any montage and window
        length; ``"flatten"`` keeps every token, so the head fits one only.
    normalization : {"window", "session", "none"}
        ``"window"`` robust z-scores the spectrograms of each raw window;
        ``"session"`` takes the spectrogram itself, normalized over the whole
        recording upstream; ``"none"`` passes the raw STFT magnitude.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks.

    Examples
    --------
    >>> import torch
    >>> from braindecode.models import MAPA
    >>> model = MAPA(
    ...     n_outputs=2,
    ...     n_chans=3,
    ...     n_times=2048,
    ...     sfreq=2048,
    ...     contact_labels=["LA1", "LA2", "LB4"],
    ...     regions=["ctx-lh-insula", "ctx-lh-insula", "Left-Hippocampus"],
    ... )
    >>> model(torch.randn(4, 3, 2048)).shape
    torch.Size([4, 2])

    The same model reads another subject, given that subject's electrodes:

    >>> other = MAPA.sensor_indices(["RC1", "RC2", "RC3", "RD7"])
    >>> model(torch.randn(4, 4, 4096), other).shape
    torch.Size([4, 2])

    References
    ----------
    .. [Tang2026] Tang, B., Spalding, Z. & Cogan, G. B. (2026). Pretraining for
       sample-efficient neural interfaces. https://arxiv.org/abs/2609.13507
    """

    def __init__(
        self,
        # --- signal-related (handled by EEGModuleMixin) ---
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # --- model hyperparameters (defaults: the released mapa_vits384) ---
        *,
        contact_labels: list[str] | None = None,
        regions: list[str | int | None] | None = None,
        d_model: int = 384,
        mlp_ratio: int = 4,
        region_embed: bool = True,
        space_rope: bool = True,
        deep_sup: bool = True,
        pooling: str = "mean",
        normalization: str = "window",
        activation: type[nn.Module] = nn.GELU,
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

        if d_model <= 0 or d_model % _HEAD_DIM:
            raise ValueError(
                f"d_model must be a positive multiple of the head dimension "
                f"{_HEAD_DIM}, got {d_model}."
            )
        if pooling not in ("mean", "flatten"):
            raise ValueError(f"pooling must be 'mean' or 'flatten', got {pooling!r}.")
        if normalization not in ("window", "session", "none"):
            raise ValueError(
                f"normalization must be 'window', 'session' or 'none', got "
                f"{normalization!r}."
            )

        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.region_embed = region_embed
        self.space_rope = space_rope
        self.deep_sup = deep_sup
        self.pooling = pooling
        self.normalization = normalization
        self.activation = activation

        try:
            sfreq = float(self.sfreq)
        except ValueError:
            sfreq = None
        expected = _FRAME_RATE if normalization == "session" else _SAMPLE_RATE
        if sfreq is not None and not math.isclose(sfreq, expected):
            warnings.warn(
                f"With normalization={normalization!r}, MAPA expects sfreq="
                f"{expected} Hz but got {sfreq} Hz: its bands are FFT bin slices "
                f"at {_SAMPLE_RATE} Hz on a {_FRAME_RATE} Hz frame clock, so "
                f"other rates shift the bands and the token rate.",
                UserWarning,
            )

        self.n_frames = _frame_count(
            self.n_times, spectrogram=normalization == "session"
        )
        self.k_full = sum(_band_lengths(self.n_frames))

        self.frontend = _SpectrogramFrontend(normalization=normalization)
        self.stem = _PerBandStem(d_model=d_model, band_bins=_BAND_BINS)
        self.encoder = _Encoder(
            d_model=d_model,
            n_heads=d_model // _HEAD_DIM,
            mlp_ratio=mlp_ratio,
            region_embed=region_embed,
            deep_sup=deep_sup,
            activation=activation,
        )
        self.mapping = {
            f"encoder.blocks.{i}.mlp.fc{fc}.{param}": (
                f"encoder.blocks.{i}.mlp.{child}.{param}"
            )
            for i in range(_DEPTH)
            for fc, child in ((1, 0), (2, 3))
            for param in ("weight", "bias")
        }

        # The montage resolved here is only the default: forward takes another
        # recording's metadata directly, which is what lets one instance read
        # subjects it was not built for. Its layout rides along as buffers, so
        # it follows the module across devices and nothing has to be laid out
        # at call time unless the montage or the window actually changes.
        indices = self.sensor_indices(self._resolve_labels(contact_labels), regions)
        self.register_buffer("default_sensor_indices", indices, persistent=False)
        default_layout = _build_token_layout(indices, self.n_frames, space_rope)
        for name, value in default_layout._asdict().items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(name, value, persistent=False)
        self._layout_cache: tuple[int, torch.Tensor, _TokenLayout] | None = None

        feature_dim = d_model * (len(_SUP_TAPS) if deep_sup else 1)
        n_features = (
            feature_dim
            if pooling == "mean"
            else feature_dim * self.n_chans * self.k_full
        )
        self.final_layer = nn.Linear(n_features, self.n_outputs)

    @staticmethod
    def sensor_indices(
        contact_labels: list[str],
        regions: list[str | int | None] | None = None,
    ) -> torch.Tensor:
        """Assemble one recording's electrode metadata for :meth:`forward`.

        Parameters
        ----------
        contact_labels : list of str
            Clinical label of each channel, such as ``"LA7"``, from which the
            array and the contact number are read.
        regions : list of str or int or None, optional
            DKT region of each channel, either an exact name from
            :data:`MAPA_DKT_REGIONS` or its integer slot, with ``None``
            selecting the reserved unassigned slot. Defaults to unassigned
            everywhere.

        Returns
        -------
        torch.Tensor
            ``(n_chans, 3)`` long tensor whose columns are the array, the
            contact number along it, and the region slot.

        Examples
        --------
        >>> from braindecode.models import MAPA
        >>> MAPA.sensor_indices(["LA1", "LA3", "LB2"]).tolist()
        [[0, 1, 74], [0, 3, 74], [1, 2, 74]]
        """
        labels = [str(label) for label in contact_labels]
        arrays, contacts = _parse_contact_labels(labels)
        return torch.stack(
            [arrays, contacts, _resolve_regions(regions, len(labels))], dim=1
        )

    def _resolve_sensor_indices(
        self, sensor_indices: torch.Tensor | None, x: torch.Tensor
    ) -> torch.Tensor:
        """Validate a recording's metadata, or fall back to the default montage."""
        n_chans = x.shape[1]
        if sensor_indices is None:
            if n_chans != self.n_chans:
                raise ValueError(
                    f"MAPA resolved a {self.n_chans}-channel montage at "
                    f"construction but got input with {n_chans} channels. Pass "
                    f"this recording's metadata as sensor_indices, which "
                    f"MAPA.sensor_indices builds from its contact labels."
                )
            return self.get_buffer("default_sensor_indices")
        indices = torch.as_tensor(sensor_indices)
        if (
            indices.is_floating_point()
            or indices.is_complex()
            or indices.dtype == torch.bool
            or indices.shape != (n_chans, 3)
            or bool((indices < 0).any())
            or bool((indices[:, 2] >= _N_REGIONS).any())
        ):
            raise ValueError(
                f"sensor_indices must hold integers of shape ({n_chans}, 3), one "
                f"row of (array, contact number, region slot) per channel, and "
                f"must be non-negative, with region slots below {_N_REGIONS}; got "
                f"{indices.dtype} of shape {tuple(indices.shape)}."
            )
        return indices.to(device=x.device, dtype=torch.long)

    def _token_layout(self, indices: torch.Tensor, n_frames: int) -> _TokenLayout:
        """Return the token layout of a montage, rebuilding it only when it changes."""
        if (
            indices is self.get_buffer("default_sensor_indices")
            and n_frames == self.n_frames
        ):
            return _TokenLayout(
                gather_idx=self.get_buffer("gather_idx"),
                key_mask=self.get_buffer("key_mask"),
                token_region=self.get_buffer("token_region"),
                scatter_idx=self.get_buffer("scatter_idx"),
                rope_cos=self.get_buffer("rope_cos"),
                rope_sin=self.get_buffer("rope_sin"),
                k_full=self.k_full,
            )
        cache = self._layout_cache
        if (
            cache is not None
            and cache[0] == n_frames
            and cache[1].shape == indices.shape
            and cache[1].device == indices.device
            and bool(torch.equal(cache[1], indices))
        ):
            return cache[2]
        layout = _build_token_layout(indices, n_frames, self.space_rope)
        self._layout_cache = (n_frames, indices.clone(), layout)
        return layout

    def _resolve_labels(self, contact_labels: list[str] | None) -> list[str]:
        """Return the clinical label of every channel."""
        if contact_labels is not None:
            if len(contact_labels) != self.n_chans:
                raise ValueError(
                    f"contact_labels has {len(contact_labels)} labels but the "
                    f"model has {self.n_chans} channels."
                )
            return [str(label) for label in contact_labels]
        if self._chs_info:
            return [str(channel["ch_name"]) for channel in self.chs_info]
        return [f"A{contact + 1}" for contact in range(self.n_chans)]

    def reset_head(self, n_outputs: int) -> None:
        """Replace the linear classification head for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        self._update_init_kwargs(n_outputs=n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        sensor_indices: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        """Encode an iEEG batch into class logits.

        Parameters
        ----------
        x : torch.Tensor
            Raw signal of shape ``(batch, n_chans, n_times)``, or with
            ``normalization="session"`` the normalized spectrogram of shape
            ``(batch, n_chans, 20, n_frames)``.
        sensor_indices : torch.Tensor, optional
            ``(n_chans, 3)`` electrode metadata of the recording this batch
            comes from, one row of (array, contact number, region slot) per
            channel, as :meth:`sensor_indices` builds it. Every sample of the
            batch shares it. Defaults to the montage resolved at construction,
            which only fits the construction-time channel count.
        return_features : bool
            Whether to also return the pooled token embedding.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_outputs)``.
        """
        spectrogram = self.normalization == "session"
        if x.ndim != (4 if spectrogram else 3) or (
            spectrogram and x.shape[2] != sum(_BAND_BINS)
        ):
            raise ValueError(
                f"normalization='session' takes a spectrogram of shape (batch, "
                f"n_chans, {sum(_BAND_BINS)}, n_frames), any other a raw signal of "
                f"shape (batch, n_chans, n_times), but got {tuple(x.shape)} with "
                f"normalization={self.normalization!r}."
            )
        indices = self._resolve_sensor_indices(sensor_indices, x)
        n_frames = _frame_count(x.shape[-1], spectrogram=spectrogram)
        if self.pooling == "flatten" and (
            x.shape[1] != self.n_chans or n_frames != self.n_frames
        ):
            raise ValueError(
                f"pooling='flatten' ties the read-out to the {self.n_chans} "
                f"channels and {self.n_frames} frames it was built for, but got "
                f"{x.shape[1]} channels and {n_frames} frames. Build the model "
                f"with pooling='mean' to encode recordings of any shape."
            )
        layout = self._token_layout(indices, n_frames)

        tokens = self.stem(self.frontend(x))
        # (batch, n_arrays, max_contacts * k_full, d_model), array-contiguous.
        packed = tokens[:, layout.gather_idx].flatten(2, 3)
        encoded = self.encoder(
            packed,
            layout.token_region,
            layout.rope_cos,
            layout.rope_sin,
            layout.key_mask,
        )
        # Back to one block of tokens per channel, in the input channel order.
        encoded = encoded.unflatten(2, (-1, layout.k_full)).flatten(1, 2)
        encoded = encoded[:, layout.scatter_idx]

        if self.pooling == "mean":
            features = encoded.mean(dim=(1, 2))
        else:
            features = encoded.flatten(1)
        logits = self.final_layer(features)

        if return_features:
            return {
                "features": features,
                "cls_token": None,  # nosec B105
            }
        return logits


def _frame_count(n_times: int, spectrogram: bool = False) -> int:
    """Usable frames of the 32 Hz clock in ``n_times`` samples (or frames)."""
    frames = n_times if spectrogram else 1 + n_times // _HOP
    n_frames = (frames // _FRAME_QUANTUM) * _FRAME_QUANTUM
    if n_frames < _FRAME_QUANTUM:
        minimum = (
            f"{_FRAME_QUANTUM} frames"
            if spectrogram
            else f"{(_FRAME_QUANTUM - 1) * _HOP} samples"
        )
        raise ValueError(
            f"MAPA needs a window of at least {minimum}, one token of the slow "
            f"band, but got {n_times}."
        )
    return n_frames


def _band_lengths(n_frames: int) -> tuple[int, ...]:
    """Number of tokens each band lays on the shared clock."""
    return tuple(n_frames // stride for *_, stride in _BANDS)


def _build_token_layout(
    sensor_indices: torch.Tensor, n_frames: int, space_rope: bool
) -> _TokenLayout:
    """Regroup the channels into arrays padded to the largest one, and build
    the gather plan, padding mask, region ids and rotary tables of that grid."""
    device = sensor_indices.device
    n_chans = sensor_indices.shape[0]
    arrays, contacts, regions = sensor_indices.unbind(dim=1)
    band_lengths = _band_lengths(n_frames)
    k_full = sum(band_lengths)

    # Renumber the arrays contiguously, so any labelling of them works.
    _, array_of_contact = torch.unique(arrays, return_inverse=True)
    counts = torch.bincount(array_of_contact)
    n_arrays, max_contacts = counts.numel(), int(counts.max())

    # Row s holds the contacts of array s in input order, padded with contact
    # 0, whose tokens are masked out of attention and dropped on the way back.
    order = torch.argsort(array_of_contact, stable=True)
    row = array_of_contact[order]
    slot = (
        torch.arange(n_chans, device=device)
        - torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])[row]
    )
    gather_idx = torch.zeros((n_arrays, max_contacts), dtype=torch.long, device=device)
    valid = torch.zeros((n_arrays, max_contacts), dtype=torch.bool, device=device)
    gather_idx[row, slot] = order
    valid[row, slot] = True

    # Position of each contact in the flattened, array-major token grid, which
    # undoes the gather after the encoder.
    scatter_idx = torch.empty(n_chans, dtype=torch.long, device=device)
    scatter_idx[order] = row * max_contacts + slot

    # Each contact carries the same block of tokens: the three bands in order,
    # each at its own rate on the shared clock.
    lattice = torch.cat(
        [
            torch.arange(length, device=device) * stride
            for length, (*_, stride) in zip(band_lengths, _BANDS)
        ]
    )
    cos, sin = _rotary_table(
        contact=contacts[gather_idx].repeat_interleave(k_full, dim=1),
        time=lattice.repeat(max_contacts).expand(n_arrays, -1),
        head_dim=_HEAD_DIM,
        space_rope=space_rope,
    )
    return _TokenLayout(
        gather_idx=gather_idx,
        # Inserted axes broadcast the mask and the tables over the batch and
        # the heads.
        key_mask=valid.repeat_interleave(k_full, dim=1)[None, :, None, None],
        token_region=regions[gather_idx].repeat_interleave(k_full, dim=1),
        scatter_idx=scatter_idx,
        rope_cos=cos[None, :, None],
        rope_sin=sin[None, :, None],
        k_full=k_full,
    )


def _parse_contact_labels(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the array id (by first appearance of the prefix) and the verbatim
    contact number off each clinical label."""
    seen: dict[str, int] = {}
    arrays, contacts = [], []
    for label in labels:
        match = _LABEL_PATTERN.match(label)
        if match is None:
            raise ValueError(
                f"MAPA reads the array and the contact number off the clinical "
                f"label, but {label!r} has no trailing number, so it has no "
                f"position along an array. Pass clinical labels in "
                f"contact_labels, and drop non-neural channels beforehand."
            )
        prefix, number = match.group(1), int(match.group(2))
        arrays.append(seen.setdefault(prefix, len(seen)))
        contacts.append(number)
    return (
        torch.tensor(arrays, dtype=torch.long),
        torch.tensor(contacts, dtype=torch.long),
    )


def _resolve_regions(
    regions: list[str | int | None] | None, n_chans: int
) -> torch.Tensor:
    """Map each channel's region to a slot of the region table.

    Names must match :data:`MAPA_DKT_REGIONS` exactly, as in the reference, so
    a misspelling cannot silently become unassigned.
    """
    if regions is None:
        return torch.full((n_chans,), _UNASSIGNED_REGION, dtype=torch.long)
    if len(regions) != n_chans:
        raise ValueError(
            f"regions has {len(regions)} entries but the model has {n_chans} channels."
        )
    lookup = {name: slot for slot, name in enumerate(MAPA_DKT_REGIONS)}
    slots = []
    for region in regions:
        if region is None:
            slots.append(_UNASSIGNED_REGION)
        elif isinstance(region, Integral) and not isinstance(region, bool):
            slot = int(region)
            if not 0 <= slot < _N_REGIONS:
                raise ValueError(
                    f"region slot {slot} is outside the {_N_REGIONS} slots of "
                    f"MAPA's region table."
                )
            slots.append(slot)
        elif isinstance(region, str) and region in lookup:
            slots.append(lookup[region])
        else:
            raise ValueError(
                f"{region!r} is not a MAPA region. Names are the exact "
                f"FreeSurfer DKT entries of MAPA_DKT_REGIONS, such as "
                f"'ctx-lh-superiortemporal' or 'Left-Hippocampus'. Pass None "
                f"for a contact that falls outside every region."
            )
    return torch.tensor(slots, dtype=torch.long)


def _robust_z(x: torch.Tensor) -> torch.Tensor:
    """Median-centre and MAD-scale each contact and bin over time, zeroing
    bins whose scale is below the floor."""
    median = x.median(dim=-1, keepdim=True).values
    sigma = _MAD_TO_SIGMA * (x - median).abs().median(dim=-1, keepdim=True).values
    z = (x - median) / sigma.clamp(min=_SIGMA_FLOOR)
    return torch.where(sigma >= _SIGMA_FLOOR, z, torch.zeros_like(z))


def _rotary_table(
    contact: torch.Tensor, time: torch.Tensor, head_dim: int, space_rope: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of a rotary encoding whose head dimension is
    split between the contact number and the frame-clock position.

    Without ``space_rope`` the contact half is the identity rotation.
    """
    pairs = head_dim // 4
    exponents = torch.arange(pairs, dtype=torch.float32) / pairs
    contact_freq = 1.0 / (_ROPE_BASE_CONTACT**exponents)
    if not space_rope:
        contact_freq = torch.zeros_like(contact_freq)
    time_freq = 1.0 / (_ROPE_BASE_TIME**exponents)
    angle = torch.cat(
        [
            contact[..., None].float() * contact_freq,
            time[..., None].float() * time_freq,
        ],
        dim=-1,
    )
    return (
        angle.cos().repeat_interleave(2, dim=-1),
        angle.sin().repeat_interleave(2, dim=-1),
    )


def _init_transformer_weights(module: nn.Module) -> None:
    """V-JEPA 2 initialization of linear layers and layer norms."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=_INIT_STD)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class _SpectrogramFrontend(nn.Module):
    """Per-band STFT magnitudes on a shared 32 Hz clock, robust z-scored,
    capped and decimated to each band's token rate."""

    def __init__(self, normalization: str):
        super().__init__()
        self.normalization = normalization
        for name, n_fft, *_ in _BANDS:
            self.register_buffer(
                f"window_{name}", torch.hann_window(n_fft), persistent=False
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return ``(batch, n_chans, n_bins, n_tokens)`` slow, mid, fast bands."""
        if self.normalization == "session":
            n_frames = _frame_count(x.shape[-1], spectrogram=True)
            bands = x[..., :n_frames].split(_BAND_BINS, dim=2)
        else:
            bands = self._stft_bands(x)
        return [
            band.clamp(-cap, cap)[..., ::stride]
            for band, cap, (*_, stride) in zip(bands, _INPUT_CLIP_Z, _BANDS)
        ]

    def _stft_bands(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Take ``(batch, n_chans, n_times)`` to ``(batch, n_chans, n_bins, n_frames)``."""
        batch, n_chans = x.shape[0], x.shape[1]
        n_frames = _frame_count(x.shape[-1])
        waveform = x.reshape(batch * n_chans, x.shape[-1])
        bands = []
        for name, n_fft, k0, k1, _ in _BANDS:
            # A window shorter than the transform is zero-padded, as in the
            # reference, so the centred transform has something to reflect; the
            # frames past the true window are then dropped.
            padded = waveform
            if padded.shape[-1] < n_fft:
                padded = F.pad(padded, (0, n_fft - padded.shape[-1]))
            spectrum = torch.stft(
                padded,
                n_fft=n_fft,
                hop_length=_HOP,
                win_length=n_fft,
                window=self.get_buffer(f"window_{name}"),
                center=True,
                normalized=False,
                return_complex=True,
            )
            band = spectrum[:, k0 : k1 + 1, :n_frames].abs()
            if self.normalization == "window":
                band = _robust_z(band)
            bands.append(band.reshape(batch, n_chans, k1 - k0 + 1, n_frames))
        return bands


class _PerBandStem(nn.Module):
    """One linear projection plus an additive vector per band, with no
    per-band norm, which would bring back the 1/f the z-score removes."""

    def __init__(self, d_model: int, band_bins: tuple[int, ...]):
        super().__init__()
        self.projs = nn.ModuleList(nn.Linear(n_bins, d_model) for n_bins in band_bins)
        self.band_type_emb = nn.Parameter(torch.empty(len(band_bins), d_model))
        self.projs.apply(_init_transformer_weights)
        nn.init.trunc_normal_(self.band_type_emb, std=_ADDITIVE_INIT_STD)

    def forward(self, bands: list[torch.Tensor]) -> torch.Tensor:
        """Return ``(batch, n_chans, k_full, d_model)`` tokens."""
        tokens = [
            proj(band.transpose(-1, -2)) + embedding
            for band, proj, embedding in zip(bands, self.projs, self.band_type_emb)
        ]
        return torch.cat(tokens, dim=-2)


class _Encoder(nn.Module):
    """Region embedding plus a pre-norm stack of within-array blocks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        region_embed: bool,
        deep_sup: bool,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.region_embed = _RegionIdentityEmbed(d_model, enabled=region_embed)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                _WithinArrayBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    activation=activation,
                )
                for _ in range(_DEPTH)
            ]
        )
        # The deepest tap's norm is the terminal norm, so there is no separate
        # one when the taps are used.
        self.norms_block: nn.ModuleList | None = (
            nn.ModuleList([nn.LayerNorm(d_model, eps=_LN_EPS) for _ in _SUP_TAPS])
            if deep_sup
            else None
        )
        self.norm_out: nn.LayerNorm | None = (
            None if deep_sup else nn.LayerNorm(d_model, eps=_LN_EPS)
        )
        self.apply(_init_transformer_weights)
        for layer, module in enumerate(self.blocks, start=1):
            block = cast(_WithinArrayBlock, module)
            rescale_parameter(block.out.weight.data, layer)
            rescale_parameter(block.mlp[3].weight.data, layer)

    def forward(
        self,
        x: torch.Tensor,
        region_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``(batch, n_arrays, n_tokens, d_model)`` array-packed tokens."""
        region = self.region_embed(region_ids)
        if region is not None:
            x = x + region.to(x.dtype)

        norms = self.norms_block
        levels = []
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, key_mask)
            if norms is not None and layer + 1 in _SUP_TAPS:
                levels.append(norms[_SUP_TAPS.index(layer + 1)](x))
        if self.norm_out is not None:
            return self.norm_out(x)
        return torch.cat(levels, dim=-1)


class _RegionIdentityEmbed(nn.Module):
    """Near-zero-initialized embedding of the atlas region, built only when
    ``enabled``."""

    def __init__(self, d_model: int, enabled: bool):
        super().__init__()
        self.embed = None
        if enabled:
            self.embed = nn.Embedding(_N_REGIONS, d_model)
            nn.init.trunc_normal_(self.embed.weight, std=_ADDITIVE_INIT_STD)

    def forward(self, region_ids: torch.Tensor) -> torch.Tensor | None:
        """Return the embedding of each token's region, or ``None`` if ablated."""
        return None if self.embed is None else self.embed(region_ids)


class _WithinArrayBlock(nn.Module):
    """Pre-norm block attending jointly over the contacts and time of one
    array."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=_LN_EPS)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model, eps=_LN_EPS)
        self.mlp = FeedForwardBlock(
            emb_size=d_model, expansion=mlp_ratio, drop_p=0.0, activation=activation
        )
        self.split_heads = Rearrange(
            "batch array seq (heads dim) -> batch array heads seq dim", heads=n_heads
        )
        self.merge_heads = Rearrange(
            "batch array heads seq dim -> batch array seq (heads dim)"
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        query = query * cos + rotate_pairs(query) * sin
        key = key * cos + rotate_pairs(key) * sin

        # Padded contacts are blocked as keys, so they cannot reach a real
        # token; their own rows are computed and then dropped by the caller.
        attention = F.scaled_dot_product_attention(
            query, key, value, attn_mask=key_mask
        )
        x = x + self.out(self.merge_heads(attention))
        return x + self.mlp(self.norm2(x))

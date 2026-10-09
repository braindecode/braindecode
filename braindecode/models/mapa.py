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

import re
from numbers import Integral

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import rescale_parameter, rotate_pairs, spectral_input
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import dkt_region_slots
from braindecode.modules import FeedForwardBlock

# Blocks whose output carries a deep-supervision norm, following V-JEPA 2.1.
_SUP_TAPS: tuple[int, ...] = (3, 6, 9, 12)

# Name, FFT length, first and last rfft bin (inclusive), and the decimation
# stride on the shared 32 Hz frame clock, in the order slow, mid, fast. At the
# reference's 2048 Hz (hop of 64 samples) the retained bins span 2-14 Hz,
# 16-56 Hz and 64-160 Hz; the largest stride is the slow band's frame quantum.
_BANDS: tuple[tuple[str, int, int, int, int], ...] = (
    ("slow", 1024, 1, 7, 8),
    ("mid", 256, 2, 7, 2),
    ("fast", 128, 4, 10, 1),
)
_BAND_BINS: tuple[int, ...] = tuple(k1 - k0 + 1 for _, _, k0, k1, _ in _BANDS)
_BAND_STRIDES: tuple[int, ...] = tuple(stride for *_, stride in _BANDS)
# The published "Guard 3" caps on the normalized inputs, per band.
_INPUT_CLIP_Z: tuple[float, float, float] = (15.0, 15.0, 20.0)

#: FreeSurfer DKT region names, in the slot order of MAPA's region table.
#:
#: The 62 hemisphere-qualified cortical parcels come first, then the 12
#: subcortical structures, which is the order that indexes the released region
#: embedding. Slot ``len(MAPA_DKT_REGIONS)`` is the reserved slot given to a
#: contact that falls outside every region. The vocabulary itself lives in
#: :func:`braindecode.models.util.dkt_region_slots`, whose names are the
#: FreeSurfer labels MNE ships in its colour table, so MAPA owns only the
#: reference to it, not the atlas data.
MAPA_DKT_REGIONS: tuple[str, ...] = dkt_region_slots()


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
       shared 32 Hz frame clock, robust z-scored and clipped (``frontend``).
    2. Embed each ``(contact, band, time)`` token with its band's linear layer
       and a per-band vector (``stem``).
    3. Add the embedding of each contact's DKT region, then apply twelve
       pre-norm transformer blocks whose attention spans contacts and time
       jointly but never crosses from one array to another (``encoder``).
    4. Pool the normed outputs of blocks 3, 6, 9 and 12 and classify them
       (``final_layer``).

    Encoding is threefold: temporal (the 32 Hz frame clock, one rotary axis
    shared by tokens on the same slot), spatial (the clinical contact number
    along the array, the other rotary axis, plus the additive region
    embedding; attention is block-diagonal over arrays and uses no coordinates)
    and spectral (the three STFT bands at 2-14, 16-56 and 64-160 Hz).

    .. rubric:: Additional Mechanisms

    *Electrode metadata.* The array and the contact number are read off the
    clinical label (``"LA7"`` is contact 7 of array ``LA``), from
    ``contact_labels`` or else the ``chs_info`` names, verbatim with gaps kept;
    regions must be exact names of :data:`MAPA_DKT_REGIONS`.

    *One model, many subjects.* The constructor's montage is only the default.
    To read another recording, pass :meth:`forward` the ``sensor_indices`` that
    :meth:`sensor_indices` builds from its labels and regions; all samples of a
    batch share them. Only ``pooling="flatten"`` is tied to one montage and one
    window length.

    *Sampling rate and window length.* Resample to 2048 Hz. A window yields
    ``1 + n_times // 64`` frames, truncated to a multiple of 8, so it needs at
    least 448 samples. With ``normalization="session"`` the input is the
    spectrogram itself (``sfreq=32``, ``n_times`` in frames): the 20 retained
    bins, robust z-scored per contact and bin over the whole recording.

    .. important::
       **Pre-trained Weights Available**

       The released ``mapa_vits384`` encoder (the default configuration,
       21,335,424 parameters) is hosted on the Hugging Face Hub. The head is
       randomly initialized, so fine-tune or linear-probe before use::

           from braindecode.models import MAPA

           model = MAPA.from_pretrained(
               "braindecode/mapa-pretrained",
               n_outputs=2,
               chs_info=raw.info["chs"],  # clinical labels, e.g. "LA7"
               regions=regions,
           )

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

    # TorchScript reads no module globals, so forward reads these constants.
    __constants__ = ("_band_strides", "_n_bins", "_n_regions")
    _band_strides = _BAND_STRIDES
    _n_bins = sum(_BAND_BINS)
    _n_regions = len(MAPA_DKT_REGIONS) + 1

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

        if d_model <= 0 or d_model % 64:  # head dimension 64
            raise ValueError(
                f"d_model must be a positive multiple of the head dimension 64, "
                f"got {d_model}."
            )
        if pooling not in ("mean", "flatten"):
            raise ValueError(f"pooling must be 'mean' or 'flatten', got {pooling!r}.")
        if normalization not in ("window", "session", "none"):
            raise ValueError(
                f"normalization must be 'window', 'session' or 'none', got "
                f"{normalization!r}."
            )
        # The bands are FFT bin slices at 2048 Hz on a 32 Hz frame clock, so
        # another rate would shift the bands and the token rate; session inputs
        # are already the 32 Hz spectrogram. sfreq is advisory for the
        # frontend, so it is validated only when it is known, and a wrong
        # value is a hard error rather than a silent warning.
        expected_sfreq = 32 if normalization == "session" else 2048
        sfreq_known = self._sfreq is not None or (
            self._input_window_seconds is not None and self._n_times is not None
        )
        if sfreq_known and float(self.sfreq) != expected_sfreq:
            raise ValueError(
                f"MAPA with normalization={normalization!r} needs sfreq="
                f"{expected_sfreq} Hz, got {self.sfreq} Hz."
            )

        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.region_embed = region_embed
        self.space_rope = space_rope
        self.deep_sup = deep_sup
        self.pooling = pooling
        self.normalization = normalization
        self.activation = activation

        self.n_frames = _frame_count(
            self.n_times, spectrogram=normalization == "session"
        )
        self.k_full = sum(self.n_frames // stride for *_, stride in _BANDS)

        self.frontend = _SpectrogramFrontend(normalization=normalization)
        self.stem = _PerBandStem(d_model=d_model, band_bins=_BAND_BINS)
        self.encoder = _Encoder(
            d_model=d_model,
            n_heads=d_model // 64,
            mlp_ratio=mlp_ratio,
            region_embed=region_embed,
            deep_sup=deep_sup,
            activation=activation,
        )
        # The montage resolved here is only the default: forward takes another
        # recording's metadata directly, which is what lets one instance read
        # subjects it was not built for. It rides along as a buffer, so it
        # follows the module across devices; the token layout is cheap and is
        # rebuilt from it at call time.
        if contact_labels is not None:
            if len(contact_labels) != self.n_chans:
                raise ValueError(
                    f"contact_labels has {len(contact_labels)} labels but the "
                    f"model has {self.n_chans} channels."
                )
            labels = [str(label) for label in contact_labels]
        elif self._chs_info:
            labels = [str(channel["ch_name"]) for channel in self.chs_info]
        else:
            labels = [f"A{contact + 1}" for contact in range(self.n_chans)]
        self.register_buffer(
            "default_sensor_indices",
            self.sensor_indices(labels, regions),
            persistent=False,
        )
        # Precompute the default montage's token layout as buffers, so it rides
        # along across devices and the common forward path stays export-stable
        # (the layout is rebuilt at call time only for a foreign montage).
        layout = self._token_layout(self.default_sensor_indices, self.n_frames)
        names = (
            "gather_idx",
            "key_mask",
            "token_region",
            "scatter_idx",
            "rope_cos",
            "rope_sin",
        )
        for name, tensor in zip(names, layout):  # zip drops the trailing k_full
            self.register_buffer(name, tensor, persistent=False)

        feature_dim = d_model * (len(_SUP_TAPS) if deep_sup else 1)
        n_features = (
            feature_dim
            if pooling == "mean"
            else feature_dim * self.n_chans * self.k_full
        )
        self.final_layer = nn.Linear(n_features, self.n_outputs)

    @property
    @torch.jit.unused
    def input_shape(self):  # 3-D, or 4-D for normalization="session"
        """Input data shape; ``normalization="session"`` takes the spectrogram."""
        if self.normalization == "session":
            return (1, self.n_chans, sum(_BAND_BINS), self.n_times)
        return super().input_shape

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
        # Array id by first appearance of the label prefix; contact number kept
        # verbatim, gaps included.
        seen: dict[str, int] = {}
        arrays, contacts = [], []
        for label in labels:
            match = re.match(r"^(.*?)(\d+)$", label)
            if match is None:
                raise ValueError(
                    f"MAPA reads the array and the contact number off the "
                    f"clinical label, but {label!r} has no trailing number. "
                    f"Pass clinical labels in contact_labels, and drop "
                    f"non-neural channels beforehand."
                )
            arrays.append(seen.setdefault(match.group(1), len(seen)))
            contacts.append(int(match.group(2)))

        # Region names must match MAPA_DKT_REGIONS exactly, so a misspelling
        # cannot silently become unassigned.
        unassigned = len(MAPA_DKT_REGIONS)
        if regions is None:
            slots = [unassigned] * len(labels)
        elif len(regions) != len(labels):
            raise ValueError(
                f"regions has {len(regions)} entries but there are "
                f"{len(labels)} channels."
            )
        else:
            lookup = {name: slot for slot, name in enumerate(MAPA_DKT_REGIONS)}
            slots = []
            for region in regions:
                if region is None:
                    slots.append(unassigned)
                elif isinstance(region, Integral) and not isinstance(region, bool):
                    if not 0 <= int(region) <= unassigned:
                        raise ValueError(
                            f"region slot {int(region)} is outside the "
                            f"{unassigned + 1} slots of MAPA's region table."
                        )
                    slots.append(int(region))
                elif isinstance(region, str) and region in lookup:
                    slots.append(lookup[region])
                else:
                    raise ValueError(
                        f"{region!r} is not a MAPA region. Names are the exact "
                        f"FreeSurfer DKT entries of MAPA_DKT_REGIONS, such as "
                        f"'ctx-lh-superiortemporal' or 'Left-Hippocampus'. Pass "
                        f"None for a contact that falls outside every region."
                    )
        return torch.stack(
            [
                torch.tensor(arrays, dtype=torch.long),
                torch.tensor(contacts, dtype=torch.long),
                torch.tensor(slots, dtype=torch.long),
            ],
            dim=1,
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
            return self.default_sensor_indices
        indices = torch.as_tensor(sensor_indices)
        n_regions = self._n_regions
        if (
            indices.is_floating_point()
            or indices.is_complex()
            or indices.dtype == torch.bool
            or indices.ndim != 2
            or indices.shape[0] != n_chans
            or indices.shape[1] != 3
            or bool((indices < 0).any())
            or bool((indices[:, 2] >= n_regions).any())
        ):
            raise ValueError(
                f"sensor_indices must hold integers of shape ({n_chans}, 3), one "
                f"row of (array, contact number, region slot) per channel, and "
                f"must be non-negative, with region slots below {n_regions}; got "
                f"{indices.dtype} of shape {list(indices.shape)}."
            )
        return indices.to(device=x.device, dtype=torch.long)

    def _rope(
        self, contact: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine and sine rotary tables, head dimension 64 split between the
        contact number and the frame-clock slot.

        The frequency grid is built on the input's device, so a foreign montage
        works on accelerators too; without ``space_rope`` the contact half is
        the identity rotation.
        """
        pairs = 64 // 4  # head dimension 64, a quarter per rotary pair-axis
        exponents = torch.arange(pairs, device=contact.device) / pairs
        # Ordinal contacts are dense, time slots are not, so the axes differ.
        contact_freq = 1.0 / (8.0**exponents)
        if not self.space_rope:
            contact_freq = torch.zeros_like(contact_freq)
        time_freq = 1.0 / (64.0**exponents)
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

    def _token_layout(
        self, indices: torch.Tensor, n_frames: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Regroup the channels into arrays padded to the largest one, and
        return the gather plan, padding mask, region ids, scatter plan, rotary
        tables and the per-contact token count of that grid."""
        device = indices.device
        n_chans = indices.shape[0]
        arrays, contacts, regions = indices.unbind(dim=1)

        # Renumber the arrays contiguously, so any labelling of them works.
        _, array_of_contact = torch.unique(arrays, return_inverse=True)
        counts = torch.bincount(array_of_contact)
        n_arrays, max_contacts = counts.numel(), int(counts.max())

        # Row s holds the contacts of array s in input order, padded with
        # contact 0, whose tokens are masked out of attention and dropped on
        # the way back.
        order = torch.argsort(array_of_contact, stable=True)
        row = array_of_contact[order]
        slot = (
            torch.arange(n_chans, device=device)
            - torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])[row]
        )
        gather_idx = torch.zeros(
            (n_arrays, max_contacts), dtype=torch.long, device=device
        )
        valid = torch.zeros((n_arrays, max_contacts), dtype=torch.bool, device=device)
        gather_idx[row, slot] = order
        valid[row, slot] = True

        # Position of each contact in the flattened, array-major token grid,
        # which undoes the gather after the encoder.
        scatter_idx = torch.empty(n_chans, dtype=torch.long, device=device)
        scatter_idx[order] = row * max_contacts + slot

        # Each contact carries the same block of tokens: the three bands in
        # order, each at its own rate on the shared clock.
        bands: list[torch.Tensor] = []
        for stride in self._band_strides:
            bands.append(torch.arange(n_frames // stride, device=device) * stride)
        lattice = torch.cat(bands)
        k_full = lattice.numel()
        cos, sin = self._rope(
            contact=contacts[gather_idx].repeat_interleave(k_full, dim=1),
            time=lattice.repeat(max_contacts).expand(n_arrays, -1),
        )
        return (
            gather_idx,
            # Inserted axes broadcast the mask and the tables over the batch,
            # the heads and the query length, keeping the key length last.
            valid.repeat_interleave(k_full, dim=1)[None, :, None, None],
            regions[gather_idx].repeat_interleave(k_full, dim=1),
            scatter_idx,
            cos[None, :, None],
            sin[None, :, None],
            k_full,
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the linear classification head for a new ``n_outputs``.

        The new head keeps the old one's device and dtype, and the value is
        validated and recorded through the mixin's :meth:`_set_n_outputs`.
        """
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(
            old.in_features,
            n_outputs,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        sensor_indices: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Encode an iEEG batch into class logits or pooled features.

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
            Whether to return the pooled token embedding instead of the logits.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or with
            ``return_features=True`` a dict whose ``"features"`` entry is the
            pooled embedding of shape ``(batch, final_layer.in_features)`` and whose
            ``"cls_token"`` entry is ``None``.
        """
        spectrogram = self.normalization == "session"
        n_bins = self._n_bins
        if x.ndim != (4 if spectrogram else 3) or (
            spectrogram and x.shape[2] != n_bins
        ):
            raise ValueError(
                f"normalization='session' takes a spectrogram of shape (batch, "
                f"n_chans, {n_bins}, n_frames), any other a raw signal of shape "
                f"(batch, n_chans, n_times), but got {list(x.shape)} with "
                f"normalization='{self.normalization}'."
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
        # The construction-time montage reads its precomputed layout.
        gather_idx, key_mask, token_region, scatter_idx, cos, sin, k_full = (
            (
                self.gather_idx,
                self.key_mask,
                self.token_region,
                self.scatter_idx,
                self.rope_cos,
                self.rope_sin,
                self.k_full,
            )
            if indices is self.default_sensor_indices and n_frames == self.n_frames
            else self._token_layout(indices, n_frames)
        )

        tokens = self.stem(self.frontend(x))
        # (batch, n_arrays, max_contacts * k_full, d_model), array-contiguous.
        packed = tokens[:, gather_idx].flatten(2, 3)
        encoded = self.encoder(packed, token_region, cos, sin, key_mask)
        # Back to one block of tokens per channel, in the input channel order.
        encoded = encoded.unflatten(2, (-1, k_full)).flatten(1, 2)
        encoded = encoded[:, scatter_idx]

        if self.pooling == "mean":
            features = encoded.mean(dim=(1, 2))
        else:
            features = encoded.flatten(1)
        logits = self.final_layer(features)

        if return_features:
            out: dict[str, torch.Tensor | None] = {
                "features": features,
                "cls_token": None,  # nosec B105
            }
            return out
        return logits


def _frame_count(
    n_times: int,
    spectrogram: bool = False,
    # The slow band's token rate, a default since TorchScript reads no globals.
    quantum: int = max(_BAND_STRIDES),
) -> int:
    """Usable frames of the 32 Hz clock in ``n_times`` samples (or frames)."""
    frames = n_times if spectrogram else 1 + n_times // 64  # hop of 64 samples
    n_frames = (frames // quantum) * quantum
    if n_frames < quantum:
        minimum = (
            f"{quantum} frames" if spectrogram else f"{(quantum - 1) * 64} samples"
        )
        raise ValueError(
            f"MAPA needs a window of at least {minimum}, one token of the slow "
            f"band, but got {n_times}."
        )
    return n_frames


def _vjepa_init(module: nn.Module) -> None:
    """V-JEPA 2 initialization of linear layers and layer norms."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class _SpectrogramFrontend(nn.Module):
    """Per-band STFT magnitudes on a shared 32 Hz clock, robust z-scored
    (window), capped and decimated to each band's token rate."""

    # TorchScript reads no module globals, so forward reads these constants.
    __constants__ = ("bands", "band_bins", "band_strides", "clip_z")
    bands = _BANDS
    band_bins = _BAND_BINS
    band_strides = _BAND_STRIDES
    clip_z = _INPUT_CLIP_Z

    def __init__(self, normalization: str):
        super().__init__()
        self.normalization = normalization

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return ``(batch, n_chans, n_bins, n_tokens)`` slow, mid, fast bands."""
        if self.normalization == "session":
            n_frames = _frame_count(x.shape[-1], spectrogram=True)
            bands = x[..., :n_frames].split(self.band_bins, dim=2)
        else:
            bands = self._stft_bands(x)
        out: list[torch.Tensor] = []
        for i, (cap, stride) in enumerate(zip(self.clip_z, self.band_strides)):
            out.append(bands[i].clamp(-cap, cap)[..., ::stride])
        return out

    def _stft_bands(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``(batch, n_chans, n_times)`` to per-band magnitudes, robust z-scored
        on the window; the frames past the true window are dropped."""
        batch, n_chans = x.shape[0], x.shape[1]
        n_frames = _frame_count(x.shape[-1])
        waveform = spectral_input(x.reshape(batch * n_chans, x.shape[-1]))
        bands: list[torch.Tensor] = []
        for _, n_fft, k0, k1, _ in self.bands:
            # A window shorter than the transform is zero-padded, as in the
            # reference, so the centred transform has something to reflect.
            padded = waveform
            if padded.shape[-1] < n_fft:
                padded = F.pad(padded, (0, n_fft - padded.shape[-1]))
            spectrum = torch.stft(
                padded,
                n_fft=n_fft,
                hop_length=64,  # 2048 Hz / 32 Hz frame clock
                win_length=n_fft,
                window=torch.hann_window(
                    n_fft, device=waveform.device, dtype=waveform.dtype
                ),
                center=True,
                normalized=False,
                return_complex=True,
            )
            band = spectrum[:, k0 : k1 + 1, :n_frames].abs()
            if self.normalization == "window":
                # Robust z-score per contact and bin over time: median centre,
                # MAD scale (x1.4826), zero the sub-floor bins.
                median = band.median(dim=-1, keepdim=True).values
                sigma = (
                    1.4826 * (band - median).abs().median(dim=-1, keepdim=True).values
                )
                z = (band - median) / sigma.clamp(min=1e-6)
                band = torch.where(sigma >= 1e-6, z, torch.zeros_like(z))
            bands.append(band.reshape(batch, n_chans, k1 - k0 + 1, n_frames).to(x))
        return bands


class _PerBandStem(nn.Module):
    """One linear projection plus an additive vector per band, with no per-band
    norm, which would bring back the 1/f the z-score removes."""

    def __init__(self, d_model: int, band_bins: tuple[int, ...]):
        super().__init__()
        self.projs = nn.ModuleList(nn.Linear(n_bins, d_model) for n_bins in band_bins)
        self.band_type_emb = nn.Parameter(torch.empty(len(band_bins), d_model))
        self.projs.apply(_vjepa_init)
        # Near-zero: the band vector is added to the residual stream.
        nn.init.trunc_normal_(self.band_type_emb, std=1e-6)

    def forward(self, bands: list[torch.Tensor]) -> torch.Tensor:
        """Return ``(batch, n_chans, k_full, d_model)`` tokens."""
        tokens: list[torch.Tensor] = []
        embeddings = self.band_type_emb.unbind(0)
        for i, proj in enumerate(self.projs):
            tokens.append(proj(bands[i].transpose(-1, -2)) + embeddings[i])
        return torch.cat(tokens, dim=-2)


class _Encoder(nn.Module):
    """Region embedding plus a pre-norm stack of within-array blocks."""

    __constants__ = ("sup_taps",)  # TorchScript reads no module globals
    sup_taps = _SUP_TAPS

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
        # ``region_embed.embed`` keeps the released key; an empty container is
        # the ``no_region`` ablation. Near-zero: added to the residual stream.
        self.region_embed = nn.ModuleDict(
            {"embed": nn.Embedding(len(MAPA_DKT_REGIONS) + 1, d_model)}
            if region_embed
            else {}
        )
        if region_embed:
            nn.init.trunc_normal_(self.region_embed["embed"].weight, std=1e-6)
        self.blocks = nn.ModuleList(
            _WithinArrayBlock(d_model, n_heads, mlp_ratio, activation)
            for _ in range(12)
        )
        # The deepest tap's norm is the terminal norm, so there is no separate
        # one when the taps are used.
        self.norms_block = (
            nn.ModuleList(nn.LayerNorm(d_model, eps=1e-6) for _ in _SUP_TAPS)
            if deep_sup
            else None
        )
        self.norm_out = None if deep_sup else nn.LayerNorm(d_model, eps=1e-6)
        self.apply(_vjepa_init)
        for layer, block in enumerate(self.blocks, start=1):
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
        if hasattr(self.region_embed, "embed"):  # static under TorchScript
            x = x + self.region_embed["embed"](region_ids).to(x.dtype)
        levels: list[torch.Tensor] = []
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, key_mask)
            if self.norms_block is not None:
                for tap, norm in zip(self.sup_taps, self.norms_block):
                    if layer + 1 == tap:
                        levels.append(norm(x))
        if self.norm_out is not None:
            return self.norm_out(x)
        return torch.cat(levels, dim=-1)


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
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
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

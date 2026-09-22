# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: BSD (3-clause)
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

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin

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

MAPA_DKT_REGIONS: tuple[str, ...] = tuple(
    f"ctx-{hemisphere}-{parcel}"
    for hemisphere in ("lh", "rh")
    for parcel in _DKT_CORTICAL_PARCELS
) + tuple(
    f"{hemisphere}-{structure}"
    for hemisphere in ("Left", "Right")
    for structure in _DKT_SUBCORTICAL_STRUCTURES
)
"""FreeSurfer DKT region names, in the slot order of MAPA's region table.

The 62 hemisphere-qualified cortical parcels come first, then the 12
subcortical structures, which is the order that indexes the released region
embedding. Slot ``len(MAPA_DKT_REGIONS)`` is the reserved slot given to a
contact that falls outside every region.
"""

_N_REGIONS = len(MAPA_DKT_REGIONS) + 1
_UNASSIGNED_REGION = len(MAPA_DKT_REGIONS)

# Clinical electrode labels are an array name followed by a contact number.
_LABEL_PATTERN = re.compile(r"^(.*?)(\d+)$")


class MAPA(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""MAPA from Tang, Spalding and Cogan (2026) [Tang2026]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`
    :bdg-dark-line:`Channel`

    .. versionadded:: 1.8.2

    A masked-autoencoder foundation model for intracranial EEG whose defining
    feature is that everything it is told about an electrode is anatomical:
    the atlas region the contact falls in, and its ordinal position along the
    array it was implanted on. Stereotactic coordinates, montage size and
    channel order never reach the model. That is what lets one pretrained
    encoder read a subject it has never seen, which is the paper's headline
    result: a linear probe on frozen MAPA features needs about 164 labelled
    trials to reach the accuracy that takes 3500 trials without pretraining
    [Tang2026]_.

    .. rubric:: Architecture Overview

    A token is a ``(contact, band, time)`` triple. Each channel is turned into
    three magnitude spectrograms -- a slow, a mid and a fast band -- that share
    one 32 Hz frame clock but are decimated to their own token rates, so one
    slow token spans the eight frames that carry eight fast tokens. Every token
    is projected by its band's own linear layer, tagged by a learned per-band
    embedding, and offset by the learned embedding of its contact's atlas
    region. Twelve identical pre-norm transformer blocks then attend over
    contacts and time *jointly*, but only inside one array: attention never
    crosses from one implanted array to another, because which contacts share
    an array is a fact about the surgery rather than about the brain. Position
    enters the attention as a two-axis rotary encoding, one axis for the
    clinical contact number and one for the frame clock, so scores depend only
    on differences along an array and on elapsed time.

    .. rubric:: Macro Components

    - **Frontend** (``MAPA.frontend``). *Operations:* three
      Hann-window short-time Fourier transforms of the raw window, with FFT
      lengths 1024, 256 and 128 and a shared hop of 64 samples; keep the
      magnitude of an inclusive bin slice per band; robust z-score each contact
      and bin; clip to the published caps; decimate each band by its own
      stride. *Role:* produce the three normalized spectrograms the released
      encoder consumes, from the raw voltage that braindecode passes around.
    - **Per-band stem** (``MAPA.stem``). *Operations:* one weight-shared linear
      layer per band maps that band's frequency bins to ``d_model``, and a
      learned per-band vector is added. *Role:* embed a patch while keeping the
      three bands distinguishable, with deliberately no frequency embedding and
      no per-band normalization, either of which would restore the :math:`1/f`
      dominance the robust z-score removes.
    - **Region embedding** (``MAPA.encoder.region_embed``). *Operations:* look
      up one learned vector per DKT region and add it to every token of the
      contacts in that region, once, at the stack input. *Role:* the anatomical
      identity that means the same thing in an unseen subject; it rides the
      residual stream into every block, so no block injects it again.
    - **Encoder** (``MAPA.encoder.blocks``). *Operations:* twelve pre-norm
      blocks of within-array multi-head self-attention, with a two-axis rotary
      encoding on the contact number and the frame clock, and a GELU
      feed-forward block of ratio ``mlp_ratio``. *Role:* mix contacts and time
      inside each array.
    - **Read-out** (``MAPA.encoder.norms_block``, ``MAPA.final_layer``).
      *Operations:* normalize the output of blocks 3, 6, 9 and 12 with their
      own affine LayerNorm and concatenate them, then pool the token axes and
      apply a linear classifier. *Role:* produce the class logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* a shared 32 Hz frame clock, on which the slow, mid and fast
      bands carry one token every 8, 2 and 1 frames. Tokens of different bands
      that land on the same lattice slot get the same rotary phase, so band
      mixing aligns them in physical time.
    - *Spatial:* the clinical contact number along the array, as one axis of
      the rotary encoding, plus the additive region embedding. Attention is
      block-diagonal over arrays. No coordinates are used anywhere.
    - *Spectral:* explicit, and the only place the signal is band-limited: the
      three STFT bands are the model's input representation.

    .. rubric:: Electrode metadata

    MAPA needs two things per channel that braindecode does not carry as such.

    The first is the *array* a contact belongs to and its *number* along that
    array, both of which are read off the clinical label: ``"LA7"`` is contact
    7 of array ``LA``. Labels are taken from ``contact_labels`` when given and
    from the ``chs_info`` channel names otherwise. Contact numbers are kept
    verbatim, gaps included, so that dropping a bad contact leaves its
    neighbours two apart rather than renumbering them: only differences along
    an array are ever used, but they must be the real ones, and the direction
    of the numbering must be the recording's. A label with no trailing number
    is rejected, since it has no canonical position. When neither
    ``contact_labels`` nor ``chs_info`` is available the channels are treated
    as one array numbered from 1, which keeps the model constructible from
    ``n_chans`` alone but encodes a montage that is almost certainly not the
    real one.

    The second is the atlas region, passed in ``regions``. This is an
    anatomical lookup that braindecode does not perform, so it must be supplied
    to get the transferable spatial prior; channels default to the reserved
    "outside every region" slot, which is also what ``None`` selects
    explicitly. Region names must be exact entries of
    :data:`MAPA_DKT_REGIONS`, as the reference rejects near-misses rather than
    letting a misspelling become silently unassigned.

    .. rubric:: Sampling frequency and window length

    The band definitions are FFT bin slices at 2048 Hz, so a recording at
    another rate puts different frequencies in each band and moves the frame
    clock off 32 Hz; resample to 2048 Hz, as the reference does, rather than
    relying on the warning this model emits. A window yields
    ``1 + n_times // 64`` frames, truncated to a multiple of 8 so the slow band
    holds a whole number of tokens, which needs at least 448 samples.

    .. rubric:: Pre-trained weights

    Four checkpoints are published, all pretrained on Brain Treebank: the
    released ``mapa_vits384`` and the three spatial-encoding ablations, which
    this port exposes as ``region_embed`` and ``space_rope``. They are not
    downloaded here, but they transfer as-is: ``MAPA.stem`` and ``MAPA.encoder``
    reproduce the reference module names, so the released
    ``checkpoint["model"]`` loads with :meth:`~torch.nn.Module.load_state_dict`
    under ``strict=False``, leaving only ``final_layer`` uninitialized. The
    region table is indexed by the atlas rather than by a subject, so it
    transfers along with the rest.

    .. note::
        Parameter counts match the released checkpoints exactly: 21,335,424
        for the default configuration, excluding the classification head.

        The reference runs attention through a jagged nested tensor on GPU and
        a materialized block-diagonal mask on CPU, packing a whole session into
        one ragged sequence. This port gathers the contacts of each array into
        a padded array axis and masks the padding, which is equivalent because
        the mask only ever blocks attention across arrays, and lets the whole
        batch run through
        :func:`~torch.nn.functional.scaled_dot_product_attention` unchanged.

        The reference fits its robust z-score on a whole recording and then
        slices windows out of the normalized spectrogram. A braindecode model
        sees one window at a time, so ``normalization="window"`` fits the same
        median and scaled median absolute deviation on the window itself. This
        is the one place where this port cannot reproduce the reference
        numerically: for windows of a second or so the statistics come from a
        few dozen frames rather than a whole session. Normalize upstream and
        pass ``normalization="none"`` to feed the reference's own inputs.

        The features this model pools are the encoder's own output, the
        concatenation of the four normed deep-supervision taps. The paper's
        frozen evaluation instead reads block 12 straight off the residual
        stream, before that norm, and fits a ridge probe on every token of
        every contact, which ``pooling="flatten"`` reproduces up to the choice
        of regularizer.

        The masked autoencoding objective, its decoder, the anatomical
        localization and the artifact detectors ("Guard 1" and "Guard 2") are
        out of scope: this port is the frozen encoder and a classification
        head. The input clipping ("Guard 3") is part of the model and is kept.

    Parameters
    ----------
    contact_labels : list of str, optional
        Clinical label of each channel, such as ``"LA7"``, from which the array
        and the contact number are read. Defaults to the ``chs_info`` channel
        names, then to a single array numbered from 1.
    regions : list of str or int or None, optional
        DKT region of each channel, either an exact name from
        :data:`MAPA_DKT_REGIONS` or its integer slot, with ``None`` selecting
        the reserved unassigned slot. Defaults to unassigned everywhere.
    d_model : int
        Token embedding dimension, a multiple of 64, which is the head
        dimension held fixed across the released widths. Default 384, the
        released ``mapa_vits384``.
    mlp_ratio : int
        Hidden dimension of the feed-forward blocks, as a multiple of
        ``d_model``.
    region_embed : bool
        Whether the region embedding is used. ``False`` is the paper's
        ``no_region`` ablation, and builds no table at all.
    space_rope : bool
        Whether the rotary encoding carries the contact number. ``False`` is
        the paper's ``no_relpos`` ablation, which leaves attention
        permutation-invariant over the contacts of an array.
    deep_sup : bool
        Whether the encoder returns the concatenation of the four
        deep-supervision taps, of width ``4 * d_model``, or a single terminal
        LayerNorm of width ``d_model``.
    pooling : {"mean", "flatten"}
        Token aggregation before the head. ``"mean"`` averages the tokens,
        giving a head that depends on neither the montage nor the window
        length; ``"flatten"`` keeps every token, as the paper's frozen linear
        probe does, at the cost of a head that grows with both.
    normalization : {"window", "none"}
        Whether the spectrograms are robust z-scored on each window, or passed
        to the stem as they come because they were normalized upstream.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks, default
        :class:`~torch.nn.GELU`.

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
        if normalization not in ("window", "none"):
            raise ValueError(
                f"normalization must be 'window' or 'none', got {normalization!r}."
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
        if sfreq is not None and not math.isclose(sfreq, _SAMPLE_RATE):
            warnings.warn(
                f"MAPA's frequency bands and its 32 Hz frame clock are defined "
                f"at {_SAMPLE_RATE} Hz, but sfreq is {sfreq} Hz, so the bands "
                f"cover other frequencies and the tokens another rate. "
                f"Resample the recording to {_SAMPLE_RATE} Hz.",
                UserWarning,
            )

        self.n_frames = ((1 + self.n_times // _HOP) // _FRAME_QUANTUM) * _FRAME_QUANTUM
        if self.n_frames < _FRAME_QUANTUM:
            raise ValueError(
                f"MAPA needs a window of at least {(_FRAME_QUANTUM - 1) * _HOP} "
                f"samples, which is {_FRAME_QUANTUM} frames of the 32 Hz clock "
                f"and one token of the slow band, but got {self.n_times}."
            )
        self.band_lengths = tuple(self.n_frames // stride for *_, stride in _BANDS)
        self.k_full = sum(self.band_lengths)

        self.frontend = _SpectrogramFrontend(
            n_frames=self.n_frames, normalization=normalization
        )
        self.stem = _PerBandStem(
            d_model=d_model, band_bins=tuple(k1 - k0 + 1 for _, _, k0, k1, _ in _BANDS)
        )
        self.encoder = _Encoder(
            d_model=d_model,
            n_heads=d_model // _HEAD_DIM,
            mlp_ratio=mlp_ratio,
            region_embed=region_embed,
            deep_sup=deep_sup,
            activation=activation,
        )

        self._register_geometry(contact_labels, regions)

        feature_dim = d_model * (len(_SUP_TAPS) if deep_sup else 1)
        n_features = (
            feature_dim
            if pooling == "mean"
            else feature_dim * self.n_chans * self.k_full
        )
        self.final_layer = nn.Linear(n_features, self.n_outputs)

    def _register_geometry(
        self,
        contact_labels: list[str] | None,
        regions: list[str | int | None] | None,
    ) -> None:
        """Lay the token grid out per array and cache what attention reads.

        The layout depends only on the montage and the window length, so the
        gather plan, the rotary tables, the padding mask and the region of each
        token are all built once here and reused for every batch.
        """
        labels = self._resolve_labels(contact_labels)
        array_of_contact, depth_of_contact = _parse_contact_labels(labels)
        region_of_contact = _resolve_regions(regions, self.n_chans)

        members: list[list[int]] = [
            [
                contact
                for contact, array in enumerate(array_of_contact.tolist())
                if array == group
            ]
            for group in range(int(array_of_contact.max()) + 1)
        ]
        n_arrays = len(members)
        max_contacts = max(len(group) for group in members)

        # Row s holds the contacts of array s, padded with contact 0, whose
        # tokens are masked out of attention and dropped on the way back.
        gather_idx = torch.zeros((n_arrays, max_contacts), dtype=torch.long)
        valid = torch.zeros((n_arrays, max_contacts), dtype=torch.bool)
        for array, group in enumerate(members):
            gather_idx[array, : len(group)] = torch.tensor(group, dtype=torch.long)
            valid[array, : len(group)] = True

        # Each contact carries the same block of tokens: the three bands in
        # order, each at its own rate on the shared clock.
        lattice = torch.cat(
            [
                torch.arange(length) * stride
                for length, (*_, stride) in zip(self.band_lengths, _BANDS)
            ]
        )
        self.register_buffer("gather_idx", gather_idx, persistent=False)
        self.register_buffer(
            "token_valid", valid.repeat_interleave(self.k_full, dim=1), persistent=False
        )
        self.register_buffer(
            "token_region",
            region_of_contact[gather_idx].repeat_interleave(self.k_full, dim=1),
            persistent=False,
        )
        # Position of each contact in the flattened, array-major token grid,
        # which undoes the gather after the encoder.
        scatter_idx = torch.empty(self.n_chans, dtype=torch.long)
        scatter_idx[gather_idx[valid]] = torch.arange(n_arrays * max_contacts)[
            valid.flatten()
        ]
        self.register_buffer("scatter_idx", scatter_idx, persistent=False)

        cos, sin = _rotary_table(
            contact=depth_of_contact[gather_idx].repeat_interleave(self.k_full, dim=1),
            time=lattice.repeat(max_contacts).expand(n_arrays, -1),
            head_dim=_HEAD_DIM,
            space_rope=self.space_rope,
        )
        # Inserted axes broadcast the tables over the batch and the heads.
        self.register_buffer("rope_cos", cos[None, :, None], persistent=False)
        self.register_buffer("rope_sin", sin[None, :, None], persistent=False)

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

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Encode an iEEG batch into class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            Whether to also return the pooled token embedding.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_outputs)``.
        """
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"MAPA was built for {self.n_chans} channels but got input "
                f"with {x.shape[1]}; rebuild the model for this montage."
            )
        # The rotary tables, the padding mask and (for pooling="flatten") the
        # read-out are all built for a fixed token count, so reject a different
        # window outright rather than silently encoding a different grid.
        # Windows that differ only in frames the frontend drops are accepted.
        if ((1 + x.shape[-1] // _HOP) // _FRAME_QUANTUM) * _FRAME_QUANTUM != (
            self.n_frames
        ):
            raise ValueError(
                f"MAPA was built for {self.n_frames} frames of the 32 Hz clock "
                f"but got input with {x.shape[-1]} samples; rebuild the model "
                f"for this window length."
            )

        tokens = self.stem(self.frontend(x))
        # (batch, n_arrays, max_contacts * k_full, d_model), array-contiguous.
        packed = tokens[:, self.gather_idx].flatten(2, 3)
        encoded = self.encoder(
            packed,
            self.token_region,
            self.rope_cos,
            self.rope_sin,
            self.token_valid[None, :, None, None],
        )
        # Back to one block of tokens per channel, in the input channel order.
        encoded = encoded.unflatten(2, (-1, self.k_full)).flatten(1, 2)
        encoded = encoded[:, self.scatter_idx]

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


def _parse_contact_labels(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the array and the contact number off each clinical label.

    Arrays are numbered in order of first appearance, and two labels sharing a
    prefix share an array. The trailing number is kept verbatim, gaps included,
    because only differences along an array are used and those differences are
    the point.

    Parameters
    ----------
    labels : list of str
        Clinical label of each channel.

    Returns
    -------
    tuple of torch.Tensor
        ``(n_chans,)`` long tensors of array ids and contact numbers.
    """
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
    """Resolve the atlas region of each channel to a slot of the region table.

    Names must be exact entries of :data:`MAPA_DKT_REGIONS`: the reference
    rejects anything else rather than letting a misspelling become silently
    unassigned, since an unassigned contact is a meaningful state of its own.

    Parameters
    ----------
    regions : list of str or int or None, optional
        Region of each channel, or ``None`` for an unassigned montage.
    n_chans : int
        Number of channels the model was built for.

    Returns
    -------
    torch.Tensor
        ``(n_chans,)`` long tensor of slots.
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
    """Median-centre and MAD-scale each contact and frequency bin over time.

    A bin whose scale falls under the floor is constant to numerical precision,
    so it is zeroed rather than amplified.
    """
    median = x.median(dim=-1, keepdim=True).values
    sigma = _MAD_TO_SIGMA * (x - median).abs().median(dim=-1, keepdim=True).values
    z = (x - median) / sigma.clamp(min=_SIGMA_FLOOR)
    return torch.where(sigma >= _SIGMA_FLOOR, z, torch.zeros_like(z))


def _rotary_table(
    contact: torch.Tensor, time: torch.Tensor, head_dim: int, space_rope: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the two-axis rotary table of a token grid.

    The head dimension is split evenly between the position along the array and
    the position on the frame clock, each on the standard rotary schedule.
    Rotation is by absolute position, but the query-key score depends only on
    the difference, which is the property that transfers: absolute position
    along an array is not shared across subjects, only the ordering is.

    Parameters
    ----------
    contact : torch.Tensor
        Clinical contact number of each token.
    time : torch.Tensor
        Frame-clock position of each token.
    head_dim : int
        Attention head dimension, a multiple of 4.
    space_rope : bool
        Whether the contact axis carries position. When ``False`` its
        frequencies are zeroed, which makes the contact half the identity
        rotation and leaves attention permutation-invariant over the contacts
        of an array, without touching the time half.

    Returns
    -------
    tuple of torch.Tensor
        The cosine and sine tables, of the grid's shape with ``head_dim``
        appended.
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate within each adjacent pair of features."""
    paired = x.unflatten(-1, (-1, 2))
    return torch.stack([-paired[..., 1], paired[..., 0]], dim=-1).flatten(-2)


def _init_transformer_weights(module: nn.Module) -> None:
    """Initialize one module the way V-JEPA 2 does.

    Linear weights are truncated normal with zero bias, layer norms are the
    identity, and embeddings are left alone: the two tables of this model are
    added to the residual stream and start near zero instead.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=_INIT_STD)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class _SpectrogramFrontend(nn.Module):
    """Take a raw window to the three bands the released encoder consumes.

    One Hann short-time Fourier transform per band, all sharing a hop of 64
    samples so that they land on one 32 Hz frame clock, of which each band
    keeps an inclusive slice of rfft bins. The magnitudes are robust z-scored
    per contact and bin, bounded by the published caps, and decimated to the
    band's own token rate.

    Parameters
    ----------
    n_frames : int
        Number of frames of the shared clock the window yields.
    normalization : {"window", "none"}
        Whether the magnitudes are robust z-scored on each window, or left as
        they come because they were normalized upstream.
    """

    def __init__(self, n_frames: int, normalization: str):
        super().__init__()
        self.n_frames = n_frames
        self.normalization = normalization
        for name, n_fft, *_ in _BANDS:
            self.register_buffer(
                f"window_{name}", torch.hann_window(n_fft), persistent=False
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Take ``(batch, n_chans, n_times)`` to one tensor per band.

        Returns ``(batch, n_chans, n_bins, n_tokens)`` tensors in the fixed
        slow, mid, fast order.
        """
        batch, n_chans = x.shape[0], x.shape[1]
        waveform = x.reshape(batch * n_chans, x.shape[-1])
        bands = []
        for (name, n_fft, k0, k1, stride), cap in zip(_BANDS, _INPUT_CLIP_Z):
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
            band = spectrum[:, k0 : k1 + 1, : self.n_frames].abs()
            if self.normalization == "window":
                band = _robust_z(band)
            bands.append(
                band.clamp(-cap, cap)[..., ::stride].reshape(
                    batch, n_chans, k1 - k0 + 1, -1
                )
            )
        return bands


class _PerBandStem(nn.Module):
    """Project the three spectrogram bands into one token sequence per channel.

    Each band keeps its own linear layer, whose separate weights are what
    identify it, plus an additive per-band vector. There is deliberately no
    frequency embedding and no per-band normalization: a per-band norm would
    reintroduce the within-band :math:`1/f` dominance that the robust z-score
    removes.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    band_bins : tuple of int
        Number of frequency bins of each band.
    """

    def __init__(self, d_model: int, band_bins: tuple[int, ...]):
        super().__init__()
        self.projs = nn.ModuleList(nn.Linear(n_bins, d_model) for n_bins in band_bins)
        self.band_type_emb = nn.Parameter(torch.empty(len(band_bins), d_model))
        self.projs.apply(_init_transformer_weights)
        nn.init.trunc_normal_(self.band_type_emb, std=_ADDITIVE_INIT_STD)

    def forward(self, bands: list[torch.Tensor]) -> torch.Tensor:
        """Embed ``(batch, n_chans, n_bins, n_tokens)`` bands into one block.

        Returns ``(batch, n_chans, k_full, d_model)``, the bands concatenated
        in their fixed slow, mid, fast order.
        """
        tokens = [
            proj(band.transpose(-1, -2)) + embedding
            for band, proj, embedding in zip(bands, self.projs, self.band_type_emb)
        ]
        return torch.cat(tokens, dim=-2)


class _Encoder(nn.Module):
    """Pre-norm stack of within-array attention blocks.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden dimension of the feed-forward blocks, as a multiple of
        ``d_model``.
    region_embed : bool
        Whether the region table is built at all.
    deep_sup : bool
        Whether the output is the concatenation of the deep-supervision taps.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks.
    """

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
        self.blocks = nn.ModuleList(
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
        self._rescale_blocks()

    def _rescale_blocks(self) -> None:
        """Damp the residual branches with depth, so their variance stays flat."""
        for layer, block in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (layer + 1))
            block.out.weight.data.div_(scale)
            block.mlp.fc2.weight.data.div_(scale)

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
    """Learned embedding of the atlas region a contact falls in.

    The vocabulary is fixed by the atlas rather than learned per subject, which
    is what lets the table transfer to an unseen montage. It starts near zero
    so the model grows into it instead of carrying a strong per-region prior
    from the first step, and it is built at all only when it is used, so an
    ablated model has no dead parameter to weight-decay.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    enabled : bool
        Whether the table is built. When it is not, the module contributes
        nothing.
    """

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
    """Pre-norm block whose attention stays inside one array.

    A token attends to every other token of its own array, over contacts and
    time together rather than over each axis in turn, and to nothing outside
    it: which contacts share an array is a fact about where the surgeon placed
    the electrodes, not about the brain. Region identity is added once at the
    stack input and rides the residual, so this block injects nothing of its
    own.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden dimension of the feed-forward block, as a multiple of
        ``d_model``.
    activation : type[nn.Module]
        Activation layer class of the feed-forward block.
    """

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
        self.mlp = _FeedForward(
            d_model=d_model, mlp_ratio=mlp_ratio, activation=activation
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

        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        # Padded contacts are blocked as keys, so they cannot reach a real
        # token; their own rows are computed and then dropped by the caller.
        attention = F.scaled_dot_product_attention(
            query, key, value, attn_mask=key_mask
        )
        x = x + self.out(self.merge_heads(attention))
        return x + self.mlp(self.norm2(x))


class _FeedForward(nn.Module):
    """Two-layer feed-forward block with a GELU in between.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    mlp_ratio : int
        Hidden dimension, as a multiple of ``d_model``.
    activation : type[nn.Module]
        Activation layer class.
    """

    def __init__(self, d_model: int, mlp_ratio: int, activation: type[nn.Module]):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

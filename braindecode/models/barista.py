# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: see the "License" rubric of the class docstring
"""BaRISTA: brain-scale informed spatiotemporal representations of iEEG.

Reimplementation of BaRISTA (Oganesian, Hashemi & Shanechi, 2025), "BaRISTA:
Brain Scale Informed Spatiotemporal Representation of Human Intracranial Neural
Activity". The architecture is transcribed from the authors' reference
implementation (https://github.com/ShanechiLab/BaRISTA), whose dilated
convolutional temporal encoder is in turn adapted from TS2Vec and SimTS. The
reference is released by the University of Southern California under an
academic, non-commercial license, which this file inherits; see the "License"
rubric of :class:`BaRISTA` before redistributing or using it commercially.

Original Authors: Oganesian, Hashemi & Shanechi, University of Southern California
Braindecode Adaptation: Julien Gadonneix
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import apply_rotary, rotary_positional_encoding
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import GLUFeedForward, PatchTokenizer

# The reference RMSNorm eps, from ``config/model.yaml``.
_NORM_EPS = 1e-8

# Destrieux parcels, in the slot order of the reference ``Destrieux`` enum: this
# is the subset of the atlas that Brain Treebank annotates, plus the subcortical
# structures the dataset labels, so it is neither the full atlas nor alphabetical.
# The order is what indexes the released parcel-level embedding table, and slot 0
# is the padding slot given to electrodes of unknown parcel.
_DESTRIEUX_REGIONS: tuple[str, ...] = (
    "UNKNOWN",
    "LEFT_AMYGDALA",
    "LEFT_HIPPOCAMPUS",
    "LEFT_INF_LAT_VENT",
    "LEFT_PUTAMEN",
    "RIGHT_AMYGDALA",
    "RIGHT_HIPPOCAMPUS",
    "RIGHT_INF_LAT_VENT",
    "RIGHT_PUTAMEN",
    "CTX_LH_G_INS_LG_AND_S_CENT_INS",
    "CTX_LH_G_AND_S_CINGUL_ANT",
    "CTX_LH_G_AND_S_CINGUL_MID_ANT",
    "CTX_LH_G_AND_S_CINGUL_MID_POST",
    "CTX_LH_G_AND_S_SUBCENTRAL",
    "CTX_LH_G_CINGUL_POST_DORSAL",
    "CTX_LH_G_FRONT_INF_OPERCULAR",
    "CTX_LH_G_FRONT_INF_ORBITAL",
    "CTX_LH_G_FRONT_INF_TRIANGUL",
    "CTX_LH_G_FRONT_MIDDLE",
    "CTX_LH_G_FRONT_SUP",
    "CTX_LH_G_INSULAR_SHORT",
    "CTX_LH_G_OC_TEMP_MED_PARAHIP",
    "CTX_LH_G_OCCIPITAL_MIDDLE",
    "CTX_LH_G_ORBITAL",
    "CTX_LH_G_PARIET_INF_ANGULAR",
    "CTX_LH_G_PARIET_INF_SUPRAMAR",
    "CTX_LH_G_PARIETAL_SUP",
    "CTX_LH_G_POSTCENTRAL",
    "CTX_LH_G_PRECENTRAL",
    "CTX_LH_G_PRECUNEUS",
    "CTX_LH_G_RECTUS",
    "CTX_LH_G_TEMP_SUP_G_T_TRANSV",
    "CTX_LH_G_TEMP_SUP_LATERAL",
    "CTX_LH_G_TEMP_SUP_PLAN_POLAR",
    "CTX_LH_G_TEMP_SUP_PLAN_TEMPO",
    "CTX_LH_G_TEMPORAL_INF",
    "CTX_LH_G_TEMPORAL_MIDDLE",
    "CTX_LH_LAT_FIS_ANT_HORIZONT",
    "CTX_LH_LAT_FIS_ANT_VERTICAL",
    "CTX_LH_LAT_FIS_POST",
    "CTX_LH_POLE_TEMPORAL",
    "CTX_LH_S_CALCARINE",
    "CTX_LH_S_CENTRAL",
    "CTX_LH_S_CINGUL_MARGINALIS",
    "CTX_LH_S_CIRCULAR_INSULA_ANT",
    "CTX_LH_S_CIRCULAR_INSULA_INF",
    "CTX_LH_S_CIRCULAR_INSULA_SUP",
    "CTX_LH_S_COLLAT_TRANSV_ANT",
    "CTX_LH_S_FRONT_INF",
    "CTX_LH_S_FRONT_MIDDLE",
    "CTX_LH_S_FRONT_SUP",
    "CTX_LH_S_INTRAPARIET_AND_P_TRANS",
    "CTX_LH_S_OC_TEMP_MED_AND_LINGUAL",
    "CTX_LH_S_ORBITAL_H_SHAPED",
    "CTX_LH_S_ORBITAL_LATERAL",
    "CTX_LH_S_ORBITAL_MED_OLFACT",
    "CTX_LH_S_PARIETO_OCCIPITAL",
    "CTX_LH_S_PERICALLOSAL",
    "CTX_LH_S_POSTCENTRAL",
    "CTX_LH_S_PRECENTRAL_INF_PART",
    "CTX_LH_S_PRECENTRAL_SUP_PART",
    "CTX_LH_S_SUBORBITAL",
    "CTX_LH_S_SUBPARIETAL",
    "CTX_LH_S_TEMPORAL_INF",
    "CTX_LH_S_TEMPORAL_SUP",
    "CTX_LH_S_TEMPORAL_TRANSVERSE",
    "CTX_RH_G_INS_LG_AND_S_CENT_INS",
    "CTX_RH_G_AND_S_CINGUL_ANT",
    "CTX_RH_G_AND_S_CINGUL_MID_ANT",
    "CTX_RH_G_AND_S_CINGUL_MID_POST",
    "CTX_RH_G_AND_S_FRONTOMARGIN",
    "CTX_RH_G_AND_S_PARACENTRAL",
    "CTX_RH_G_AND_S_SUBCENTRAL",
    "CTX_RH_G_CINGUL_POST_DORSAL",
    "CTX_RH_G_FRONT_INF_OPERCULAR",
    "CTX_RH_G_FRONT_INF_ORBITAL",
    "CTX_RH_G_FRONT_INF_TRIANGUL",
    "CTX_RH_G_FRONT_MIDDLE",
    "CTX_RH_G_FRONT_SUP",
    "CTX_RH_G_INSULAR_SHORT",
    "CTX_RH_G_OC_TEMP_LAT_FUSIFOR",
    "CTX_RH_G_OC_TEMP_MED_PARAHIP",
    "CTX_RH_G_ORBITAL",
    "CTX_RH_G_PARIET_INF_ANGULAR",
    "CTX_RH_G_PARIET_INF_SUPRAMAR",
    "CTX_RH_G_PRECENTRAL",
    "CTX_RH_G_RECTUS",
    "CTX_RH_G_TEMP_SUP_G_T_TRANSV",
    "CTX_RH_G_TEMP_SUP_LATERAL",
    "CTX_RH_G_TEMP_SUP_PLAN_POLAR",
    "CTX_RH_G_TEMP_SUP_PLAN_TEMPO",
    "CTX_RH_G_TEMPORAL_INF",
    "CTX_RH_G_TEMPORAL_MIDDLE",
    "CTX_RH_LAT_FIS_ANT_HORIZONT",
    "CTX_RH_LAT_FIS_ANT_VERTICAL",
    "CTX_RH_LAT_FIS_POST",
    "CTX_RH_POLE_TEMPORAL",
    "CTX_RH_S_CENTRAL",
    "CTX_RH_S_CINGUL_MARGINALIS",
    "CTX_RH_S_CIRCULAR_INSULA_ANT",
    "CTX_RH_S_CIRCULAR_INSULA_INF",
    "CTX_RH_S_CIRCULAR_INSULA_SUP",
    "CTX_RH_S_COLLAT_TRANSV_ANT",
    "CTX_RH_S_FRONT_INF",
    "CTX_RH_S_FRONT_MIDDLE",
    "CTX_RH_S_FRONT_SUP",
    "CTX_RH_S_INTRAPARIET_AND_P_TRANS",
    "CTX_RH_S_OC_TEMP_LAT",
    "CTX_RH_S_OC_TEMP_MED_AND_LINGUAL",
    "CTX_RH_S_ORBITAL_H_SHAPED",
    "CTX_RH_S_ORBITAL_LATERAL",
    "CTX_RH_S_ORBITAL_MED_OLFACT",
    "CTX_RH_S_PERICALLOSAL",
    "CTX_RH_S_POSTCENTRAL",
    "CTX_RH_S_PRECENTRAL_INF_PART",
    "CTX_RH_S_PRECENTRAL_SUP_PART",
    "CTX_RH_S_SUBORBITAL",
    "CTX_RH_S_SUBPARIETAL",
    "CTX_RH_S_TEMPORAL_INF",
    "CTX_RH_S_TEMPORAL_SUP",
    "CTX_RH_S_TEMPORAL_TRANSVERSE",
)

# Desikan-Killiany cortical regions grouped into lobe slots, as (left slot,
# right slot, region names without the ``CTX_?H_`` prefix). Slots 11 and 12 are
# the left and right occipital lobe: the reference reserves them but no Brain
# Treebank electrode falls there, so the released lobe embedding table has two
# rows that were never trained.
_LOBE_CORTICAL: tuple[tuple[int, int, tuple[str, ...]], ...] = (
    (
        5,
        6,
        (
            "SUPERIORFRONTAL",
            "ROSTRALMIDDLEFRONTAL",
            "CAUDALMIDDLEFRONTAL",
            "PARSOPERCULARIS",
            "PARSORBITALIS",
            "PARSTRIANGULARIS",
            "LATERALORBITOFRONTAL",
            "MEDIALORBITOFRONTAL",
            "PRECENTRAL",
            "PARACENTRAL",
        ),
    ),
    (
        7,
        8,
        (
            "SUPERIORPARIETAL",
            "INFERIORPARIETAL",
            "SUPRAMARGINAL",
            "POSTCENTRAL",
            "PRECUNEUS",
        ),
    ),
    (
        9,
        10,
        (
            "SUPERIORTEMPORAL",
            "MIDDLETEMPORAL",
            "INFERIORTEMPORAL",
            "BANKSSTS",
            "FUSIFORM",
            "TRANSVERSETEMPORAL",
            "ENTORHINAL",
            "TEMPORALPOLE",
            "PARAHIPPOCAMPAL",
        ),
    ),
    (
        13,
        14,
        (
            "ROSTRALANTERIORCINGULATE",
            "CAUDALANTERIORCINGULATE",
            "POSTERIORCINGULATE",
            "ISTHMUSCINGULATE",
        ),
    ),
    (15, 16, ("INSULA",)),
)

# Subcortical structures, which the reference keeps as lobe slots of their own.
_LOBE_SUBCORTICAL: tuple[tuple[int, int, str], ...] = (
    (1, 2, "AMYGDALA"),
    (3, 4, "HIPPOCAMPUS"),
    (17, 18, "PUTAMEN"),
    (19, 20, "INF_LAT_VENT"),
)

_LOBE_REGIONS: dict[str, int] = {"UNKNOWN": 0}
for _left, _right, _names in _LOBE_CORTICAL:
    for _name in _names:
        _LOBE_REGIONS[f"CTX_LH_{_name}"] = _left
        _LOBE_REGIONS[f"CTX_RH_{_name}"] = _right
for _left, _right, _name in _LOBE_SUBCORTICAL:
    _LOBE_REGIONS[f"LEFT_{_name}"] = _left
    _LOBE_REGIONS[f"RIGHT_{_name}"] = _right

_N_LOBES = 21
_DESTRIEUX_LOOKUP: dict[str, int] = {
    name: slot for slot, name in enumerate(_DESTRIEUX_REGIONS)
}
_SPATIAL_VOCABULARIES: dict[str, tuple[dict[str, int], int]] = {
    "parcels": (_DESTRIEUX_LOOKUP, len(_DESTRIEUX_REGIONS)),
    "lobes": (_LOBE_REGIONS, _N_LOBES),
}


def _region_indices(labels: list[str], vocabulary: dict[str, int]) -> torch.Tensor:
    """Index per-channel region ``labels`` into ``vocabulary``.

    Labels are matched the way the reference reads them off its localization
    files: upper-cased with hyphens turned into underscores. Labels outside the
    vocabulary fall back to slot 0, the padding slot, whose embedding stays zero.

    Parameters
    ----------
    labels : list of str
        One region label per channel.
    vocabulary : dict
        Region name to embedding slot.

    Returns
    -------
    torch.Tensor
        ``(n_chans,)`` long tensor of slots.
    """
    slots = []
    unmatched = set()
    for label in labels:
        key = str(label).replace("-", "_").upper()
        if key in vocabulary:
            slots.append(vocabulary[key])
        else:
            slots.append(0)
            unmatched.add(label)
    if unmatched:
        warnings.warn(
            f"BaRISTA did not recognise the region labels {sorted(unmatched)}; "
            f"the corresponding channels get a zero spatial embedding. Region "
            f"names follow the reference atlas tables, e.g. "
            f"'ctx-lh-G_front_middle' for parcels or 'ctx-lh-superiorfrontal' "
            f"for lobes.",
            UserWarning,
            stacklevel=3,
        )
    return torch.tensor(slots, dtype=torch.long)


class BaRISTA(EEGModuleMixin, nn.Module, license="other"):
    r"""BaRISTA from Oganesian, Hashemi and Shanechi (2025) [Oganesian2025]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`
    :bdg-dark-line:`Channel`

    .. versionadded:: 1.8.2

    A self-supervised intracranial EEG (iEEG) foundation model whose defining
    feature is that the *spatial scale* at which electrodes are encoded is a
    free choice. Each channel is cut into temporal patches and tokenized
    independently of space; space then enters as a single learned embedding
    added to the token, selected by the channel's category at the chosen scale
    -- the electrode coordinate, the atlas parcel it sits in, or the lobe. The
    whole ``(patch, channel)`` token sequence is then processed by one
    transformer that attends over space and time jointly, rather than by
    cascaded spatial and temporal transformers [Oganesian2025]_.

    The paper's central finding is that encoding space at a scale *larger* than
    the single channel improves downstream decoding: parcel-level encoding beat
    channel-level encoding and both published iEEG baselines (Brant and
    Population Transformer) on sentence-onset and speech detection, despite
    BaRISTA being 20x smaller than PopT and 500x smaller than Brant.

    .. rubric:: Architecture Overview

    Tokens are assembled additively, for the :math:`i`-th patch of the
    :math:`j`-th channel,

    .. math::
        \mathbf{S}_{ij} = \mathcal{F}(\mathbf{P}_{ij}) + \mathbf{E}_{sp(j)},

    where :math:`\mathbf{P}_{ij}` is the raw patch, :math:`\mathcal{F}` the
    tokenizer and :math:`\mathbf{E}_{sp(j)}` the embedding of the channel's
    spatial category. The tokens are laid out with space and time interleaved,

    .. math::
        \mathbf{S} = [\mathbf{S}_{11}, \ldots, \mathbf{S}_{1C}, \mathbf{S}_{21},
        \ldots, \mathbf{S}_{nC}],

    so a single encoder attends over all :math:`nC` of them at once.

    .. rubric:: Macro Components

    - **Patch tokenizer** (``BaRISTA.patch_tokenizer``,
      ``BaRISTA.temporal_encoder``, ``BaRISTA.temporal_pooler``).
      *Operations:* split each channel into non-overlapping patches of
      ``patch_size`` samples; run every patch through a shared dilated CNN
      (``cnn_depth + 1`` residual blocks of two width-``cnn_kernel_size``
      convolutions with exponentially growing dilation, each followed by a
      parameter-free layer norm over time and a GELU), which maps a patch back
      to a univariate signal of the same length; then apply one bias-free
      linear layer to get a ``d_model`` token. *Role:* turn a
      ``(n_chans, n_times)`` segment into an ``(n_patches, n_chans, d_model)``
      token grid, one token per electrode and patch, with no mixing across
      channels. The dilated CNN is used for its wide receptive field over the
      oscillatory content of the patch.
    - **Spatial embedding** (``BaRISTA.spatial_emb``). *Operations:* look up one
      learned vector per channel, selected by the channel's category at the
      chosen ``spatial_scale``, and add it to every token of that channel.
      Multi-dimensional scales (the three electrode coordinates) keep one
      embedding table per dimension and sum the lookups. *Role:* the only place
      space enters the model, which is what makes the spatial scale a single
      knob.
    - **Encoder** (``BaRISTA.backbone``). *Operations:* ``n_layers`` pre-norm
      blocks with RMSNorm, multi-head self-attention over the full interleaved
      sequence with rotary embeddings on the *patch* index, and a gated
      feed-forward block. *Role:* model cross-channel and cross-time
      interactions concurrently in one attention stack.
    - **Read-out** (``BaRISTA.token_pooling``, ``BaRISTA.final_layer``).
      *Operations:* collapse the token sequence with a learned bias-free linear
      combination (or a mean), then apply a linear classifier. *Role:* produce
      the class logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* non-overlapping patches of ``patch_size`` samples, with the
      dilated CNN encoding within-patch dynamics and rotary embeddings encoding
      the patch index across the sequence. All channels of a patch share one
      rotary position, so attention sees "same time, different electrode" and
      "same electrode, different time" alike.
    - *Spatial:* one additive learned embedding per channel, indexed by its
      category at the chosen scale. At scales coarser than the channel, two
      electrodes in the same parcel or lobe get *identical* spatial encodings.
    - *Spectral:* none explicitly; oscillatory structure is left to the dilated
      CNN.

    .. rubric:: Spatial scales

    ``spatial_scale`` selects which of the paper's three scales is used, and is
    the knob its experiments turn:

    .. list-table::
       :header-rows: 1

       * - ``spatial_scale``
         - Paper name
         - Categories
         - Source
       * - ``"coords"``
         - Channel
         - ``coord_bins`` per axis, summed over 3 axes
         - ``chs_info`` electrode positions
       * - ``"parcels"``
         - Atlas parcels
         - 121 Destrieux parcels and subcortical structures
         - ``spatial_regions``
       * - ``"lobes"``
         - Lobes
         - 21 lobes and subcortical structures
         - ``spatial_regions``
       * - ``"none"``
         - --
         - --
         - no spatial encoding at all

    ``"coords"`` is the default because it is the only scale that braindecode
    can derive on its own, from the electrode positions in ``chs_info``.
    ``"parcels"`` -- the paper's best configuration -- and ``"lobes"`` need an
    electrode-to-region assignment, which is an atlas lookup that braindecode
    does not perform, so the labels must be passed in ``spatial_regions``.
    Labels are matched case-insensitively against the reference atlas tables
    with hyphens read as underscores, so FreeSurfer spellings such as
    ``"ctx-lh-G_front_middle"`` work as they come. Unrecognised labels, like the
    reference's ``UNKNOWN``, get a zeroed embedding, and a warning.

    .. rubric:: Channel metadata

    Only ``spatial_scale="coords"`` reads ``chs_info``, and it needs the
    ``"loc"`` entries. Positions are taken in metres, as MNE stores them,
    converted to millimetres, flipped from MNE's RAS convention to the LPI
    convention the paper uses, and rounded to a 1 mm integer grid centred on the
    head origin, which is the index into the per-axis embedding tables.
    Positions outside the ``coord_bins``-millimetre cube are clipped to its
    faces. A channel whose position is missing or not
    finite, which is how MNE marks one it cannot place, is an error: it would
    otherwise be encoded as sitting at the head origin.

    This binning is braindecode's own. The reference performs no conversion,
    because it indexes the embedding directly with the integer LPI voxel
    coordinates tabulated by Brain Treebank, which braindecode has no
    equivalent of; a millimetre grid about the head origin is the closest
    stand-in that ``chs_info`` supports.

    .. rubric:: Pre-trained weights

    Three checkpoints are published in the reference repository, one per spatial
    encoding scale, all masked-pretrained on Brain Treebank. They are not
    exposed here, because their spatial embedding tables are indexed by Brain
    Treebank's own electrode-localization tables, which braindecode has no
    access to. The parcel and lobe tables do transfer as-is, since this port
    reproduces the reference's category ordering exactly, but the coordinate
    tables are indexed by that dataset's volumetric convention rather than by
    the millimetre grid used here. Loading a checkpoint is otherwise only a key
    rename, as the module names differ
    (``tokenizer.temporal_encoder.feature_extractor.net.*`` to
    ``temporal_encoder.blocks.*``, ``tokenizer.temporal_pooler.final_layer`` to
    ``temporal_pooler``, ``tokenizer.spatial_encoder.subcomponent_embeddings.*``
    to ``spatial_emb.tables.*`` and ``backbone.layers.*.attention`` to
    ``backbone.layers.*.self_attn``).

    .. rubric:: License

    The reference implementation is Copyright (c) 2025 University of Southern
    California and is licensed for educational, research and non-profit use
    only; commercial use requires an agreement with the USC Stevens Center for
    Innovation. This file inherits those terms and is therefore *not* covered by
    braindecode's BSD-3-Clause license.

    .. note::
        Logits were checked against the reference implementation for all three
        spatial scales, by transplanting each released checkpoint into this port
        and into the reference, and agree to floating-point noise. Parameter
        counts match those checkpoints exactly: 869,800 for ``"coords"``,
        839,144 for ``"parcels"`` and 832,744 for ``"lobes"``, excluding the
        classification head.

        The reference runs attention through ``xformers`` with a block-diagonal
        mask, packing the whole minibatch into one sequence; this port uses
        :func:`~torch.nn.functional.scaled_dot_product_attention` over a regular
        batch axis, which is equivalent because the mask only ever blocks
        attention across samples.

        The masked latent reconstruction objective used for pretraining, its
        spatially-guided masking, the EMA target tokenizer and the predictor
        network are out of scope: this port is the encoder and a classification
        head. Note that the paper's downstream protocol also uses the EMA target
        tokenizer rather than the online one, a distinction that only exists
        during pretraining.

    Parameters
    ----------
    spatial_scale : {"coords", "parcels", "lobes", "none"}
        Spatial scale at which electrodes are encoded. ``"coords"`` reads the
        electrode positions from ``chs_info``; ``"parcels"`` and ``"lobes"``
        need ``spatial_regions``; ``"none"`` disables spatial encoding.
    spatial_regions : list of str, optional
        Region label of each channel, required by ``spatial_scale="parcels"``
        and ``"lobes"`` and unused otherwise.
    coord_bins : int
        Number of embedding slots per coordinate axis, i.e. the width in
        millimetres of the cube of head positions that can be encoded. Default
        200, as in the reference.
    patch_size : int
        Number of samples per temporal patch, default 512 (250 ms at the
        paper's 2048 Hz). Windows are tokenized into whole patches, so a window
        that is not a multiple of ``patch_size`` loses its trailing samples, as
        in the reference.
    d_model : int
        Token embedding dimension.
    n_layers : int
        Number of transformer encoder blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden dimension of the feed-forward blocks, as a multiple of
        ``d_model``.
    cnn_depth : int
        Number of hidden blocks of the dilated CNN temporal encoder; the
        encoder has ``cnn_depth + 1`` blocks in total, the last one mapping
        back to a univariate signal.
    cnn_channels : int
        Number of feature maps of the hidden blocks of the dilated CNN.
    cnn_kernel_size : int
        Convolution width of the dilated CNN.
    pooling : {"learned", "mean"}
        Token aggregation before the head. ``"learned"`` reproduces the paper's
        finetuning protocol, a bias-free linear combination of the tokens;
        ``"mean"`` averages them instead, giving a head that is independent of
        ``n_chans`` and ``n_times``.
    drop_prob : float
        Dropout rate used in the encoder.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks, default
        :class:`~torch.nn.GELU`.

    References
    ----------
    .. [Oganesian2025] Oganesian, L. L., Hashemi, S. & Shanechi, M. M. (2025).
       BaRISTA: Brain scale informed spatiotemporal representation of human
       intracranial neural activity. Advances in Neural Information Processing
       Systems 38. https://arxiv.org/abs/2512.12135
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
        # --- model hyperparameters (defaults: the published BaRISTA) ---
        *,
        spatial_scale: str = "coords",
        spatial_regions: list[str] | None = None,
        coord_bins: int = 200,
        patch_size: int = 512,
        d_model: int = 64,
        n_layers: int = 12,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        cnn_depth: int = 4,
        cnn_channels: int = 5,
        cnn_kernel_size: int = 3,
        pooling: str = "learned",
        drop_prob: float = 0.1,
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

        if spatial_scale not in ("coords", "parcels", "lobes", "none"):
            raise ValueError(
                f"spatial_scale must be one of 'coords', 'parcels', 'lobes' or "
                f"'none', got {spatial_scale!r}."
            )
        if pooling not in ("learned", "mean"):
            raise ValueError(f"pooling must be 'learned' or 'mean', got {pooling!r}.")
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if d_model % num_heads:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if (d_model // num_heads) % 2:
            raise ValueError(
                f"The attention head dimension (d_model // num_heads = "
                f"{d_model // num_heads}) must be even for the rotary embedding."
            )

        self.spatial_scale = spatial_scale
        self.coord_bins = coord_bins
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.pooling = pooling
        self.drop_prob = drop_prob
        self.activation = activation

        # Kept as plain ints because EEGModuleMixin hides its signal properties
        # from TorchScript, so a scripted forward cannot read self.n_chans.
        self.n_chans_grid = self.n_chans
        # The trailing samples are dropped when n_times is not a multiple of
        # patch_size, as in the reference.
        self.n_patches = self.n_times // patch_size

        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size,
            n_times=self.n_times,
            on_non_divisible="crop",
            output_order="patch_channel",
        )
        # Every (patch, channel) pair is encoded independently, so the grid is
        # folded into the height axis of a single-channel 2D convolution, which
        # only ever slides along time. Folding patches before channels is what
        # makes the sequence time-space interleaved, as in Eq. 1 of the paper.
        self.fold_grid = Rearrange("batch patch chan time -> batch 1 (patch chan) time")
        self.temporal_encoder = _DilatedConvEncoder(
            patch_size=patch_size,
            hidden_channels=cnn_channels,
            depth=cnn_depth,
            kernel_size=cnn_kernel_size,
        )
        self.unfold_grid = Rearrange("batch 1 seq time -> batch seq time")
        self.temporal_pooler = nn.Linear(patch_size, d_model, bias=False)

        self.spatial_emb = self._build_spatial_embedding(spatial_scale, spatial_regions)

        # All channels of a patch share one rotary position, so the temporal
        # index is constant over each block of n_chans consecutive tokens.
        self.register_buffer(
            "position_ids",
            torch.arange(self.n_patches).repeat_interleave(self.n_chans),
            persistent=False,
        )
        self.backbone = _Transformer(
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
            activation=activation,
            max_position=max(1024, self.n_patches),
        )

        n_tokens = self.n_patches * self.n_chans
        self.token_pooling = (
            nn.Linear(n_tokens, 1, bias=False) if pooling == "learned" else None
        )
        self.final_layer = nn.Linear(d_model, self.n_outputs)

    def _build_spatial_embedding(
        self, spatial_scale: str, spatial_regions: list[str] | None
    ) -> _SpatialEmbedding | None:
        """Resolve the per-channel spatial categories and build their embedding."""
        if spatial_scale == "none":
            return None

        if spatial_scale == "coords":
            if not self._chs_info:
                raise ValueError(
                    "BaRISTA reads the electrode coordinates from chs_info when "
                    "spatial_scale='coords', which must therefore be given and "
                    "non-empty. Pass region labels in spatial_regions and use "
                    "spatial_scale='parcels' or 'lobes' instead, or disable "
                    "spatial encoding with spatial_scale='none'."
                )
            locations = extract_channel_locations_from_chs_info(
                self.chs_info, num_channels=self.n_chans
            )
            # The helper gives up at the first channel without a usable "loc",
            # returning a short array, or None when no position is usable at all.
            # It lets NaN through, though, which is what MNE stores for a channel
            # whose position it does not know, so that is rejected here as well.
            positions = (
                None
                if locations is None
                else torch.as_tensor(locations, dtype=torch.float32)
            )
            if (
                positions is None
                or len(positions) != self.n_chans
                or not positions.isfinite().all()
            ):
                raise ValueError(
                    "BaRISTA needs a position for every channel when "
                    "spatial_scale='coords', but the 'loc' entries of chs_info "
                    "are missing, degenerate or not finite. Set a montage, or "
                    "encode space at a coarser scale with spatial_scale='parcels' "
                    "or 'lobes' and region labels in spatial_regions, or disable "
                    "spatial encoding with spatial_scale='none'."
                )
            # The reference converts nothing: it indexes the embedding with the
            # integer LPI voxel coordinates that Brain Treebank's localization
            # table already provides. chs_info gives metres in MNE's RAS
            # convention instead, so the axes are negated for LPI and binned at
            # 1 mm about the head origin -- a different origin from that
            # localization volume, which is why the released coordinate tables
            # do not transfer. Rounding stands in for the reference's truncation,
            # a no-op on its already-integer input, and the clamp keeps far-field
            # positions in range.
            coords_mm = -1e3 * positions
            indices = coords_mm.round() + self.coord_bins // 2
            indices = indices.clamp(0, self.coord_bins - 1).to(torch.long)
            # One table per axis, summed, as in Appendix D of the paper.
            return _SpatialEmbedding(
                indices.T, self.d_model, self.coord_bins, padding_idx=None
            )

        vocabulary, n_regions = _SPATIAL_VOCABULARIES[spatial_scale]
        if spatial_regions is None:
            raise ValueError(
                f"BaRISTA cannot derive the {spatial_scale} of each electrode on "
                f"its own, so spatial_regions must be given when "
                f"spatial_scale={spatial_scale!r}."
            )
        if len(spatial_regions) != self.n_chans:
            raise ValueError(
                f"spatial_regions has {len(spatial_regions)} labels but the model "
                f"has {self.n_chans} channels."
            )
        indices = _region_indices(list(spatial_regions), vocabulary)
        # Slot 0 is the reference's UNKNOWN, kept as the padding index so that
        # electrodes of unknown region contribute nothing.
        return _SpatialEmbedding(
            indices.unsqueeze(0), self.d_model, n_regions, padding_idx=0
        )

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
        if x.shape[1] != self.n_chans_grid:
            raise ValueError(
                f"BaRISTA was built for {self.n_chans_grid} channels but got input "
                f"with {x.shape[1]}; rebuild the model for this montage."
            )
        # The spatial embedding is tiled for a fixed patch count, and so (for
        # pooling="learned") is the read-out, so reject a different input length
        # outright rather than silently cropping it to a different grid. Windows
        # that differ only in samples the tokenizer drops are accepted.
        if x.shape[-1] // self.patch_size != self.n_patches:
            raise ValueError(
                f"BaRISTA was built for {self.n_patches} temporal patches of "
                f"{self.patch_size} samples but got input with {x.shape[-1]} "
                f"samples; rebuild the model for this window length."
            )

        patches = self.patch_tokenizer(x)
        patches = self.fold_grid(patches)
        patches = self.temporal_encoder(patches)
        patches = self.unfold_grid(patches)
        tokens = self.temporal_pooler(patches)

        if self.spatial_emb is not None:
            # (n_chans, d_model) tiled over patches, which matches the
            # interleaved (patch, channel) token order.
            spatial = self.spatial_emb().repeat(self.n_patches, 1)
            tokens = tokens + spatial[None]

        latents = self.backbone(tokens, self.position_ids)

        if self.token_pooling is not None:
            features = self.token_pooling(latents.transpose(1, 2)).squeeze(dim=-1)
        else:
            features = latents.mean(dim=1)
        logits = self.final_layer(features)

        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {
                "features": features,
                "cls_token": None,  # nosec B105
            }
        return logits


class _SpatialEmbedding(nn.Module):
    """Sum of one learned embedding table per spatial dimension.

    Reproduces :math:`\\mathbf{E}_j = \\sum_{w} e_{sp_w(j)}` of Appendix D: a
    scale made of several dimensions, such as the three electrode coordinates,
    keeps one table per dimension and sums the per-channel lookups. Scales with
    a single category per channel, such as parcels and lobes, are the
    one-dimensional case.

    Parameters
    ----------
    indices : torch.Tensor
        ``(n_dims, n_chans)`` long tensor of embedding slots.
    d_model : int
        Token embedding dimension.
    n_slots : int
        Number of slots of each table.
    padding_idx : int, optional
        Slot whose embedding is pinned to zero, used for unknown categories.
    """

    def __init__(
        self,
        indices: torch.Tensor,
        d_model: int,
        n_slots: int,
        padding_idx: int | None,
    ):
        super().__init__()
        self.register_buffer("indices", indices, persistent=False)
        self.tables = nn.ModuleList(
            [
                nn.Embedding(n_slots, d_model, padding_idx=padding_idx)
                for _ in range(indices.shape[0])
            ]
        )

    def forward(self) -> torch.Tensor:
        """Return the ``(n_chans, d_model)`` spatial encoding of the montage."""
        encoding = torch.zeros(
            self.indices.shape[1],
            self.tables[0].embedding_dim,
            device=self.indices.device,
            dtype=self.tables[0].weight.dtype,
        )
        for dim, table in enumerate(self.tables):
            encoding = encoding + table(self.indices[dim])
        return encoding


class _DilatedConvEncoder(nn.Module):
    """Dilated CNN that encodes each temporal patch, from TS2Vec and SimTS.

    A stack of ``depth + 1`` residual blocks whose dilation doubles with depth,
    so the receptive field spans the patch while every block keeps the patch
    length. The last block maps back to a single feature map, so the encoder
    takes a univariate patch to a univariate patch of the same length, and it is
    the following linear layer that forms the token.

    Parameters
    ----------
    patch_size : int
        Number of samples per patch, which is also the width the layer norms
        normalise over.
    hidden_channels : int
        Number of feature maps of the hidden blocks.
    depth : int
        Number of hidden blocks.
    kernel_size : int
        Convolution width.
    """

    def __init__(
        self, patch_size: int, hidden_channels: int, depth: int, kernel_size: int
    ):
        super().__init__()
        if depth < 0:
            raise ValueError(f"cnn_depth must be non-negative, got {depth}.")
        channels = [hidden_channels] * depth + [1]
        in_channels = 1
        blocks = []
        for i, out_channels in enumerate(channels):
            blocks.append(
                _DilatedConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    norm_size=patch_size,
                    final=i == len(channels) - 1,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(batch, 1, n_tokens, patch_size)`` patches in place."""
        return self.blocks(x)


class _DilatedConvBlock(nn.Module):
    """Two dilated convolutions over time with a residual stream.

    Each convolution is followed by a layer norm over the time axis, without
    learnable affine parameters as in the reference, and a GELU. The residual is
    projected when the block changes the number of feature maps, and always on
    the final block.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    kernel_size : int
        Convolution width.
    dilation : int
        Convolution dilation.
    norm_size : int
        Length of the time axis, which the layer norm normalises over.
    final : bool
        Whether this is the last block of the stack, which always projects its
        residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        norm_size: int,
        final: bool,
    ):
        super().__init__()
        self.conv1 = _SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _SamePadConv(out_channels, out_channels, kernel_size, dilation)
        # Parameter-free, so one instance is shared by both halves of the block.
        self.norm = nn.LayerNorm(norm_size, elementwise_affine=False)
        self.activation = nn.GELU()
        if in_channels != out_channels or final:
            self.projector = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            nn.init.kaiming_normal_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projector(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class _SamePadConv(nn.Module):
    """Dilated convolution along time that preserves the input length.

    The grid of patches is carried in the height axis of a 2D convolution of
    width ``kernel_size`` and height 1, so the convolution only ever slides
    along time and every patch is encoded independently. An even receptive field
    over-pads by one sample, which is then trimmed from the right.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    kernel_size : int
        Convolution width.
    dilation : int
        Convolution dilation.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, receptive_field // 2),
            dilation=(1, dilation),
        )
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.remove = 1 - receptive_field % 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.remove > 0:
            x = x[..., : -self.remove]
        return x


class _Transformer(nn.Module):
    """Stack of pre-norm transformer blocks over the interleaved token sequence.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_layers : int
        Number of blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden dimension of the feed-forward blocks, as a multiple of
        ``d_model``.
    drop_prob : float
        Dropout rate.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks.
    max_position : int
        Largest patch index the rotary tables must cover.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_heads: int,
        mlp_ratio: int,
        drop_prob: float,
        activation: type[nn.Module],
        max_position: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_prob=drop_prob,
                    activation=activation,
                    max_position=max_position,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=_NORM_EPS)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Encode ``(batch, seq, d_model)`` tokens tagged by their patch index."""
        for layer in self.layers:
            x = layer(x, position_ids)
        return self.norm(x)


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm block: rotary self-attention then a gated feed-forward.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden dimension of the feed-forward block, as a multiple of
        ``d_model``.
    drop_prob : float
        Dropout rate.
    activation : type[nn.Module]
        Activation layer class of the feed-forward block.
    max_position : int
        Largest patch index the rotary tables must cover.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        drop_prob: float,
        activation: type[nn.Module],
        max_position: int,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=_NORM_EPS)
        self.self_attn = _RotarySelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            drop_prob=drop_prob,
            max_position=max_position,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm2 = nn.RMSNorm(d_model, eps=_NORM_EPS)
        self.mlp = GLUFeedForward(
            d_model=d_model,
            d_ff=mlp_ratio * d_model,
            drop_prob=drop_prob,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, position_ids)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x + residual


class _RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings on the patch index.

    Attention is unmasked over the whole interleaved sequence, so a token
    attends to every other electrode and every other patch at once. Rotary
    embeddings carry the temporal order only; space is already in the token
    through the additive spatial embedding.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    drop_prob : float
        Dropout applied to the attention output.
    max_position : int
        Largest patch index the rotary tables must cover.
    """

    def __init__(
        self, d_model: int, num_heads: int, drop_prob: float, max_position: int
    ):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        cos, sin = rotary_positional_encoding(max_position, d_model // num_heads)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.dropout = nn.Dropout(drop_prob)
        self.split_heads = Rearrange(
            "batch seq (heads dim) -> batch heads seq dim", heads=num_heads
        )
        self.merge_heads = Rearrange("batch heads seq dim -> batch seq (heads dim)")

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # The positions are shared by the whole batch and by every head, so both
        # axes are inserted as size-1 broadcast axes against (batch, heads, seq,
        # head_dim).
        cos = self.cos_cached[position_ids].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[position_ids].unsqueeze(0).unsqueeze(0)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        # Dropout falls on the attention output rather than on the weights,
        # through dropout_p, because the reference disables the latter outright.
        attention = F.scaled_dot_product_attention(query, key, value)
        attention = self.dropout(attention)
        attention = self.merge_heads(attention)
        return self.o_proj(attention)

# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: Apache-2.0
"""DIVER-1: an any-variate iEEG foundation model.

Reimplementation of DIVER-1 (Han et al., 2025), "DIVER-1: Scaling Intracranial
EEG Foundation Models for Transferable Representations". The architecture is
transcribed from the authors' reference implementation, whose Transformer
encoder is adapted from Salesforce's MOIRAI / ``uni2ts`` (Copyright Salesforce,
Inc.), released under the Apache License, Version 2.0; this file is therefore
distributed under Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0).
The reference repository states no license of its own, so check the terms of the
original implementation before redistributing.

Original Authors: Han et al., Seoul National University
Braindecode Adaptation: Julien Gadonneix
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    INTRACRANIAL_CH_TYPES,
    channel_types_from_chs_info,
    extract_channel_locations_from_chs_info,
)
from braindecode.modules import PatchTokenizer

# Channel-modality vocabulary of the reference ``ChannelTypeEmbedding``: slot 0
# is scalp EEG, slot 1 intracranial EEG.
_MODALITIES = ("EEG", "iEEG")
# Electrode sub-modality vocabulary of the reference ``ChannelSubTypeEmbedding``:
# ECoG grids and strips, and SEEG depth electrodes.
_SUBTYPES = ("grid", "strip", "depth")
# Electrode sub-modality implied by each intracranial channel type. Unlike the
# reference implementation, which reads it off a site-specific table of
# electrode-group name prefixes, we key off the channel kind, so strips are
# never inferred: no MNE kind tells them apart from grids.
_TYPE_TO_SUBTYPE = {"seeg": "depth", "dbs": "depth", "ecog": "grid"}
# Recording modality implied by each channel type.
_TYPE_TO_MODALITY = {t: "iEEG" for t in INTRACRANIAL_CH_TYPES} | {"eeg": "EEG"}


def _label_indices(
    labels: Sequence[str], vocabulary: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index per-channel ``labels`` into ``vocabulary``, flagging the known ones.

    Labels outside the vocabulary are given index 0 and a zero flag, which
    :class:`_ChannelMetaEmbedding` uses to zero their embedding out.

    Parameters
    ----------
    labels : sequence of str
        One label per channel.
    vocabulary : sequence of str
        Embedding slots, in order.

    Returns
    -------
    indices : torch.Tensor
        ``(n_chans,)`` long tensor of slots.
    known : torch.Tensor
        ``(n_chans, 1)`` float mask of the labels found in ``vocabulary``.
    """
    known = [label in vocabulary for label in labels]
    indices = torch.tensor(
        [vocabulary.index(lab) if k else 0 for lab, k in zip(labels, known)],
        dtype=torch.long,
    )
    return indices, torch.tensor(known, dtype=torch.float32).unsqueeze(-1)


class DIVER1(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""DIVER-1 from Han et al. (2025) [Han2025]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`
    :bdg-dark-line:`Channel`

    .. figure:: ../_static/model/diver1_arch.png
        :align: center
        :alt: DIVER-1 architecture overview
        :width: 1000px

        DIVER-1 architecture, pretraining and downstream evaluation pipeline,
        reproduced from [Han2025]_.

    .. versionadded:: 1.8.2

    A self-supervised intracranial EEG (iEEG) foundation model for
    variable-input recordings. Every electrode is cut into temporal patches and
    the resulting ``(channel, time-patch)`` token grid is processed by *any-variate*
    self-attention: all ``n_chans * n_patches`` tokens attend to each other, with
    temporal order carried by RoPE and same- versus cross-channel structure
    carried by a learned binary attention bias. Because the channel term depends
    only on *whether* two tokens share an electrode -- never on the electrode
    index -- the encoder is channel-permutation equivariant and works without
    electrode metadata [Han2025]_.

    .. rubric:: Architecture Overview

    The token grid is assembled additively,

    .. math::
        \mathbf{X} = \mathbf{Y}_{\mathrm{CNN}} + \mathbf{E}_{\mathrm{spectral}}
        + [\mathbf{E}_{\mathrm{position}}, \mathbf{E}_{\mathrm{modality}}]
        + \mathbf{E}_{\mathrm{STCPE}},

    where :math:`[\cdot,\cdot]` is a concatenation along the feature axis, and is
    then encoded by ``n_layers`` any-variate Transformer blocks. Three learned
    register tokens (per-channel, per-patch, and a global one) are prepended to
    the grid before the encoder. As in the reference implementation, their
    encoder outputs are discarded afterwards: the registers only ever act
    through attention, and the read-out uses the token grid itself.

    .. rubric:: Macro Components

    - **Patch encoding** (``DIVER1.patch_tokenizer``, ``DIVER1.patch_cnn``).
      *Operations:* split each channel into non-overlapping patches of
      ``patch_size`` samples, then apply a ``cnn_depth``-layer strided CNN
      (Conv2d + GroupNorm + GELU) that maps every patch to a ``d_model`` token.
      *Role:* turn a ``(n_chans, n_times)`` segment into an
      ``(n_chans, n_patches, d_model)`` grid of local waveform features.
    - **Spectral embedding** (``DIVER1.spectral_emb``). *Operations:* take the
      magnitude of the real FFT of each token and project it back to
      ``d_model``. *Role:* expose frequency content explicitly rather than
      leaving it to be rediscovered by attention.
    - **Position and modality embedding** (``DIVER1.chan_emb``).
      *Operations:*
      encode the MNI :math:`(x, y, z)` coordinate of each electrode with the
      sinusoidal coordinate encoding of PopT, and concatenate it with a learned
      electrode-type embedding (EEG/iEEG plus grid/strip/depth). *Role:* tell
      the encoder where each electrode sits and what kind of contact it is,
      when that metadata is available.
    - **STCPE** (``DIVER1.stcpe``). *Operations:* project the grid down to
      ``d_model // stcpe_ratio``, slide a ``stcpe_window``-wide temporal window
      over it, run a one-layer any-variate Transformer inside each window,
      average the overlapping window outputs and project back up. *Role:*
      an input-conditioned local positional bias that is translation
      equivariant in time and permutation equivariant in channels, replacing
      the channel-axis convolutions of ACPE.
    - **Any-variate encoder** (``DIVER1.encoder``). *Operations:* ``n_layers``
      pre-norm blocks with RMSNorm, grouped-query-shaped attention with QK-norm,
      rotary embeddings on the patch index, a learned binary same/cross-channel
      attention bias per layer and head, and a SwiGLU feed-forward block.
      *Role:* model direct cross-channel, cross-time interactions.
    - **Read-out** (``DIVER1.final_layer``). *Operations:* flatten the
      ``(n_chans, n_patches, d_model)`` token grid (or mean-pool it) and apply a
      linear layer. *Role:* produce the class logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* non-overlapping patches of ``patch_size`` samples; patch order
      enters the attention energy through rotary embeddings on the temporal
      offset, plus the sliding-window STCPE bias.
    - *Spatial (channels):* a sinusoidal encoding of the MNI electrode
      coordinates and a learned electrode-type embedding at the input, and a
      learned binary same/cross-channel bias inside every attention head.
    - *Spectral:* the magnitude of the real FFT of each token, linearly
      projected and added to the grid.

    .. rubric:: Additional Mechanisms

    Electrodes whose coordinates are unknown receive
    :math:`\mathbf{E}_{\mathrm{position}} = \mathbf{0}`, and electrodes of
    unknown sub-modality receive a zero sub-type embedding, so the model runs on
    recordings with incomplete metadata (the paper's ablations show the
    coordinate term contributes little). Set ``use_position_emb=False`` to drop
    the coordinate and type terms entirely, which makes the whole model exactly
    channel-permutation equivariant.

    .. rubric:: Variants

    The published variants differ only in width (``n_layers=12`` throughout),
    and come at two temporal granularities, ``patch_size=500`` (DIVER-1-1s) and
    ``patch_size=50`` (DIVER-1-0.1s), both at 500 Hz:

    .. list-table::
       :header-rows: 1

       * - Variant
         - ``d_model``
         - ``num_heads``
         - Table 4 parameters (1 s / 0.1 s)
       * - Tiny
         - 256
         - 8
         - 13.03M / 12.72M
       * - Small
         - 512
         - 16
         - 51.36M / 50.75M
       * - Base
         - 768
         - 24
         - 115.00M / 114.07M
       * - Large
         - 1024
         - 32
         - 203.95M / 202.70M
       * - XL
         - 2048
         - 64
         - 812.85M / 810.19M
       * - XXL
         - 3072
         - 96
         - 1.83B / 1.82B

    Those are the paper's totals, which include the pretraining reconstruction
    heads and mask token; this encoder-only port is correspondingly smaller
    (12.67M rather than 13.03M for Tiny-1s, plus the classification head).

    .. rubric:: Channel metadata

    All per-electrode metadata is read from ``chs_info``, so it must be passed
    and its entries filled in:

    - ``"kind"`` gives the recording modality. Intracranial kinds (SEEG, ECoG,
      DBS) become ``"iEEG"`` and scalp EEG becomes ``"EEG"``. The modality is
      never guessed, so a kind that is missing or identifies neither raises a
      :class:`ValueError`.
    - ``"kind"`` also gives the electrode sub-modality: SEEG and DBS map to
      ``"depth"`` and ECoG to ``"grid"``. Strips cannot be told apart from
      grids by kind alone, so they are never inferred, and an unresolved
      sub-modality simply gets a zeroed embedding.
    - ``"loc"`` gives the electrode coordinates, in metres as MNE stores them,
      converted internally to the millimetres the sinusoidal encoding expects.
      Coordinates that are missing, non-finite or exactly zero get a zeroed
      coordinate embedding, which is how the paper handles unknown positions.

    .. rubric:: Pre-trained weights

    Both released encoders are on the Hugging Face Hub, all at 500 Hz:

    - ``braindecode/DIVER-1-0.1s-tiny``: the iEEG encoder (``patch_size=50``,
      ``d_model=256``, ``n_layers=12``), ``weights/ieeg_pretrained_weights.pt``;
    - ``braindecode/DIVER-1-1s-small``: the joint iEEG and EEG encoder of paper
      versions 1 and 2 (``patch_size=500``, ``d_model=512``, ``n_layers=12``),
      ``weights/i_eeg_pretrained_weights.pt``.

    ::

        model = DIVER1.from_pretrained(
            "braindecode/DIVER-1-0.1s-tiny", chs_info=raw.info["chs"],
            n_times=500, n_outputs=2,
        )

    ``scripts/convert_diver1_weights.py`` converts the official files; the
    encoder features match the reference model exactly on CPU. The releases
    have no classification head, so the head is initialized on load and needs
    fine-tuning.

    .. rubric:: License

    The code is Apache-2.0, inherited from the MOIRAI / ``uni2ts`` code that the
    reference encoder is adapted from. The released weights are MIT-licensed by
    the DIVER Project.

    .. note::
        Numerical equivalence of the encoder features with the reference
        implementation has been verified layer by layer, for both patch-size
        variants, by transplanting a randomly initialised reference state dict,
        and with both released checkpoints, whose encoder features match
        exactly on CPU (``scripts/convert_diver1_weights.py``). The comparison requires
        disabling the reference's attention dropout, which stays active in eval
        mode there because ``dropout_p`` is passed straight to
        :func:`~torch.nn.functional.scaled_dot_product_attention`; this port
        gates it on ``self.training`` instead. Parameter counts match the
        reference exactly, and Table 4 of [Han2025]_ once the pretraining-only
        reconstruction heads and mask token are excluded (12.67M for the 1 s
        Tiny encoder, 12.66M for 0.1 s).

        Two points where the paper and the reference implementation disagree
        were resolved in favour of the code and of Table 7:
        :math:`\mathbf{E}_{\mathrm{spectral}}` is the FFT of the ``d_model``-dimensional
        CNN token (FFT size ``d_model // 2 + 1``), not of the raw patch as the
        prose suggests; and STCPE averages the overlapping windows rather than
        summing them. The masked multi-domain reconstruction objective (MDRO),
        the patch masking, the spatio-temporal resampling (STR) and the
        :math:`\mu`-parameterization used for pretraining are out of scope.

    Parameters
    ----------
    patch_size : int
        Number of samples per temporal patch, default 500 (1 s at the paper's
        500 Hz; the other published variant is 50).
    d_model : int
        Token embedding dimension, default 256 (the Tiny variant).
    n_layers : int
        Number of any-variate Transformer blocks.
    num_heads : int, optional
        Number of attention heads. Defaults to ``d_model // 32``, as in the
        reference implementation.
    d_ff : int, optional
        Hidden dimension of the feed-forward blocks. Defaults to
        ``4 * d_model``.
    cnn_depth : int
        Number of convolution layers in the patch encoder.
    cnn_stride : int, optional
        Stride of the first (strided) convolution of the patch encoder.
        Defaults to the reference setting: the padded patch length divided by 8
        for ``patch_size >= 100`` and by 16 below, i.e. 64 for the 1 s variant
        and 4 for the 0.1 s variant.
    cnn_kernel_size : int
        Kernel width of the first convolution of the patch encoder; must be odd.
        The remaining ``cnn_depth - 1`` convolutions use width 3 and stride 1.
    use_stcpe : bool
        Whether to add the spatio-temporal conditional positional embedding.
    stcpe_window : int
        Width (in patches) of the STCPE sliding window; must be odd.
    stcpe_ratio : int
        Bottleneck ratio of STCPE: it operates at ``d_model // stcpe_ratio``.
    use_spectral_emb : bool
        Whether to add the spectral embedding.
    use_position_emb : bool
        Whether to add the electrode coordinate and type embeddings.
    pooling : {"flatten", "mean"}
        Token aggregation before the head. ``"flatten"`` reproduces the paper's
        finetuning protocol (a linear classifier on the flattened token grid);
        ``"mean"`` averages over channels and patches first, giving a head that
        is independent of ``n_chans`` and ``n_times``.
    mup_attention : bool
        Scale attention scores by ``1 / head_dim`` (the muP scaling the released
        checkpoints were trained with, ``original_moirai_encoder.py:709`` in the
        official code) instead of the standard ``1 / sqrt(head_dim)``. Keep it
        ``True`` to load the pretrained weights.
    drop_prob : float
        Dropout rate used in the encoder and the spectral embedding.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks, default
        :class:`~torch.nn.SiLU`.

    References
    ----------
    .. [Han2025] Han, D. D., Gwon, Y., Lee, A. L., Lee, T., Lee, S. J., Choi,
       J., Lee, S., Bang, J., Lee, S., Park, D. K., Yoo, S., Chung, C. K. &
       Cha, J. (2025). DIVER-1: Scaling intracranial EEG foundation models for
       transferable representations. arXiv preprint arXiv:2512.19097.
       https://arxiv.org/abs/2512.19097
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
        # --- model hyperparameters (defaults: DIVER-1-1s Tiny) ---
        patch_size: int = 500,
        d_model: int = 256,
        n_layers: int = 12,
        num_heads: int | None = None,
        d_ff: int | None = None,
        cnn_depth: int = 3,
        cnn_stride: int | None = None,
        cnn_kernel_size: int = 63,
        use_stcpe: bool = True,
        stcpe_window: int = 7,
        stcpe_ratio: int = 8,
        use_spectral_emb: bool = True,
        use_position_emb: bool = True,
        pooling: str = "flatten",
        mup_attention: bool = True,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.SiLU,
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

        # Every per-electrode embedding is read from chs_info, so n_chans alone
        # is not enough to build the model.
        if not self._chs_info:
            raise ValueError(
                "DIVER1 reads the electrode modality, sub-modality and "
                "coordinates from chs_info, which must therefore be given and "
                "non-empty."
            )
        if pooling not in ("flatten", "mean"):
            raise ValueError(f"pooling must be 'flatten' or 'mean', got {pooling!r}.")
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        num_heads = num_heads if num_heads is not None else max(1, d_model // 32)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        if d_model % num_heads:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if (d_model // num_heads) % 2:
            raise ValueError(
                f"The attention head dimension (d_model // num_heads = "
                f"{d_model // num_heads}) must be even for the rotary embedding."
            )
        if stcpe_window % 2 == 0:
            raise ValueError(
                f"stcpe_window must be odd so the window is centred, got "
                f"{stcpe_window}."
            )

        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_stcpe = use_stcpe
        self.stcpe_window = stcpe_window
        self.stcpe_ratio = stcpe_ratio
        self.use_spectral_emb = use_spectral_emb
        self.use_position_emb = use_position_emb
        self.pooling = pooling
        self.drop_prob = drop_prob
        self.activation = activation

        # Height of the token grid, kept as a plain int because
        # EEGModuleMixin hides its signal properties from TorchScript, so a
        # scripted forward cannot read self.n_chans.
        self.n_chans_grid = self.n_chans
        # Patches are right-zero-padded when n_times is not a multiple of
        # patch_size (braindecode's default), so round up.
        self.n_patches = -(-self.n_times // patch_size)
        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size, n_times=self.n_times
        )
        self.patch_cnn = _PatchCNN(
            n_chans=self.n_chans,
            patch_size=patch_size,
            d_model=d_model,
            stride=cnn_stride,
            kernel_size=cnn_kernel_size,
            depth=cnn_depth,
        )

        self.spectral_emb = (
            _SpectralEmbedding(d_model, drop_prob) if use_spectral_emb else None
        )

        # Channel metadata, all read from chs_info. Registered non-persistently
        # because it describes the montage the model was built for, not learned
        # state.
        # MNE stores coordinates in metres; DIVER-1 encodes MNI coordinates in
        # millimetres. Channels whose coordinates are non-finite or exactly zero
        # (MNE's two ways of spelling "no montage", as in ``has_valid_locations``)
        # are flagged so their embedding is zeroed, which is how the paper
        # handles unknown positions.
        coords = 1e3 * torch.as_tensor(
            extract_channel_locations_from_chs_info(
                self.chs_info, num_channels=self.n_chans, fill_missing=True
            ),
            dtype=torch.float32,
        )
        coords_known = (
            torch.isfinite(coords).all(dim=-1, keepdim=True)
            & (coords != 0).any(dim=-1, keepdim=True)
        ).to(coords.dtype)
        self.register_buffer("chan_coords", torch.nan_to_num(coords), persistent=False)
        self.register_buffer("chan_coords_known", coords_known, persistent=False)

        # Modality and sub-modality both come from the chs_info channel kinds.
        # The modality is never guessed: the reference takes it as mandatory
        # metadata, and mislabelling scalp EEG as intracranial (or the reverse)
        # picks the wrong learned embedding slot. An undeterminable sub-modality
        # is tolerated with a zeroed embedding, as in the reference.
        types = channel_types_from_chs_info(self.chs_info, num_channels=self.n_chans)
        undetermined = sorted({t for t in types if t not in _TYPE_TO_MODALITY})
        if undetermined:
            raise ValueError(
                f"DIVER1 cannot determine the recording modality of every "
                f"channel: the chs_info 'kind' of some resolves to "
                f"{undetermined}, which is neither scalp EEG nor an "
                f"intracranial type ({sorted(INTRACRANIAL_CH_TYPES)}). Set the "
                f"channel kinds in chs_info accordingly."
            )
        type_idx, _ = _label_indices([_TYPE_TO_MODALITY[t] for t in types], _MODALITIES)
        subtype_idx, subtype_known = _label_indices(
            [_TYPE_TO_SUBTYPE.get(t, "unknown") for t in types], _SUBTYPES
        )
        self.register_buffer("chan_type_idx", type_idx, persistent=False)
        self.register_buffer("chan_subtype_idx", subtype_idx, persistent=False)
        self.register_buffer("chan_subtype_known", subtype_known, persistent=False)

        self.chan_emb = _ChannelMetaEmbedding(d_model) if use_position_emb else None

        self.stcpe = (
            _STCPE(
                d_model=d_model,
                ratio=stcpe_ratio,
                window=stcpe_window,
                activation=activation,
                n_chans=self.n_chans,
                n_patches=self.n_patches,
                mup_attention=mup_attention,
            )
            if use_stcpe
            else None
        )

        # Learned register tokens, prepended as an extra patch column
        # (per-channel), an extra channel row (per-patch) and their corner (one
        # global token). The reference initialises all three as 0.02 * N(0, 1)
        # and reads none of them back, so they serve as attention sinks.
        self.patch_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.chan_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Any-variate attention runs over the flattened (channel, patch) grid.
        # Unflattening only needs the channel count, registers included.
        self.flatten_grid = Rearrange("batch chan patch dim -> batch (chan patch) dim")
        self.unflatten_grid = Rearrange(
            "batch (chan patch) dim -> batch chan patch dim", chan=self.n_chans + 1
        )

        self.encoder = _AnyVariateEncoder(
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            activation=activation,
            # One extra patch position for the register column.
            max_len=max(512, self.n_patches + 1),
            mup_attention=mup_attention,
        )

        head_in_features = (
            self.n_chans * self.n_patches * d_model if pooling == "flatten" else d_model
        )
        self.final_layer = nn.Linear(head_in_features, self.n_outputs)

    def reset_head(self, n_outputs):
        """Replace the linear classification head for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an iEEG batch into class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_outputs)``.
        """
        if x.shape[1] != self.n_chans_grid:
            raise ValueError(
                f"DIVER1 was built for {self.n_chans_grid} channels but got input "
                f"with {x.shape[1]}; rebuild the model for this montage."
            )
        # The rotary tables and (for ``pooling="flatten"``) the head are sized
        # for a fixed patch count, so reject a different input length outright
        # rather than silently zero-padding it to a different grid.
        if -(-x.shape[-1] // self.patch_size) != self.n_patches:
            raise ValueError(
                f"DIVER1 was built for {self.n_patches} temporal patches of "
                f"{self.patch_size} samples but got input with {x.shape[-1]} "
                f"samples; rebuild the model for this window length."
            )

        # Build the token grid. Every term is added to the running grid, in the
        # order of the reference ``Embedder``: the spectral and positional terms
        # see the CNN output, and STCPE sees all of them (as in Eq. 1, which
        # feeds it X).
        tokens = self.patch_tokenizer(x)
        tokens = self.patch_cnn(tokens)
        if self.spectral_emb is not None:
            tokens = tokens + self.spectral_emb(tokens)
        if self.chan_emb is not None:
            # (n_chans, d_model), broadcast over batch and patches.
            chan_emb = self.chan_emb(
                self.chan_coords,
                self.chan_coords_known,
                self.chan_type_idx,
                self.chan_subtype_idx,
                self.chan_subtype_known,
            )
            tokens = tokens + chan_emb[None, :, None, :]
        if self.stcpe is not None:
            tokens = tokens + self.stcpe(tokens)

        # Prepend the register column (per-channel), row (per-patch) and their
        # corner (global).
        batch = tokens.shape[0]
        patch_reg = self.patch_register[None].expand(batch, self.n_chans_grid, -1, -1)
        tokens = torch.cat([patch_reg, tokens], dim=2)
        chan_reg = self.chan_register[None].expand(batch, -1, self.n_patches, -1)
        global_reg = self.global_register[None].expand(batch, -1, -1, -1)
        row = torch.cat([global_reg, chan_reg], dim=2)
        tokens = torch.cat([row, tokens], dim=1)

        _, n_chans, n_patches, _ = tokens.shape
        # Any-variate attention is over the flattened (channel, patch) grid; the
        # two id vectors are what tells the attention which token is which. The
        # reference carries them per sample, to support the variable-length
        # subviews of pretraining; a braindecode batch always shares one montage
        # and length, so a single shared pair of ids is equivalent and keeps the
        # attention bias at (1, num_heads, seq, seq) instead of a per-sample copy.
        var_id = torch.arange(n_chans, device=x.device).repeat_interleave(n_patches)
        time_id = torch.arange(n_patches, device=x.device).repeat(n_chans)
        tokens = self.encoder(
            self.flatten_grid(tokens),
            var_id=var_id,
            time_id=time_id,
        )
        tokens = self.unflatten_grid(tokens)
        # Drop the register row and column, as the reference does: nothing reads
        # their encoder outputs, not even its finetuning protocol.
        tokens = tokens[:, 1:, 1:]

        if self.pooling == "flatten":
            pooled = tokens.flatten(start_dim=1)
        else:
            pooled = tokens.mean(dim=(1, 2))
        return self.final_layer(pooled)


class _PatchCNN(nn.Module):
    """Strided CNN patch encoder mapping each patch to a ``d_model`` token.

    Each patch is symmetrically zero-padded to the next power of two (500 to
    512, 50 to 64), then a strided convolution reduces it to ``out_size``
    positions of ``d_model // out_size`` channels, which are flattened into the
    token. ``depth - 1`` width-3 convolutions refine the result at constant
    length. With the defaults this reproduces Table 7 of [Han2025]_ exactly:
    ``d_model / 8`` intermediate channels and stride 64 for ``patch_size=500``,
    ``d_model / 16`` and stride 4 for ``patch_size=50``, kernels ``{63, 3, 3}``
    and padding ``{31, 1, 1}``.

    Parameters
    ----------
    n_chans : int
        Number of channels of the grid, needed to split the ``(chan patch)``
        axis back apart after the convolutions.
    patch_size : int
        Number of samples per patch.
    d_model : int
        Token embedding dimension.
    stride : int, optional
        Stride of the first convolution. Defaults to the padded patch length
        divided by 8 (``patch_size >= 100``) or 16 (below).
    kernel_size : int
        Width of the first convolution; must be odd.
    depth : int
        Total number of convolution layers.
    """

    def __init__(
        self,
        n_chans: int,
        patch_size: int,
        d_model: int,
        stride: int | None = None,
        kernel_size: int = 63,
        depth: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"cnn_kernel_size must be odd, got {kernel_size}.")
        if depth < 1:
            raise ValueError(f"cnn_depth must be at least 1, got {depth}.")

        padded = 1 << (patch_size - 1).bit_length()
        if stride is None:
            # The divisor is the output length the reference keeps: 8 positions
            # for the 1 s patch, 16 for the 0.1 s one, which 100 separates.
            stride = padded // (8 if patch_size >= 100 else 16)
        if stride < 1 or padded % stride:
            raise ValueError(
                f"cnn_stride ({stride}) must be a positive divisor of the padded "
                f"patch length ({padded})."
            )
        out_size = padded // stride
        if d_model % out_size:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by the number of patch-CNN "
                f"output positions ({out_size} = {padded} // {stride})."
            )
        hidden = d_model // out_size

        pad_total = padded - patch_size
        self.pad = (pad_total // 2, pad_total - pad_total // 2)
        # num_groups is out_size in the reference; fall back to the gcd so
        # narrow configurations (hidden < out_size) stay constructible.
        n_groups = math.gcd(out_size, hidden)

        layers: list[nn.Module] = [
            nn.Conv2d(
                1,
                hidden,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, (kernel_size - 1) // 2),
            ),
            nn.GroupNorm(n_groups, hidden),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv2d(
                    hidden, hidden, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
                ),
                nn.GroupNorm(n_groups, hidden),
                nn.GELU(),
            ]
        # Every (channel, patch) pair is encoded independently, so the grid is
        # folded into the height axis of a single-channel 2D convolution, which
        # only ever slides along ``time``.
        self.fold_grid = Rearrange("batch chan patch time -> batch 1 (chan patch) time")
        self.proj_in = nn.Sequential(*layers)
        # The ``hidden`` filters at each of the ``out`` surviving time positions
        # are what makes up a token: ``d_model = hidden * out``.
        self.unfold_grid = Rearrange(
            "batch hidden (chan patch) out -> batch chan patch (hidden out)",
            chan=n_chans,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, n_chans, n_patches, patch_size)`` to ``d_model`` tokens."""
        x = F.pad(x, self.pad)
        x = self.fold_grid(x)
        x = self.proj_in(x)
        x = self.unfold_grid(x)
        return x


class _SpectralEmbedding(nn.Module):
    """Linear projection of the token magnitude spectrum (CBraMod-style).

    Note that, as in the reference implementation and in Table 7 of [Han2025]_
    (FFT size ``d_model // 2 + 1``), the FFT is taken over the ``d_model``
    features of the CNN token, not over the raw patch samples.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    drop_prob : float
        Dropout applied after the projection.
    """

    def __init__(self, d_model: int, drop_prob: float):
        super().__init__()
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model), nn.Dropout(drop_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rfft is not implemented for half precision: compute it in float32 and
        # cast the amplitudes back.
        spectrum = torch.fft.rfft(x.float(), dim=-1, norm="forward")
        amplitude = spectrum.abs()
        amplitude = amplitude.to(x.dtype)
        return self.spectral_proj(amplitude)


class _ChannelMetaEmbedding(nn.Module):
    r"""Per-electrode coordinate and type embedding, concatenated.

    Reproduces :math:`[\mathbf{E}_{\mathrm{position}},
    \mathbf{E}_{\mathrm{modality}}]`: a sinusoidal encoding of the MNI
    coordinates over the leading features, and the sum of a recording-modality
    and an electrode-sub-modality embedding over the trailing quarter
    (Table 7 of [Han2025]_). Electrodes with unknown coordinates or unknown
    sub-modality contribute zero to the corresponding term.

    Parameters
    ----------
    d_model : int
        Total width of the concatenated embedding.
    """

    def __init__(self, d_model: int):
        super().__init__()
        d_type = d_model // 4
        if d_type < 1:
            raise ValueError(
                f"d_model must be at least 4 for the electrode-type embedding, "
                f"got {d_model}."
            )
        self.coord_emb = _SinusoidalCoordEmbedding(d_model - d_type)
        self.type_emb = nn.Embedding(len(_MODALITIES), d_type)
        self.subtype_emb = nn.Embedding(len(_SUBTYPES), d_type)

    def forward(
        self,
        coords: torch.Tensor,
        coords_known: torch.Tensor,
        type_idx: torch.Tensor,
        subtype_idx: torch.Tensor,
        subtype_known: torch.Tensor,
    ) -> torch.Tensor:
        """Embed ``(n_chans, 3)`` coordinates and type indices into ``d_model``."""
        position = self.coord_emb(coords) * coords_known
        modality = self.type_emb(type_idx) + self.subtype_emb(subtype_idx) * (
            subtype_known
        )
        return torch.cat([position, modality], dim=-1)


class _SinusoidalCoordEmbedding(nn.Module):
    r"""Sinusoidal encoding of 3D electrode coordinates, following PopT.

    Each of the three axes is encoded on its own, exactly as a Transformer
    encodes a token position: the coordinate is divided by a geometric
    progression of :math:`n / 2` wavelengths, and every resulting angle is read
    out as a sine and a cosine,

    .. math::
        \mathbf{e}(p)_{2k} = \sin\left(\frac{s \, p}{\tau^{2k/n}}\right), \quad
        \mathbf{e}(p)_{2k+1} = \cos\left(\frac{s \, p}{\tau^{2k/n}}\right),

    for a coordinate :math:`p` in millimetres, with :math:`s = 2 \pi` ``scale``,
    :math:`\tau` the ``temperature`` and :math:`n` features per axis. The
    defaults make for a deliberately coarse bank: the shortest wavelength,
    256 mm, is already wider than a head, and the rest stretch to hundreds of
    metres, so they act as near-linear ramps. The code therefore says roughly
    where in the brain an electrode sits rather than separating neighbouring
    contacts. The three encodings are concatenated and right-padded with zeros
    to ``d_model``.

    Parameters
    ----------
    d_model : int
        Output width of the encoding.
    temperature : float
        Base of the frequency progression. Defaults to PopT's setting.
    scale : float
        Multiplier applied to the coordinates before encoding; its inverse is
        the shortest wavelength of the bank. Defaults to PopT's setting.
    """

    def __init__(
        self, d_model: int, temperature: float = 2000.0, scale: float = 1 / 256
    ):
        super().__init__()
        n_dim = 3
        # An even number of features per axis, so that every wavelength gets
        # both its sine and its cosine. What the three axes then leave short of
        # d_model is made up by zero padding.
        self.n_feats = d_model // n_dim // 2 * 2
        self.padding = d_model - self.n_feats * n_dim
        self.scale = scale * 2.0 * math.pi
        # The n_feats // 2 wavelengths of the bank, in geometric progression.
        dim_t = torch.arange(0, self.n_feats, 2, dtype=torch.float32)
        dim_t = temperature ** (dim_t / self.n_feats)
        self.register_buffer("dim_t", dim_t, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode coordinates of shape ``(..., 3)`` into ``(..., d_model)``."""
        # One angle per (axis, wavelength) pair.
        angles = (xyz * self.scale).unsqueeze(-1) / self.dim_t
        # Every wavelength is read out twice, as a sine and as a cosine, and
        # the stack puts the two of them side by side.
        sin = angles.sin()
        cos = angles.cos()
        pairs = torch.stack([sin, cos], dim=-1)
        # Flatten the three axes and their features into one vector, then pad.
        emb = pairs.flatten(start_dim=-3)
        emb = F.pad(emb, (0, self.padding))
        return emb


class _STCPE(nn.Module):
    """Spatio-temporal conditional positional embedding.

    Projects the token grid down to ``d_model // ratio``, runs a one-layer
    any-variate Transformer over every ``window``-wide temporal window (all
    channels at once), averages the overlapping window outputs, and projects
    back up. Sliding windows make the bias translation equivariant in time and
    the inner any-variate encoder makes it permutation equivariant in channels.

    Parameters
    ----------
    d_model : int
        Token embedding dimension of the grid.
    ratio : int
        Bottleneck ratio; the inner encoder runs at ``d_model // ratio``.
    window : int
        Width of the temporal window, in patches.
    activation : type[nn.Module]
        Activation layer class of the inner feed-forward block.
    n_chans : int
        Height of the token grid, i.e. the height of a window.
    n_patches : int
        Width of the token grid, which fixes how many windows cover it.
    """

    def __init__(
        self,
        d_model: int,
        ratio: int,
        window: int,
        activation: type[nn.Module],
        n_chans: int,
        n_patches: int,
        mup_attention: bool = True,
    ):
        super().__init__()
        inner_dim = d_model // ratio
        if inner_dim < 2 or inner_dim % 2:
            raise ValueError(
                f"d_model // stcpe_ratio must be even and at least 2, got "
                f"{inner_dim} (d_model={d_model}, stcpe_ratio={ratio})."
            )
        # d_model // 256 heads as in Table 7, walked down until the head
        # dimension is a whole even number (the rotary embedding needs it).
        inner_heads = max(1, inner_dim // 32)
        while inner_heads > 1 and (
            inner_dim % inner_heads or (inner_dim // inner_heads) % 2
        ):
            inner_heads -= 1
        self.window = window
        self.n_chans = n_chans
        # Full-height, window-wide patches of the (channel, patch) grid.
        # Zero-padding by window - 1 keeps a window centred on every patch, so
        # there are that many more windows than patches.
        self.kernel_size = (n_chans, window)
        self.stride = (1, 1)
        self.padding = (0, window - 1)
        self.grid_size = (n_chans, n_patches)
        self.n_windows = n_patches + window - 1
        # The feature axis is folded into the batch so that unfold and fold only
        # ever slide over time, and the window is full height so that every
        # window holds all the channels.
        self.fold_features = Rearrange(
            "batch chan patch dim -> (batch dim) 1 chan patch"
        )
        self.unfold_features = Rearrange(
            "(batch dim) 1 chan patch -> batch chan patch dim", dim=inner_dim
        )
        # One sequence per window for the inner encoder, and back again.
        self.windows_to_batch = Rearrange(
            "(batch dim) (chan window) n_win -> (batch n_win) (chan window) dim",
            dim=inner_dim,
            window=window,
        )
        self.batch_to_windows = Rearrange(
            "(batch n_win) (chan window) dim -> (batch dim) (chan window) n_win",
            n_win=self.n_windows,
            window=window,
        )
        self.down = nn.Linear(d_model, inner_dim)
        self.encoder = _AnyVariateEncoder(
            d_model=inner_dim,
            n_layers=1,
            num_heads=inner_heads,
            d_ff=4 * inner_dim,
            drop_prob=0.0,
            activation=activation,
            max_len=window,
            mup_attention=mup_attention,
        )
        self.up = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a token grid to a positional bias of the same shape."""
        x = self.down(x)
        flat = self.fold_features(x)
        unfolded = F.unfold(
            flat,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        windows = self.windows_to_batch(unfolded)

        # Inside a window, tokens are tagged by their channel and their offset
        # from the start of the window.
        var_id = torch.arange(self.n_chans, device=x.device).repeat_interleave(
            self.window
        )
        time_id = torch.arange(self.window, device=x.device).repeat(self.n_chans)
        encoded = self.encoder(windows, var_id=var_id, time_id=time_id)
        encoded = self.batch_to_windows(encoded)

        folded = F.fold(
            encoded,
            output_size=self.grid_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        # Average rather than sum the overlapping windows, as the reference
        # implementation does (the paper writes the un-normalised sum). The
        # divisor counts how many windows cover each patch.
        overlap = F.fold(
            torch.ones(
                1,
                self.n_chans * self.window,
                self.n_windows,
                dtype=x.dtype,
                device=x.device,
            ),
            output_size=self.grid_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        averaged = folded / overlap
        out = self.unfold_features(averaged)
        out = self.up(out)
        return out


class _AnyVariateEncoder(nn.Module):
    """Stack of any-variate Transformer blocks over flattened grid tokens.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_layers : int
        Number of blocks.
    num_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the feed-forward blocks.
    drop_prob : float
        Dropout rate.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks.
    max_len : int
        Largest temporal index the rotary tables must cover.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_ff: int,
        drop_prob: float,
        activation: type[nn.Module],
        max_len: int,
        mup_attention: bool = True,
    ):
        super().__init__()
        # The rotary tables are parameter-free, so all layers share one module
        # (``shared_time_qk_proj=True`` upstream); the binary channel bias is
        # learned and therefore per-layer.
        self.rotary = _RotaryEmbedding(d_model // num_heads, max_len)
        self.layers = nn.ModuleList(
            [
                _AnyVariateEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    drop_prob=drop_prob,
                    activation=activation,
                    rotary=self.rotary,
                    mup_attention=mup_attention,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``(batch, seq, d_model)`` tokens tagged by channel and patch.

        ``var_id`` and ``time_id`` are 1D tensors of length ``seq`` holding the
        channel and temporal-patch index of every token.
        """
        for layer in self.layers:
            x = layer(x, var_id=var_id, time_id=time_id)
        return self.norm(x)


class _AnyVariateEncoderLayer(nn.Module):
    """Pre-norm block: any-variate self-attention then a SwiGLU feed-forward.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the feed-forward block.
    drop_prob : float
        Dropout rate.
    activation : type[nn.Module]
        Activation layer class of the feed-forward block.
    rotary : _RotaryEmbedding
        Shared rotary embedding applied to the queries and keys.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        drop_prob: float,
        activation: type[nn.Module],
        rotary: _RotaryEmbedding,
        mup_attention: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.self_attn = _AnyVariateAttention(
            d_model=d_model,
            num_heads=num_heads,
            drop_prob=drop_prob,
            rotary=rotary,
            mup_attention=mup_attention,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn = _SwiGLUFeedForward(d_model, d_ff, drop_prob, activation)

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), var_id, time_id))
        return x + self.ffn(self.norm2(x))


class _AnyVariateAttention(nn.Module):
    """Self-attention with rotary temporal offsets and a binary channel bias.

    Implements the attention energy of [Han2025]_,

    .. math::
        E_{ij,mn} = (\\mathbf{W}^Q \\mathbf{x}_{i,m})^\\top \\mathbf{R}_{i-j}
        (\\mathbf{W}^K \\mathbf{x}_{j,n}) + u^{(1)} \\mathbb{1}_{\\{m = n\\}}
        + u^{(2)} \\mathbb{1}_{\\{m \\neq n\\}},

    with per-layer, per-head biases :math:`u^{(1)}, u^{(2)}`. Queries and keys
    are RMS-normalised per head before the rotation (QK-norm). The reference
    uses grouped-query attention configured with one head per group, i.e. plain
    multi-head attention, which is what we implement.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    drop_prob : float
        Attention dropout rate.
    rotary : _RotaryEmbedding
        Shared rotary embedding applied to the queries and keys.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        drop_prob: float,
        rotary: _RotaryEmbedding,
        mup_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # None lets SDPA use its default 1 / sqrt(head_dim).
        self.scale = 1.0 / self.head_dim if mup_attention else None
        self.drop_prob = drop_prob
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.channel_bias = nn.Embedding(2, num_heads)
        self.split_heads = Rearrange(
            "batch seq (heads dim) -> batch heads seq dim", heads=num_heads
        )
        self.merge_heads = Rearrange("batch heads seq dim -> batch seq (heads dim)")

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        query = self.rotary(self.q_norm(self.split_heads(self.q_proj(x))), time_id)
        key = self.rotary(self.k_norm(self.split_heads(self.k_proj(x))), time_id)
        value = self.split_heads(self.v_proj(x))

        # u(1) on the diagonal blocks (same electrode), u(2) off them. Only the
        # same/different distinction enters, never the channel index, which is
        # what makes the encoder channel-permutation equivariant.
        same_channel = var_id.unsqueeze(-1) == var_id.unsqueeze(-2)
        weight = self.channel_bias.weight  # (2, num_heads)
        bias = torch.where(
            same_channel[None, None],
            weight[1].view(1, -1, 1, 1),
            weight[0].view(1, -1, 1, 1),
        ).to(query.dtype)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.drop_prob if self.training else 0.0,
            scale=self.scale,
        )
        out = self.merge_heads(out)
        return self.out_proj(out)


class _RotaryEmbedding(nn.Module):
    """Interleaved rotary position embedding indexed by an explicit position id.

    Parameters
    ----------
    head_dim : int
        Dimension of an attention head; must be even.
    max_len : int
        Number of positions to tabulate.
    base : float
        Base of the geometric frequency progression.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}.")
        theta = 1.0 / torch.pow(
            base, torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), theta)
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack([-odd, even], dim=-1).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, position_id: torch.Tensor) -> torch.Tensor:
        """Rotate ``(batch, heads, seq, head_dim)`` by the angle of each position."""
        cos = self.cos[position_id].to(x.dtype)
        sin = self.sin[position_id].to(x.dtype)
        return cos * x + sin * self._rotate(x)


class _SwiGLUFeedForward(nn.Module):
    """Gated-linear-unit feed-forward block, without biases.

    Parameters
    ----------
    d_model : int
        Input and output dimension.
    d_ff : int
        Hidden dimension.
    drop_prob : float
        Dropout applied to the hidden and output activations.
    activation : type[nn.Module]
        Activation layer class applied to the gate.
    """

    def __init__(
        self, d_model: int, d_ff: int, drop_prob: float, activation: type[nn.Module]
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc_gate = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation()
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc_gate(x)
        gate = self.activation(gate)
        hidden = gate * self.fc1(x)
        hidden = self.dropout1(hidden)
        out = self.fc2(hidden)
        out = self.dropout2(out)
        return out

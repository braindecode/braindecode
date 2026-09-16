# Authors: Julien Gadonneix <145470783+julien-gadonneix@users.noreply.github.com>
"""DIVER-1: an any-variate iEEG foundation model.

Reimplementation of DIVER-1 (Han et al., 2025), "DIVER-1: Scaling Intracranial
EEG Foundation Models for Transferable Representations". The architecture is
transcribed from the authors' reference implementation, whose Transformer
encoder is adapted from Salesforce's MOIRAI / ``uni2ts`` (Copyright Salesforce,
Inc.), released under the Apache License, Version 2.0. The license this port
should carry follows from that provenance but is still to be confirmed with the
braindecode maintainers.
The braindecode reimplementation is pure-PyTorch (no ``mup``, no ``jaxtyping``)
and covers the downstream encoder only.

Original Authors: Han et al., Seoul National University
Braindecode Adaptation: Julien Gadonneix
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import PatchTokenizer

# Channel-modality vocabulary of the reference ``ChannelTypeEmbedding``: slot 0
# is scalp EEG, slot 1 intracranial EEG.
_MODALITIES = ("EEG", "iEEG")
# Electrode sub-modality vocabulary of the reference ``ChannelSubTypeEmbedding``:
# ECoG grids and strips, and SEEG depth electrodes.
_SUBTYPES = ("grid", "strip", "depth")
# FIFF channel-kind codes (mne.io.constants.FIFF), so chs_info can be read
# without importing mne.
_FIFF_SEEG, _FIFF_DBS, _FIFF_ECOG = 802, 803, 902


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
    the grid before the encoder; the global register acts as the CLS token.

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

    .. rubric:: Pre-trained weights

    None. The reference repository ships a placeholder in place of
    ``weights/ieeg_pretrained_weights.pt``, so no DIVER-1 checkpoint is publicly
    available and this port provides the architecture only.

    .. note::
        Numerical equivalence of the encoder features with the reference
        implementation has been verified layer by layer, for both patch-size
        variants, by transplanting a randomly initialised reference state dict
        (no checkpoint exists to verify against). The comparison requires
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
    chan_pos : array-like, optional
        MNI coordinates of each electrode in millimetres, shape
        ``(n_chans, 3)``. If omitted, they are read from the ``"loc"`` entries
        of ``chs_info`` (metres, MNE convention). Channels whose coordinates
        are missing, non-finite or exactly zero get a zero coordinate
        embedding.
    chan_modality : list of str, optional
        Per-channel recording modality, ``"EEG"`` or ``"iEEG"``. If omitted, it
        is derived from the ``"kind"`` entries of ``chs_info``.
    chan_subtype : list of str, optional
        Per-channel electrode sub-modality, ``"grid"``, ``"strip"`` or
        ``"depth"``; any other value (e.g. ``"unknown"``) gets a zero sub-type
        embedding. If omitted, it is derived from the ``"kind"`` entries of
        ``chs_info`` (SEEG maps to ``"depth"``, ECoG to ``"grid"``).
    pooling : {"flatten", "mean"}
        Token aggregation before the head. ``"flatten"`` reproduces the paper's
        finetuning protocol (a linear classifier on the flattened token grid);
        ``"mean"`` averages over channels and patches first, giving a head that
        is independent of ``n_chans`` and ``n_times``.
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
        chan_pos=None,
        chan_modality=None,
        chan_subtype=None,
        pooling: str = "flatten",
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

        # Patches are right-zero-padded when n_times is not a multiple of
        # patch_size (braindecode's default), so round up.
        self.n_patches = -(-self.n_times // patch_size)
        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size, n_times=self.n_times
        )
        self.patch_cnn = _PatchCNN(
            patch_size=patch_size,
            d_model=d_model,
            stride=cnn_stride,
            kernel_size=cnn_kernel_size,
            depth=cnn_depth,
        )

        self.spectral_emb = (
            _SpectralEmbedding(d_model, drop_prob) if use_spectral_emb else None
        )

        # Channel metadata: explicit arguments win, otherwise read chs_info.
        # Registered non-persistently because they describe the montage the
        # model was built for, not learned state.
        # (The buffers cannot be named after the constructor arguments:
        # EEGModuleMixin back-fills those as plain instance attributes.)
        coords, coords_known = self._resolve_chan_pos(chan_pos)
        self.register_buffer("chan_coords", coords, persistent=False)
        self.register_buffer("chan_coords_known", coords_known, persistent=False)
        type_idx, subtype_idx, subtype_known = self._resolve_chan_types(
            chan_modality, chan_subtype
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
            )
            if use_stcpe
            else None
        )

        # Learned register tokens, prepended as an extra patch column
        # (per-channel), an extra channel row (per-patch) and their corner (the
        # CLS token). The reference initialises all three as 0.02 * N(0, 1).
        self.patch_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.chan_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.encoder = _AnyVariateEncoder(
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            activation=activation,
            # One extra patch position for the register column.
            max_len=max(512, self.n_patches + 1),
        )

        head_in_features = (
            self.n_chans * self.n_patches * d_model if pooling == "flatten" else d_model
        )
        self.final_layer = nn.Linear(head_in_features, self.n_outputs)

    def reset_head(self, n_outputs):
        """Replace the linear classification head for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def _resolve_chan_pos(self, chan_pos) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve per-electrode MNI coordinates, in millimetres.

        Explicit ``chan_pos`` wins; otherwise the first three entries of each
        ``chs_info`` ``"loc"`` are used and converted from the MNE convention
        (metres) to the millimetres the sinusoidal encoding is tuned for.
        Channels whose coordinates are non-finite or exactly zero (MNE's two
        ways of spelling "no montage", as in
        :func:`~braindecode.models.util.has_valid_locations`) are flagged so
        their coordinate embedding is zeroed, which is how the paper handles
        unknown positions.
        """
        if chan_pos is not None:
            pos = torch.as_tensor(chan_pos, dtype=torch.float32)
            if pos.shape != (self.n_chans, 3):
                raise ValueError(
                    f"chan_pos must have shape ({self.n_chans}, 3), got "
                    f"{tuple(pos.shape)}."
                )
        else:
            try:
                chs_info = self.chs_info
            except ValueError:
                chs_info = None
            rows = []
            for i in range(self.n_chans):
                loc = chs_info[i].get("loc") if chs_info else None
                if loc is None or len(loc) < 3:
                    rows.append([float("nan")] * 3)
                    continue
                # MNE stores channel positions in metres; DIVER-1 encodes MNI
                # coordinates in millimetres.
                rows.append([float(v) * 1e3 for v in list(loc)[:3]])
            pos = torch.tensor(rows, dtype=torch.float32)
        known = torch.isfinite(pos).all(dim=-1, keepdim=True) & (pos != 0).any(
            dim=-1, keepdim=True
        )
        return torch.nan_to_num(pos), known.to(pos.dtype)

    def _resolve_chan_types(
        self, chan_modality, chan_subtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve the electrode modality and sub-modality of each channel.

        Explicit arguments win; otherwise both are derived from the ``"kind"``
        entries of ``chs_info``, accepting either MNE's FIFF integer codes or
        plain strings. Unlike the reference implementation, which infers the
        sub-modality from a site-specific table of electrode-group name
        prefixes, we key off the channel kind: SEEG contacts are depth
        electrodes and ECoG contacts are grids. Pass ``chan_subtype`` to
        distinguish grids from strips.
        """
        try:
            chs_info = self.chs_info
        except ValueError:
            chs_info = None
        kinds = [
            (chs_info[i].get("kind") if chs_info else None) for i in range(self.n_chans)
        ]

        if chan_modality is not None:
            modality = list(chan_modality)
            if len(modality) != self.n_chans:
                raise ValueError(
                    f"chan_modality must have {self.n_chans} entries, got "
                    f"{len(modality)}."
                )
            unknown = sorted({m for m in modality if m not in _MODALITIES})
            if unknown:
                raise ValueError(
                    f"chan_modality entries must be in {_MODALITIES}, got {unknown}."
                )
        else:
            modality = ["iEEG" if _is_intracranial(k) else "EEG" for k in kinds]

        if chan_subtype is not None:
            subtype = list(chan_subtype)
            if len(subtype) != self.n_chans:
                raise ValueError(
                    f"chan_subtype must have {self.n_chans} entries, got "
                    f"{len(subtype)}."
                )
        else:
            subtype = [_default_subtype(k) for k in kinds]

        type_idx = torch.tensor(
            [_MODALITIES.index(m) for m in modality], dtype=torch.long
        )
        known = [s in _SUBTYPES for s in subtype]
        subtype_idx = torch.tensor(
            [_SUBTYPES.index(s) if k else 0 for s, k in zip(subtype, known)],
            dtype=torch.long,
        )
        subtype_known = torch.tensor(known, dtype=torch.float32).unsqueeze(-1)
        return type_idx, subtype_idx, subtype_known

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Build the token grid ``(batch, n_chans, n_patches, d_model)``."""
        # Every term is added to the running grid, in the order of the reference
        # ``Embedder``: the spectral and positional terms see the CNN output,
        # and STCPE sees all of them (as in Eq. 1, which feeds it X).
        x = self.patch_cnn(self.patch_tokenizer(x))
        if self.spectral_emb is not None:
            x = x + self.spectral_emb(x)
        if self.chan_emb is not None:
            # (n_chans, d_model), broadcast over batch and patches.
            chan_emb = self.chan_emb(
                self.chan_coords,
                self.chan_coords_known,
                self.chan_type_idx,
                self.chan_subtype_idx,
                self.chan_subtype_known,
            )
            x = x + chan_emb[None, :, None, :]
        if self.stcpe is not None:
            x = x + self.stcpe(x)
        return x

    def _add_registers(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend the register row, column and corner to the token grid."""
        batch, n_chans, _, _ = x.shape
        patch_reg = self.patch_register[None].expand(batch, n_chans, -1, -1)
        x = torch.cat([patch_reg, x], dim=2)  # (batch, C, 1 + N, d_model)
        chan_reg = self.chan_register[None].expand(batch, -1, x.shape[2] - 1, -1)
        cls_reg = self.cls_register[None].expand(batch, -1, -1, -1)
        row = torch.cat([cls_reg, chan_reg], dim=2)  # (batch, 1, 1 + N, d_model)
        return torch.cat([row, x], dim=1)  # (batch, 1 + C, 1 + N, d_model)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Encode an iEEG batch into class logits (or encoder features).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If ``True``, return the encoder representation as
            ``{"features": tokens, "cls_token": cls}`` instead of the class
            logits (the unified braindecode foundation-model API), with
            ``tokens`` of shape ``(batch, n_chans * n_patches, d_model)`` and
            ``cls`` the global register token.

        Returns
        -------
        torch.Tensor | dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            ``{"features", "cls_token"}`` when ``return_features`` is set.
        """
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"DIVER1 was built for {self.n_chans} channels but got input with "
                f"{x.shape[1]}; rebuild the model for this montage."
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

        tokens = self._add_registers(self._embed(x))
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
            rearrange(tokens, "batch chans patches dim -> batch (chans patches) dim"),
            var_id=var_id,
            time_id=time_id,
        )
        tokens = rearrange(
            tokens,
            "batch (chans patches) dim -> batch chans patches dim",
            chans=n_chans,
            patches=n_patches,
        )
        # Drop the registers, keeping the CLS corner as the sequence summary.
        cls_token = tokens[:, 0, 0]
        tokens = tokens[:, 1:, 1:]

        if return_features:
            return {
                "features": rearrange(
                    tokens, "batch chans patches dim -> batch (chans patches) dim"
                ),
                "cls_token": cls_token,
            }

        if self.pooling == "flatten":
            pooled = tokens.flatten(start_dim=1)
        else:
            pooled = tokens.mean(dim=(1, 2))
        return self.final_layer(pooled)


def _is_intracranial(kind) -> bool:
    """Whether an MNE channel ``kind`` denotes an intracranial contact."""
    if isinstance(kind, str):
        return kind.lower() in ("ecog", "seeg", "dbs", "ieeg")
    if isinstance(kind, int):
        return kind in (_FIFF_SEEG, _FIFF_DBS, _FIFF_ECOG)
    return False


def _default_subtype(kind) -> str:
    """Electrode sub-modality implied by an MNE channel ``kind``."""
    if isinstance(kind, str):
        kind = kind.lower()
        if kind in ("seeg", "dbs"):
            return "depth"
        if kind == "ecog":
            return "grid"
    elif isinstance(kind, int):
        if kind in (_FIFF_SEEG, _FIFF_DBS):
            return "depth"
        if kind == _FIFF_ECOG:
            return "grid"
    return "unknown"


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
        self.d_model = d_model
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
        self.proj_in = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, n_chans, n_patches, patch_size)`` to ``d_model`` tokens."""
        batch, n_chans, n_patches, _ = x.shape
        x = F.pad(x, self.pad)
        # All (channel, patch) pairs are encoded independently: fold them into
        # the height axis of a single-channel 2D convolution.
        x = self.proj_in(x.reshape(batch, 1, n_chans * n_patches, -1))
        x = x.permute(0, 2, 1, 3)  # (batch, chans * patches, hidden, out_size)
        return x.reshape(batch, n_chans, n_patches, self.d_model)


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
        amplitude = torch.fft.rfft(x.float(), dim=-1, norm="forward").abs()
        return self.spectral_proj(amplitude.to(x.dtype))


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
    """Sinusoidal encoding of 3D electrode coordinates, following PopT.

    Each coordinate is scaled by ``scale``, expanded over a geometric
    progression of frequencies, and encoded as interleaved sines and cosines;
    the three coordinate encodings are concatenated and right-padded to
    ``d_model``.

    Parameters
    ----------
    d_model : int
        Output width of the encoding.
    temperature : float
        Base of the frequency progression.
    scale : float
        Multiplier applied to the coordinates before encoding.
    """

    def __init__(
        self, d_model: int, temperature: float = 2000.0, scale: float = 1 / 256
    ):
        super().__init__()
        n_dim = 3
        self.n_feats = d_model // n_dim // 2 * 2
        self.padding = d_model - self.n_feats * n_dim
        self.scale = scale * 2.0 * math.pi
        dim_t = torch.arange(self.n_feats, dtype=torch.float32)
        dim_t = temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.n_feats
        )
        self.register_buffer("dim_t", dim_t, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode coordinates of shape ``(..., 3)`` into ``(..., d_model)``."""
        divided = (xyz * self.scale).unsqueeze(-1) / self.dim_t
        emb = torch.stack([divided[..., 0::2].sin(), divided[..., 1::2].cos()], dim=-1)
        emb = emb.reshape(*xyz.shape[:-1], -1)
        return F.pad(emb, (0, self.padding))


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
    """

    def __init__(
        self, d_model: int, ratio: int, window: int, activation: type[nn.Module]
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
        self.down = nn.Linear(d_model, inner_dim)
        self.encoder = _AnyVariateEncoder(
            d_model=inner_dim,
            n_layers=1,
            num_heads=inner_heads,
            d_ff=4 * inner_dim,
            drop_prob=0.0,
            activation=activation,
            max_len=window,
        )
        self.up = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a token grid to a positional bias of the same shape."""
        batch, n_chans, n_patches, _ = x.shape
        x = self.down(x)
        inner_dim = x.shape[-1]
        # Full-height, window-wide patches of the (channel, patch) grid. The
        # feature axis is folded into the batch so unfold/fold only slide over
        # time; zero-padding by window - 1 keeps a window centred on every
        # patch.
        unfold_kwargs = {
            "kernel_size": (n_chans, self.window),
            "stride": (1, 1),
            "padding": (0, self.window - 1),
        }
        flat = rearrange(x, "batch chans patches dim -> (batch dim) 1 chans patches")
        unfolded = F.unfold(flat, **unfold_kwargs)
        n_windows = unfolded.shape[-1]

        var_id = torch.arange(n_chans, device=x.device).repeat_interleave(self.window)
        time_id = torch.arange(self.window, device=x.device).repeat(n_chans)
        encoded = self.encoder(
            rearrange(
                unfolded,
                "(batch dim) (chans window) n_win -> (batch n_win) (chans window) dim",
                batch=batch,
                dim=inner_dim,
                chans=n_chans,
                window=self.window,
            ),
            var_id=var_id,
            time_id=time_id,
        )
        encoded = rearrange(
            encoded,
            "(batch n_win) (chans window) dim -> (batch dim) (chans window) n_win",
            batch=batch,
            n_win=n_windows,
            chans=n_chans,
            window=self.window,
        )

        folded = F.fold(encoded, output_size=(n_chans, n_patches), **unfold_kwargs)
        # Average rather than sum the overlapping windows, as the reference
        # implementation does (the paper writes the un-normalised sum). The
        # divisor counts how many windows cover each patch.
        overlap = F.fold(
            torch.ones(
                1, n_chans * self.window, n_windows, dtype=x.dtype, device=x.device
            ),
            output_size=(n_chans, n_patches),
            **unfold_kwargs,
        )
        out = rearrange(
            folded / overlap,
            "(batch dim) 1 chans patches -> batch chans patches dim",
            batch=batch,
            dim=inner_dim,
        )
        return self.up(out)


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
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = _RMSNorm(d_model)

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
    ):
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.self_attn = _AnyVariateAttention(
            d_model=d_model, num_heads=num_heads, drop_prob=drop_prob, rotary=rotary
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm2 = _RMSNorm(d_model)
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
        self, d_model: int, num_heads: int, drop_prob: float, rotary: _RotaryEmbedding
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.drop_prob = drop_prob
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = _RMSNorm(self.head_dim)
        self.k_norm = _RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.channel_bias = nn.Embedding(2, num_heads)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x, "batch seq (heads dim) -> batch heads seq dim", heads=self.num_heads
        )

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        query = self.rotary(self.q_norm(self._split_heads(self.q_proj(x))), time_id)
        key = self.rotary(self.k_norm(self._split_heads(self.k_proj(x))), time_id)
        value = self._split_heads(self.v_proj(x))

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
        )
        return self.out_proj(
            rearrange(out, "batch heads seq dim -> batch seq (heads dim)")
        )


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
        hidden = self.activation(self.fc_gate(x)) * self.fc1(x)
        return self.dropout2(self.fc2(self.dropout1(hidden)))


class _RMSNorm(nn.Module):
    """Root-mean-square layer normalisation with a learned gain.

    :class:`torch.nn.RMSNorm` is only available from PyTorch 2.4, while
    braindecode supports ``torch>=2.0``; this equivalent keeps the model
    importable on older PyTorch, as done in
    :class:`~braindecode.models.ZUNA` and :class:`~braindecode.models.REVE`.

    Parameters
    ----------
    normalized_shape : int
        Size of the trailing dimension to normalise.
    eps : float
        Term added to the mean square for numerical stability.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight

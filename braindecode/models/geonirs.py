# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
#
# Adapted from https://github.com/GeoNIRS/GeoNIRS (81f8995; paper arXiv:2610.17845).

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import (
    check_ch_mask,
    pad_or_crop_to_patches,
    patch_count,
    scaled_dot_product_attention,
)
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr
from braindecode.modules import FeedForwardBlock


class GeoNIRS(EEGModuleMixin, nn.Module):
    r"""GeoNIRS from Li et al. (2026).

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. figure:: ../_static/model/geonirs_arch.png
       :align: center
       :alt: GeoNIRS framework
       :width: 1000px

       Overview of GeoNIRS: HbO and HbR temporal encoders, a geometry encoder, the
       geometry-aware channel compressor and the bidirectional cross-chromophore
       reconstruction used for pretraining. Reproduced from Fig. 1 of
       :footcite:`li2026geonirs` (copyright of the authors).

    .. versionadded:: 2.1

    GeoNIRS is an fNIRS foundation model pretrained on unlabelled recordings
    made with different probe layouts :footcite:`li2026geonirs`. A channel
    index means a different cortical location in each layout, so a channel is
    known only by the positions of its source and detector, and any set of
    channels is compressed into a fixed number of latent channels. Slow
    hemodynamic signals let a model fill a masked span of one signal by
    interpolation, so GeoNIRS is pretrained to rebuild one chromophore from
    the other. This class is the pretrained encoder of Figure 1 of the paper
    with the paper's linear head; the pretraining decoders are not part of it.

    .. rubric:: Architecture Overview

    For :math:`C` HbO/HbR pairs at 5 Hz, cut into :math:`P` patches of
    :math:`S` samples::

        x                                         (batch, 2C, n_times)
        - temporal encoders, one per chromophore  -> 2 x (batch, C, P, D)
        - geometry encoder                        -> (batch, C, D)
        - channel compressor, M latent queries    -> 2 x (batch, M, P, D)
        - chromophore projections, D -> F         -> (batch, 2, M, P, F)
        - read-out: pool="flatten"                -> (batch, 2 * M * P * F)
                    pool="mean_std"               -> (batch, 4 * M * F)
        - head_dropout, final_layer               -> (batch, n_outputs)

    GeoNIRS-Base on a 12 s window (``n_times=60``) gives tokens of shape
    ``(batch, 2, 16, 6, 64)``, read out as 12288 or 4096 features.

    .. rubric:: Macro Components

    - ``GeoNIRS.hbo_encoder``, ``GeoNIRS.hbr_encoder`` **Temporal encoders**
      (§2.2)

      *Operations.* Each patch of :math:`S` samples becomes a normalised
      :math:`D`-dimensional token plus a learnable embedding of its index; a
      pre-norm Transformer attends over the patches of one channel.

      *Role.* :math:`H^O` and :math:`H^R`, one encoder per chromophore. The
      channels share the weights, so any number of channels is accepted.

    - ``GeoNIRS.geometry_encoder`` **Geometry encoder** (Appendix B.2)

      *Operations.* Channel :math:`c` is described by the midpoint and the
      offset of its source :math:`s_c` and detector :math:`d_c`,

      .. math::

          g_c = [\mathrm{mid}_c, \Delta_c], \qquad
          \mathrm{mid}_c = \frac{s_c + d_c}{2}, \qquad \Delta_c = d_c - s_c,

      centred on the montage and divided by ``geometry_scale``, 25.46 mm by
      default. An MLP and one Transformer layer over the channels give
      :math:`G`.

      *Role.* The only identity of a channel: there is no channel index
      embedding.

    - ``GeoNIRS.channel_compressor`` **Geometry-aware channel compressor**
      (§2.3)

      *Operations.* At each patch :math:`p`, :math:`M` learnable latent
      queries :math:`L` attend over the channels. The keys hold the mean of
      the two chromophores plus the geometry, and each chromophore keeps its
      own values:

      .. math::

          \bar{H}_p = \tfrac{1}{2}\left(H^O_{:,p} + H^R_{:,p}\right), \quad
          Q = W_Q\,\mathrm{LN}(L), \quad
          K_p = W_K\,\mathrm{LN}\left(\bar{H}_p + G\right), \quad
          A_p = \mathrm{softmax}\left(\frac{Q K_p^\top}{\sqrt{d_h}}\right),
          \quad Z^s_p = A_p\,W_V\,\mathrm{LN}\left(H^s_{:,p}\right),
          \quad s \in \{O, R\}.

      The invalid channels are masked before the softmax; an output
      projection and a feed-forward block follow.

      *Role.* Maps any montage onto :math:`M` latent channels, with one
      attention map for HbO and HbR.

    - ``GeoNIRS.hbo_projection``, ``GeoNIRS.hbr_projection``
      **Chromophore projections** (§2.4)

      *Operations.* Residual blocks that change the width of each token from
      :math:`D` to :math:`F` through the widths of ``projection_dims``,
      without mixing latent channels or patches.

      *Role.* :math:`Z^O` and :math:`Z^R`, the tokens of the read-out.

    - ``GeoNIRS.final_layer`` **Head** (§2.5, Appendix B.5.2)

      *Operations.* ``pool="flatten"`` flattens the tokens of both
      chromophores, the head of the paper without its standardization by
      training-set statistics; ``pool="mean_std"`` takes their mean and
      standard deviation over the valid patches, the features of the
      reference's ``mean_std_linear`` head. Dropout and a linear layer follow.

      *Role.* The paper's linear head on the tokens. The feature size of
      ``"flatten"`` grows with the window length; that of ``"mean_std"`` does
      not.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** A patch is :math:`S = 10` samples, 2 s at 5 Hz. The
      positional embedding has a row for each of the 15 patches of the 30 s
      pretraining window, more for a longer ``n_times``. The compressor keeps
      the patch axis.
    - **Spatial.** A channel is known only through :math:`g_c`, in the keys of
      the compressor, so any number of channels in any layout maps to the same
      :math:`M` latent channels. Zeroing or shuffling the geometry at
      inference lowers the task-averaged balanced accuracy by 2.34 and 3.24
      points (Table 3).
    - **Spectral.** No frequency decomposition.
    - **Chromophores.** The rows are HbO and HbR concentration changes (no
      HbT), each with its own temporal encoder and projection.

    .. rubric:: Additional Mechanisms

    - **Input.** One ``(batch, 2C, n_times)`` tensor of HbO/HbR pairs, stacked
      or interleaved (the MNE order), see ``channel_order``, instead of the
      reference's batch of ``hbo``, ``hbr``, validity masks and ``geometry``.
      The geometry comes from ``chs_info`` (``loc[3:9]``, centred and scaled as
      the reference's ``scripts/build_fair_input.py``; a pair without positions
      gets zeros) or from ``forward(geometry=...)``.
    - **Validity masks.** A non-finite sample (NaN or inf) is missing, where
      the reference zero-fills and flags it. A patch needs
      ``minimum_patch_valid_fraction`` of its samples; invalid patches and
      channels are masked out of the attention and give zero tokens.
      ``ch_mask``, an addition, marks the real pairs of a batch padded with
      channels.
    - **Scope.** ``on_non_divisible="pad"``/``"crop"`` are additions; the
      decoders, the temporal masking and LoRA are not part of the class.

    .. rubric:: Pretrained Weights

    The authors' GeoNIRS-Base and GeoNIRS-Large (``checkpoint/`` of
    `GeoNIRS/GeoNIRS@81f8995
    <https://github.com/GeoNIRS/GeoNIRS/tree/81f8995e5c3aa6b70e55812b194fc04e4e7ea5b7/checkpoint>`_)
    are on the Hugging Face Hub in braindecode's format, at
    ``braindecode/geonirs-base-pretrained`` (the defaults) and
    ``braindecode/geonirs-large-pretrained`` (``embed_dim=384``, ``depth=6``,
    ``num_heads=12``, ``feature_dim=128`` and
    ``projection_dims=(192, 96)``). GeoNIRS-Base was pretrained on 8
    datasets (383.30 h) and has 4,443,462 parameters without the head;
    GeoNIRS-Large on 22 datasets (827.04 h), with 24,394,310. Both
    repositories hold the encoder, so ``from_pretrained`` builds the head for
    your ``n_times`` and ``n_outputs``. The weights expect HbO and HbR
    concentration changes (modified Beer-Lambert law, no band-pass filter,
    Appendix B.1) at 5 Hz, each standardized per recording, in windows of at
    most 150 samples. On the paper's windows (1628 for DS42, 1190 for DS19)
    this class gives the authors' encoder features (max-abs difference 0.0,
    Base and Large, eval mode). A linear probe of the frozen GeoNIRS-Base (a
    re-implementation of the authors' probe, with the paper's folds and seeds)
    gives macro-F1 0.7039 and 0.5304 on DS42 and DS19, against 0.7018 and
    0.5598 in the paper and 0.7000 and 0.5599 for the authors' pipeline at
    ``81f8995`` (:gh:`1591`).

    .. code-block:: python

        from braindecode.models import GeoNIRS

        model = GeoNIRS.from_pretrained(
            "braindecode/geonirs-base-pretrained",
            chs_info=raw.info["chs"],
            n_outputs=2,
            n_times=60,
        )

    Parameters
    ----------
    patch_size : int, default=10
        Samples per patch, :math:`S`: 2 s at 5 Hz.
    max_patches : int, default=15
        Rows of the temporal positional embedding: the 15 patches of a 30 s
        pretraining window, raised to the patch count of ``n_times``.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        ``n_times`` not a multiple of ``patch_size``: refuse (as the
        reference); pad with missing samples, to which the validity rule
        applies; or crop.
    embed_dim : int, default=192
        Token width, :math:`D` ("Embedding width", Table 6).
    num_heads : int, default=8
        Heads of the temporal Transformers (Table 6).
    depth : int, default=4
        Layers of each temporal Transformer (Table 6).
    mlp_ratio : float, default=4.0
        Feed-forward width of the temporal Transformers, over ``embed_dim``.
    geometry_hidden_dim : int or None, default=None
        Hidden width of the geometry MLP; ``None`` is ``embed_dim`` (Table 6).
    geometry_heads : int or None, default=None
        Heads of the geometry Transformer layer; ``None`` is ``num_heads``.
    geometry_mlp_ratio : float, default=2.0
        Feed-forward width of the geometry Transformer layer, over
        ``embed_dim`` (fixed at 2 in the reference).
    geometry_depth : int, default=1
        Layers of the geometry Transformer (``geometry_spatial_layers`` in the
        reference; the released weights have one).
    geometry_scale : float, default=0.02546058255683924
        Scale, in metres, that divides the centred source and detector
        coordinates of ``chs_info``: 25.46 mm, the ``GEOMETRY_SCALE_M`` of the
        authors' ``scripts/build_fair_input.py`` (81f8995). A ``geometry``
        argument of :meth:`forward` is taken as given.
    n_latent_channels : int, default=16
        Learnable latent queries, :math:`M`.
    compressor_heads : int or None, default=None
        Heads of the compressor's attention; ``None`` is ``num_heads``.
    compressor_mlp_ratio : float, default=2.0
        Width of the compressor's feed-forward block, over ``embed_dim``
        (fixed at 2 in the reference).
    feature_dim : int, default=64
        Token width after the projections, :math:`F` (Table 6).
    projection_dims : tuple of int, default=(128, 96)
        Inner token widths of the chromophore projections: the residual blocks
        run over ``(embed_dim, *projection_dims, feature_dim)``, one block per
        step, 192 -> 128 -> 96 -> 64 by default (Table 6 of the paper).
        GeoNIRS-Large uses ``(192, 96)``.
    use_channel_geometry : bool, default=True
        Add the geometry :math:`G` to the keys of the compressor. ``False``
        builds no geometry encoder: the "w/o Geometry" variant of Table 4.
    pool : {"flatten", "mean_std"}, default="flatten"
        Read-out of the ``(batch, 2, M, P, F)`` tokens: all of them flattened,
        or their mean and standard deviation over the valid patches.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or ``[HbO_1, HbR_1, ...]``; ``None``
        reads it from ``chs_info`` (stacked without it).
    drop_prob : float, default=0.1
        Dropout in the backbone, 0.1 in both checkpoints.
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer``, :math:`p = 0.1` in Appendix B.5.2.
    minimum_patch_valid_fraction : float, default=0.8
        Share of measured samples a patch needs to be valid, 80 % in
        Appendix B.1.2.
    activation : type[nn.Module], default=nn.GELU
        Activation of the backbone; GELU in the reference.
    time_embedding_init_std : float, default=0.02
        Standard deviation of the initial temporal position embeddings (the
        reference's ``time_embedding_init_std``).
    latent_init_std : float, default=0.02
        Standard deviation of the initial latent queries (fixed at 0.02 in the
        reference).
    residual_init_scale : float, default=0.1
        Initial value of the learnable weight of the update path of each
        projection block, so that a block starts close to its linear skip
        (fixed at 0.1 in the reference).

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "fixed"  # fixed: final_layer.weight is sized by n_times
    _channel_axis = "fixed"  # fixed: default_geometry is sized by n_chans

    def __init__(
        self,
        # Braindecode parameters
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        # Model-specific parameters
        patch_size: int = 10,
        max_patches: int = 15,
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
        embed_dim: int = 192,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        geometry_hidden_dim: int | None = None,
        geometry_heads: int | None = None,
        geometry_mlp_ratio: float = 2.0,
        geometry_depth: int = 1,
        geometry_scale: float = 0.02546058255683924,
        n_latent_channels: int = 16,
        compressor_heads: int | None = None,
        compressor_mlp_ratio: float = 2.0,
        feature_dim: int = 64,
        projection_dims: tuple[int, ...] = (128, 96),
        use_channel_geometry: bool = True,
        pool: Literal["flatten", "mean_std"] = "flatten",
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.1,
        head_drop_prob: float = 0.1,
        minimum_patch_valid_fraction: float = 0.8,
        activation: type[nn.Module] = nn.GELU,
        time_embedding_init_std: float = 0.02,
        latent_init_std: float = 0.02,
        residual_init_scale: float = 0.1,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Parameters
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.embed_dim = embed_dim
        self.n_latent_channels = n_latent_channels
        self.feature_dim = feature_dim
        self.pool = pool
        # None takes the value of the temporal encoders, as in both checkpoints
        self.geometry_hidden_dim = geometry_hidden_dim or embed_dim
        self.geometry_heads = geometry_heads or num_heads
        self.compressor_heads = compressor_heads or num_heads
        # Measured samples a valid patch needs: 8 of 10 with the defaults
        self.min_valid_samples = math.ceil(
            patch_size * float(minimum_patch_valid_fraction)
        )

        # Checkers
        # The rows are HbO/HbR pairs of the same source-detector channels
        if self.n_chans % 2:
            raise ValueError("GeoNIRS needs HbO/HbR pairs: n_chans must be even.")
        if pool not in ("flatten", "mean_std"):
            raise ValueError(f"pool must be 'flatten' or 'mean_std', got {pool!r}.")
        # Every attention splits the embed_dim features evenly over its heads
        heads = {"num_heads": num_heads, "compressor_heads": self.compressor_heads}
        if use_channel_geometry:
            heads["geometry_heads"] = self.geometry_heads
        for name, n_heads in heads.items():
            if embed_dim % n_heads:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by {name} "
                    f"({n_heads}): a head takes embed_dim / {name} features."
                )
        # A fraction above 1 would drop every patch and give all-zero tokens
        if not 0.0 <= minimum_patch_valid_fraction <= 1.0:
            raise ValueError(
                "minimum_patch_valid_fraction must be in [0, 1], got "
                f"{minimum_patch_valid_fraction}."
            )
        # A scale of 0 divides the coordinates by zero: NaN geometry tokens
        if geometry_scale <= 0.0:
            raise ValueError(f"geometry_scale must be positive, got {geometry_scale}.")
        # Where the HbO and HbR rows are; raises when chs_info does not list
        # HbO/HbR pairs
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        # P, the patches of one window; raises for a bad on_non_divisible, and
        # for a window the patches do not tile when it is "error"
        self.n_patches = patch_count(self.n_times, patch_size, on_non_divisible)

        # Layers
        # Temporal encoders: H^O and H^R, a token per patch of each pair. The
        # positional embedding needs a row per patch of the window.
        self.hbo_encoder = _TemporalEncoder(
            patch_size=self.patch_size,
            max_patches=max(max_patches, self.n_patches),
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
            init_std=time_embedding_init_std,
            activation=activation,
        )
        self.hbr_encoder = _TemporalEncoder(
            patch_size=self.patch_size,
            max_patches=max(max_patches, self.n_patches),
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
            init_std=time_embedding_init_std,
            activation=activation,
        )

        # Geometry encoder: G, a token per pair from its geometry g_c. The
        # geometry ablation has none.
        self.geometry_encoder = None
        if use_channel_geometry:
            self.geometry_encoder = _GeometryEncoder(
                embed_dim=self.embed_dim,
                hidden_dim=self.geometry_hidden_dim,
                num_heads=self.geometry_heads,
                mlp_ratio=geometry_mlp_ratio,
                drop_prob=drop_prob,
                activation=activation,
                depth=geometry_depth,
            )

        # Channel compressor: the pairs of each patch onto M latent channels
        self.channel_compressor = _ChannelCompressorAttention(
            embed_dim=self.embed_dim,
            n_latent_channels=self.n_latent_channels,
            num_heads=self.compressor_heads,
            mlp_ratio=compressor_mlp_ratio,
            drop_prob=drop_prob,
            init_std=latent_init_std,
            activation=activation,
        )

        # Chromophore projections: Z^O and Z^R, from D to F through the widths
        # of projection_dims
        widths = (embed_dim, *projection_dims, feature_dim)
        self.hbo_projection = _ChromophoreProjection(
            dims=widths,
            drop_prob=drop_prob,
            init_scale=residual_init_scale,
            activation=activation,
        )
        self.hbr_projection = _ChromophoreProjection(
            dims=widths,
            drop_prob=drop_prob,
            init_scale=residual_init_scale,
            activation=activation,
        )

        # Geometry g_c of the pairs of chs_info, used when forward gets no
        # geometry. Derived from chs_info, so not saved with the weights.
        self.register_buffer(
            "default_geometry",
            torch.from_numpy(
                _optode_geometry(self._chs_info, self.channel_order, geometry_scale)
            ),
            persistent=False,
        )

        # Patches of the sample validity mask, to count the samples of a patch
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=self.patch_size,
        )

        # Read-out of the (batch, 2, M, P, F) tokens: every token, or their
        # mean and standard deviation over the valid patches
        self.flatten_foundation = Rearrange(
            "batch chrom latents patches dim -> batch (chrom latents patches dim)"
        )
        self.mean_std = _ValidMeanStd()
        if self.pool == "flatten":
            n_features = 2 * n_latent_channels * self.n_patches * feature_dim
        else:
            n_features = 4 * n_latent_channels * feature_dim

        # Head: dropout and a linear layer
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(n_features, self.n_outputs)
        self._n_times_step = self.patch_size
        self._snaps_in_prepare_inputs = False
        self._min_n_times = 1 if self.on_non_divisible == "pad" else self.patch_size
        if self.pool != "flatten":
            self._time_axis = "free"

    def _accepts_any_n_chans(self, args: tuple, kwargs: dict) -> bool:
        if super()._accepts_any_n_chans(args, kwargs):
            return True
        geom = self._call_arg(args, kwargs, "geometry")
        return geom is not None or self.geometry_encoder is None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        geometry: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            HbO/HbR signals of shape ``(batch, n_chans, n_times)``. A NaN
            marks a missing sample.
        return_features : bool, default=False
            Return the features the head reads instead of the outputs.
        geometry : torch.Tensor, optional
            Geometry vector of each pair, ``(n_pairs, 6)`` or ``(batch,
            n_pairs, 6)``; see :meth:`encode`.
        ch_mask : torch.Tensor, optional
            ``(batch, n_chans)`` mask of the real channels of a padded batch;
            see :meth:`encode`.

        Returns
        -------
        torch.Tensor or dict
            Outputs of shape ``(batch, n_outputs)``. With
            ``return_features=True``, ``{"features": Tensor, "cls_token":
            None}`` with the ``(batch, n_features)`` input of ``final_layer``.
        """
        # Tokens of the two chromophores and the patches that hold data
        # (batch, n_chans, n_times) -> (batch, 2, M, P, F) and (batch, M, P)
        encoded = self.encode(x, geometry=geometry, ch_mask=ch_mask)
        x = encoded["foundation"]

        if self.pool == "flatten":
            # Every token, as the head of the paper reads them
            # (batch, 2, M, P, F) -> (batch, 2 * M * P * F)
            x = self.flatten_foundation(x)
        else:
            # Mean and standard deviation over the valid patches
            # (batch, 2, M, P, F) -> (batch, 4 * M * F)
            x = self.mean_std(x, encoded["valid"])

        # Dropout before the head: the features are the head's input
        x = self.head_dropout(x)
        if return_features:
            return {"features": x, "cls_token": None}

        # Linear head
        # (batch, n_features) -> (batch, n_outputs)
        x = self.final_layer(x)
        return x

    def encode(
        self,
        x: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a window into the latent tokens of the two chromophores.

        Parameters
        ----------
        x : torch.Tensor
            HbO/HbR signals of shape ``(batch, n_chans, n_times)``. A NaN
            marks a missing sample.
        geometry : torch.Tensor, optional
            Geometry vector :math:`g_c` of each pair, ``(n_pairs, 6)`` or
            ``(batch, n_pairs, 6)``. ``None`` takes the one of ``chs_info``.
            Not read with ``use_channel_geometry=False``.
        ch_mask : torch.Tensor, optional
            ``(batch, n_chans)`` mask of the real channels of a padded batch.
            It must keep whole HbO/HbR pairs: with the stacked order, the same
            pair slots in both halves. :func:`~braindecode.datasets.pad_channels_collate`
            pads each window at the end, which keeps the pairs whole in the
            interleaved order.

        Returns
        -------
        dict of torch.Tensor
            ``"foundation"``: :math:`Z^O` and :math:`Z^R`, ``(batch, 2,
            n_latent_channels, n_patches, feature_dim)``. ``"valid"``:
            boolean ``(batch, n_latent_channels, n_patches)``, ``False`` for
            the patches that have no valid channel (their tokens are zero).
        """
        # Real pairs of a padded batch: both rows of a pair flagged alike
        # (batch, n_chans) -> (batch, n_pairs)
        pair_mask: Optional[torch.Tensor] = None
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None:
            hbo_mask, hbr_mask = fnirs_split_hbo_hbr(ch_mask, self.channel_order)
            if not torch.equal(hbo_mask, hbr_mask):
                raise ValueError("ch_mask needs whole HbO/HbR pairs.")
            pair_mask = hbo_mask

        # Pad or crop the window to whole patches of S samples
        # (batch, n_chans, n_times) -> (batch, n_chans, P * S)
        n_times = x.shape[-1]
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)

        # Separate the chromophores of the source-detector pairs
        # (batch, n_chans, P * S) -> 2 x (batch, n_pairs, P * S)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape

        # Measured samples: both chromophores finite, and not padding
        # (batch, n_pairs, P * S)
        finite = torch.isfinite(hbo) & torch.isfinite(hbr)
        finite[:, :, n_times:] = False
        # Valid pairs: real, with at least one measured sample
        # (batch, n_pairs)
        channel_valid = finite.any(dim=-1)
        if pair_mask is not None:
            channel_valid = pair_mask & channel_valid
        # Valid samples: the measured samples of the valid pairs
        # (batch, n_pairs, P * S)
        sample_valid = finite & channel_valid[..., None]

        # Missing samples set to zero
        # 2 x (batch, n_pairs, P * S)
        zeros = torch.zeros_like(hbo)
        hbo = torch.where(sample_valid, hbo, zeros)
        hbr = torch.where(sample_valid, hbr, zeros)

        # Valid patches: enough measured samples, in a valid pair
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, S) -> (batch, n_pairs, P)
        patch_valid = self.split_patches(sample_valid)
        patch_valid = patch_valid.sum(dim=-1) >= self.min_valid_samples
        patch_valid = patch_valid & channel_valid[..., None]

        # Geometry tokens G, from the geometry of chs_info or the argument
        # (n_pairs, 6) or (batch, n_pairs, 6) -> (batch, n_pairs, D)
        geometry_embedding: Optional[torch.Tensor] = None
        if self.geometry_encoder is not None:
            if geometry is None:
                pair_geometry = self.default_geometry
                if pair_geometry.numel() == 0:
                    raise ValueError(
                        "GeoNIRS needs the channel geometry: chs_info with "
                        "source/detector locations (loc[3:9]) or the geometry "
                        "argument."
                    )
                if pair_geometry.shape[0] != n_pairs:
                    raise ValueError(
                        "x and chs_info have different channel counts: pass the "
                        "geometry of the HbO/HbR pairs of x."
                    )
            else:
                pair_geometry = geometry
                pair_shape = [n_pairs, self.default_geometry.shape[-1]]
                if list(pair_geometry.shape[-2:]) != pair_shape:
                    raise ValueError(
                        "geometry and x have different channel counts, or "
                        "geometry is not 6-D: pass (n_pairs, 6) or (batch, "
                        "n_pairs, 6), one row per HbO/HbR pair of x."
                    )
            pair_geometry = pair_geometry.to(device=x.device, dtype=hbo.dtype)
            pair_geometry = pair_geometry.expand(batch_size, -1, -1)
            geometry_embedding = self.geometry_encoder(pair_geometry, channel_valid)

        # Temporal encoders: a token per patch of each pair, H^O and H^R
        # 2 x (batch, n_pairs, P * S) -> 2 x (batch, n_pairs, P, D)
        hbo = self.hbo_encoder(hbo, patch_valid)
        hbr = self.hbr_encoder(hbr, patch_valid)

        # Channel compressor: the pairs of each patch onto M latent channels
        # 2 x (batch, n_pairs, P, D) -> 2 x (batch, M, P, D)
        hbo, hbr = self.channel_compressor(hbo, hbr, geometry_embedding, patch_valid)

        # Patches that keep their tokens: those with a valid pair
        # (batch, n_pairs, P) -> (batch, P) -> (batch, 1, P, 1)
        time_valid = patch_valid.any(dim=1)
        valid_scale = time_valid[:, None, :, None].to(hbo.dtype)

        # Chromophore projections, zero at the patches without tokens: Z^O, Z^R
        # 2 x (batch, M, P, D) -> 2 x (batch, M, P, F)
        hbo = self.hbo_projection(hbo)
        hbo = hbo * valid_scale
        hbr = self.hbr_projection(hbr)
        hbr = hbr * valid_scale

        # The two chromophores side by side
        # 2 x (batch, M, P, F) -> (batch, 2, M, P, F)
        foundation = torch.stack([hbo, hbr], dim=1)

        # Validity of the tokens, the same for every latent channel
        # (batch, P) -> (batch, M, P)
        valid = time_valid[:, None].expand(-1, self.n_latent_channels, -1)
        return {"foundation": foundation, "valid": valid}

    def reset_head(self, n_outputs: int) -> None:
        """Replace ``final_layer`` by a new one with ``n_outputs`` outputs.

        Parameters
        ----------
        n_outputs : int
            New number of outputs.
        """
        self._set_n_outputs(n_outputs)
        # The read-out does not change, so the head keeps its input width
        n_features = self.final_layer.in_features
        self.final_layer = nn.Linear(n_features, self.n_outputs)


class _TemporalEncoder(nn.Module):
    """Temporal encoder of one chromophore: a token per patch and a
    Transformer over the patches of each channel.

    For ``(batch, n_pairs, n_patches * patch_size)`` signals whose missing
    samples are zero:

    1. cut each channel into patches, ``(batch, n_pairs, n_patches,
       patch_size)``;
    2. project each patch to a token, LayerNorm it and add the learnable
       embedding of its index, ``(batch, n_pairs, n_patches, embed_dim)``;
    3. a pre-norm Transformer over the patches of each channel, the invalid
       patches masked out of the attention;
    4. a LayerNorm, the invalid patches zero, ``(batch, n_pairs, n_patches,
       embed_dim)``.

    The channels share the weights, so any number of channels is accepted.

    Parameters
    ----------
    patch_size : int
        Samples per patch, :math:`S`.
    max_patches : int
        Rows of the temporal positional embedding.
    embed_dim : int
        Token width, :math:`D`.
    num_heads : int
        Heads of the Transformer.
    depth : int
        Layers of the Transformer.
    mlp_ratio : float
        Feed-forward width of the Transformer, over ``embed_dim``.
    drop_prob : float
        Dropout of the Transformer.
    init_std : float
        Standard deviation of the initial positional embedding.
    activation : type[nn.Module], default=nn.GELU
        Activation of the Transformer.
    """

    def __init__(
        self,
        patch_size: int,
        max_patches: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        drop_prob: float,
        init_std: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.init_std = init_std

        # Patches of each channel: the time axis becomes (patches, samples)
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_size,
        )
        # A normalised D-wide token per patch of S samples
        self.patch_projection = nn.Linear(patch_size, embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        # Learnable embedding of the patch index
        self.time_embedding = nn.Embedding(max_patches, embed_dim)
        # Pre-norm Transformer, with a final LayerNorm, over the patches of
        # one channel
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=round(embed_dim * mlp_ratio),
                dropout=drop_prob,
                activation=activation(),
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(embed_dim)

        # Each channel becomes a sequence of patches for the Transformer
        self.merge_pairs = Rearrange(
            "batch pairs patches dim -> (batch pairs) patches dim"
        )
        self.merge_pair_mask = Rearrange("batch pairs patches -> (batch pairs) patches")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.time_embedding.weight, std=self.init_std)

    def forward(self, x: torch.Tensor, patch_valid: torch.Tensor) -> torch.Tensor:
        """Encode the patches of each channel.

        Parameters
        ----------
        x : torch.Tensor
            One chromophore, missing samples set to zero, of shape
            ``(batch, n_pairs, n_patches * patch_size)``.
        patch_valid : torch.Tensor
            Boolean ``(batch, n_pairs, n_patches)``, ``True`` for the patches
            that hold enough measured samples.

        Returns
        -------
        torch.Tensor
            :math:`H^s`, ``(batch, n_pairs, n_patches, embed_dim)``, zero at
            the invalid patches.
        """
        # Cut each channel into non-overlapping patches
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, S)
        x = self.split_patches(x)
        n_patches = x.shape[2]

        # A normalised token per patch
        # (batch, n_pairs, P, S) -> (batch, n_pairs, P, D)
        x = self.patch_projection(x)
        x = self.patch_norm(x)

        # Add the embedding of the patch index, (P, D) broadcast over batch
        # and pairs
        time_position = self.time_embedding.weight[:n_patches]
        x = x + time_position

        # Zero the tokens of the invalid patches
        # (batch, n_pairs, P) -> (batch, n_pairs, P, 1)
        weight = patch_valid[..., None].to(x.dtype)
        x = x * weight
        token_shape = x.shape

        # Key padding mask of the invalid patches. A channel without a valid
        # patch attends to all of them, since an all-masked row gives NaN; its
        # tokens are zeroed below.
        # (batch, n_pairs, P) -> (batch * n_pairs, P)
        rows_valid = self.merge_pair_mask(patch_valid)
        row_active = rows_valid.any(dim=1, keepdim=True)
        attend = rows_valid | ~row_active
        padding = ~attend

        # Transformer over the patches of each channel
        # (batch, n_pairs, P, D) -> (batch * n_pairs, P, D) -> (batch, n_pairs, P, D)
        x = self.merge_pairs(x)
        x = self.temporal_transformer(x, src_key_padding_mask=padding)
        x = x.reshape(token_shape)

        # Output LayerNorm, the invalid patches zero before and after it
        # (batch, n_pairs, P, D)
        x = x * weight
        x = self.output_norm(x)
        x = x * weight
        return x


class _GeometryEncoder(nn.Module):
    """Geometry encoder: a token per HbO/HbR pair from its geometry vector
    :math:`g_c` (midpoint and source-to-detector offset).

    For ``(batch, n_pairs, geometry_dim)`` geometries:

    1. zero the geometry of the pairs without data;
    2. an MLP (linear, activation, linear, LayerNorm), ``(batch, n_pairs,
       embed_dim)``;
    3. a pre-norm Transformer layer over the pairs of the montage, the
       invalid pairs masked out of the attention and zero, ``(batch, n_pairs,
       embed_dim)``.

    Parameters
    ----------
    embed_dim : int
        Width of the geometry tokens, :math:`D`.
    hidden_dim : int
        Hidden width of the MLP.
    num_heads : int
        Heads of the Transformer.
    mlp_ratio : float
        Feed-forward width of the Transformer, over ``embed_dim``.
    drop_prob : float
        Dropout of the Transformer.
    depth : int
        Layers of the Transformer.
    activation : type[nn.Module], default=nn.GELU
        Activation of the MLP and of the Transformer.
    geometry_dim : int, default=6
        Size of :math:`g_c`: three midpoint and three offset coordinates.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_prob: float,
        depth: int,
        activation: type[nn.Module] = nn.GELU,
        geometry_dim: int = 6,
    ):
        super().__init__()

        # MLP: a D-wide token per pair from its g_c
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Pre-norm Transformer, with a final LayerNorm, over the pairs: each
        # pair sees where the others are
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=round(embed_dim * mlp_ratio),
                dropout=drop_prob,
                activation=activation(),
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )

    def forward(self, geometry: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Embed the geometry of each pair.

        Parameters
        ----------
        geometry : torch.Tensor
            :math:`g_c` of each pair, ``(batch, n_pairs, 6)``.
        valid : torch.Tensor
            Boolean ``(batch, n_pairs)``, ``True`` for the pairs that hold
            data.

        Returns
        -------
        torch.Tensor
            :math:`G`, ``(batch, n_pairs, embed_dim)``, zero for the invalid
            pairs.
        """
        # Zero the geometry of the pairs without data
        # (batch, n_pairs) -> (batch, n_pairs, 1); (batch, n_pairs, 6)
        valid_over_features = valid[..., None]
        zeros = torch.zeros_like(geometry)
        x = torch.where(valid_over_features, geometry, zeros)

        # A token per pair, zero for the invalid pairs
        # (batch, n_pairs, 6) -> (batch, n_pairs, D)
        x = self.geometry_projection(x)
        weight = valid_over_features.to(x.dtype)
        x = x * weight

        # Key padding mask of the invalid pairs. A montage without a valid
        # pair attends to all of them, since an all-masked row gives NaN.
        # (batch, n_pairs)
        any_valid = valid.any(dim=1, keepdim=True)
        attend = valid | ~any_valid
        padding = ~attend

        # Transformer over the pairs, the invalid pairs zeroed
        # (batch, n_pairs, D)
        x = self.spatial_transformer(x, src_key_padding_mask=padding)
        x = x * weight
        return x


class _ChannelCompressorAttention(nn.Module):
    """Geometry-aware channel compressor: ``n_latent_channels`` learnable
    queries attend, at each patch, over the HbO/HbR pairs.

    For :math:`H^O` and :math:`H^R`, each ``(batch, n_pairs, n_patches,
    embed_dim)``, and the geometry tokens :math:`G`, ``(batch, n_pairs,
    embed_dim)``:

    1. the patches join the batch axis, one attention per patch: ``(batch *
       n_patches, n_pairs, embed_dim)``;
    2. queries from the normalised latent queries; keys from the normalised
       mean of the two chromophores plus :math:`G`; values from each
       normalised chromophore. Each is split into heads, ``(batch *
       n_patches, heads, tokens, head_dim)``;
    3. one softmax over the valid pairs weighs the values of both
       chromophores, ``(batch * n_patches, heads, n_latent_channels, 2 *
       head_dim)``;
    4. per chromophore, an output projection with a residual on the queries,
       a LayerNorm and a residual pre-norm feed-forward block, ``(batch,
       n_latent_channels, n_patches, embed_dim)``, zero at a patch without a
       valid pair.

    Private because one softmax weighs the values of both chromophores (with
    one dropout mask), and the released weights name its projections
    ``q_proj``/``k_proj``/``v_proj``/``out_proj``.

    Parameters
    ----------
    embed_dim : int
        Token width, :math:`D`.
    n_latent_channels : int
        Learnable latent queries, :math:`M`.
    num_heads : int
        Attention heads, each of width :math:`d_h` = ``embed_dim / num_heads``.
    mlp_ratio : float
        Width of the feed-forward block, over ``embed_dim``.
    drop_prob : float
        Dropout on the attention weights and in the feed-forward block.
    init_std : float
        Standard deviation of the initial latent queries.
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward block.
    """

    def __init__(
        self,
        embed_dim: int,
        n_latent_channels: int,
        num_heads: int,
        mlp_ratio: float,
        drop_prob: float,
        init_std: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.init_std = init_std

        # Latent queries L: the M channels of the compressed montage, shared
        # by every recording whatever its probe layout
        self.latent_queries = nn.Parameter(torch.empty(n_latent_channels, embed_dim))

        # LayerNorms of the queries, the keys and the values
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
        # Projections W_Q, W_K and W_V, the last shared by HbO and HbR
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output projection, LayerNorm and attention dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(drop_prob)
        self.out_norm = nn.LayerNorm(embed_dim)
        # Pre-norm feed-forward block: its LayerNorm, then embed_dim ->
        # round(embed_dim * mlp_ratio) -> embed_dim
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(
            emb_size=embed_dim,
            expansion=1,
            drop_p=drop_prob,
            activation=activation,
            hidden_features=round(embed_dim * mlp_ratio),
        )

        # One attention per patch: the patches join the batch axis, for the
        # tokens, their validity and the geometry
        self.patches_first = Rearrange(
            "batch pairs patches dim -> (batch patches) pairs dim"
        )
        self.patches_first_mask = Rearrange(
            "batch pairs patches -> (batch patches) pairs"
        )
        self.merge_patches = Rearrange(
            "batch patches pairs dim -> (batch patches) pairs dim"
        )
        # Multi-head layout of the queries, keys and values
        self.split_heads = Rearrange(
            "batch tokens (heads dim) -> batch heads tokens dim", heads=num_heads
        )
        # Merge the heads and separate the two chromophores, attended
        # together as [V^O | V^R]
        self.merge_heads = Rearrange(
            "batch heads latents (chrom dim) -> chrom batch latents (heads dim)",
            chrom=2,
        )
        # Output layout: latent channels before patches
        self.latents_first = Rearrange(
            "batch patches latents dim -> batch latents patches dim"
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.latent_queries, 0.0, self.init_std)

    def forward(
        self,
        hbo: torch.Tensor,
        hbr: torch.Tensor,
        geometry: Optional[torch.Tensor],
        patch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress the pairs of each patch into the latent channels.

        Parameters
        ----------
        hbo, hbr : torch.Tensor
            :math:`H^O` and :math:`H^R`, each ``(batch, n_pairs, n_patches,
            embed_dim)``.
        geometry : torch.Tensor or None
            :math:`G`, ``(batch, n_pairs, embed_dim)``; ``None`` leaves it out
            of the keys.
        patch_valid : torch.Tensor
            Boolean ``(batch, n_pairs, n_patches)``.

        Returns
        -------
        tuple of torch.Tensor
            The compressed HbO and HbR tokens, each ``(batch,
            n_latent_channels, n_patches, embed_dim)``, zero at the patches
            without a valid pair.
        """
        batch_size, _, n_patches, _ = hbo.shape

        # One attention per patch: the patches join the batch axis
        # (batch, n_pairs, P, D) -> (batch * P, n_pairs, D)
        hbo = self.patches_first(hbo)
        hbr = self.patches_first(hbr)
        # (batch, n_pairs, P) -> (batch * P, n_pairs)
        valid = self.patches_first_mask(patch_valid)

        # Normalised latent queries LN(L), one copy per patch
        # (M, D) -> (1, M, D) -> (batch * P, M, D)
        latents = self.q_norm(self.latent_queries)
        latents = latents[None].expand(valid.shape[0], -1, -1)

        # Inputs of the keys: the mean of the two chromophores, plus the
        # geometry of each pair repeated over the patches
        # (batch * P, n_pairs, D)
        fused = hbo + hbr
        fused = 0.5 * fused
        if geometry is not None:
            # (batch, n_pairs, D) -> (batch, P, n_pairs, D) -> (batch * P, n_pairs, D)
            patch_geometry = geometry[:, None].expand(-1, n_patches, -1, -1)
            patch_geometry = self.merge_patches(patch_geometry)
            fused = fused + patch_geometry

        # Queries Q = W_Q LN(L)
        # (batch * P, M, D) -> (batch * P, heads, M, d_h)
        query = self.q_proj(latents)
        query = self.split_heads(query)

        # Keys K_p = W_K LN(fused)
        # (batch * P, n_pairs, D) -> (batch * P, heads, n_pairs, d_h)
        key = self.k_norm(fused)
        key = self.k_proj(key)
        key = self.split_heads(key)

        # Values W_V LN(H^s) of each chromophore, side by side so that one
        # attention map (and one dropout mask) weighs both
        # 2 x (batch * P, n_pairs, D) -> (batch * P, heads, n_pairs, 2 * d_h)
        value_hbo = self.v_norm(hbo)
        value_hbo = self.v_proj(value_hbo)
        value_hbo = self.split_heads(value_hbo)
        value_hbr = self.v_norm(hbr)
        value_hbr = self.v_proj(value_hbr)
        value_hbr = self.split_heads(value_hbr)
        value = torch.cat([value_hbo, value_hbr], dim=-1)

        # Mask the invalid pairs out of the keys. A patch without a valid pair
        # attends to all of them, since an all-masked row gives NaN; its
        # tokens are zeroed in the output block.
        # (batch * P, n_pairs) -> (batch * P, 1, 1, n_pairs)
        active = valid.any(dim=1, keepdim=True)
        attend = valid | ~active
        attend = attend[:, None, None]

        # Attention of the latent queries over the pairs
        # (batch * P, heads, M, 2 * d_h)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        x = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attend,
            dropout_p=dropout_p,
            scale=self.head_dim**-0.5,
            exact=True,
        )
        # Merge the heads and separate the chromophores
        # (batch * P, heads, M, 2 * d_h) -> (2, batch * P, M, D)
        x = self.merge_heads(x)

        # Output block of each chromophore, zero at the patches without a
        # valid pair
        # (batch * P, 1) -> (batch * P, 1, 1); (batch * P, M, D)
        active_weight = active[..., None].to(x.dtype)
        hbo = self._output_block(x[0], latents, active_weight)
        hbr = self._output_block(x[1], latents, active_weight)

        # Patches leave the batch axis, latent channels before patches
        # (batch * P, M, D) -> (batch, P, M, D) -> (batch, M, P, D)
        # A dense copy, as the reference's compressor returns it
        # (GeoNIRS@81f8995, model/ours.py): the next layers round as there
        hbo = hbo.unflatten(0, [batch_size, n_patches])
        hbo = self.latents_first(hbo)
        hbo = hbo.contiguous()
        hbr = hbr.unflatten(0, [batch_size, n_patches])
        hbr = self.latents_first(hbr)
        hbr = hbr.contiguous()
        return hbo, hbr

    def _output_block(
        self, x: torch.Tensor, latents: torch.Tensor, active_weight: torch.Tensor
    ) -> torch.Tensor:
        """Output projection and feed-forward block of one chromophore.

        Called once per chromophore, so each draws its own dropout mask.

        Parameters
        ----------
        x : torch.Tensor
            :math:`Z^s_p`, ``(batch * n_patches, n_latent_channels,
            embed_dim)``.
        latents : torch.Tensor
            The normalised latent queries, same shape.
        active_weight : torch.Tensor
            ``(batch * n_patches, 1, 1)``, 1 for the patches that have a
            valid pair and 0 for the others.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        # Output projection with a residual on the latent queries, LayerNorm
        # (batch * P, M, D)
        x = self.out_proj(x)
        x = latents + x
        x = self.out_norm(x)

        # Pre-norm feed-forward block with its residual connection
        # (batch * P, M, D)
        update = self.ffn_norm(x)
        update = self.ffn(update)
        x = x + update

        # Zero the patches without a valid pair
        x = x * active_weight
        return x


class _ChromophoreProjection(nn.Module):
    """Chromophore projection: residual blocks that change the token width
    along ``dims``, on the last axis only.

    One ``_ResidualProjectionBlock`` per step of ``dims``: ``(...,
    dims[0])`` -> ``(..., dims[-1])``. Latent channels and patches are never
    mixed.

    Parameters
    ----------
    dims : sequence of int
        Token widths, from ``embed_dim`` to ``feature_dim``.
    drop_prob : float
        Dropout inside each residual block.
    init_scale : float
        Initial value of the learnable ``residual_scale`` of each block.
    activation : type[nn.Module], default=nn.GELU
        Activation inside each residual block.
    """

    def __init__(
        self,
        dims: Sequence[int],
        drop_prob: float,
        init_scale: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        # One residual block per change of width; ``blocks`` is the name of
        # the released weights
        self.blocks = nn.ModuleList(
            [
                _ResidualProjectionBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    drop_prob=drop_prob,
                    init_scale=init_scale,
                    activation=activation,
                )
                for input_dim, output_dim in zip(dims[:-1], dims[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the latent tokens of one chromophore.

        Parameters
        ----------
        x : torch.Tensor
            Compressed tokens, ``(batch, n_latent_channels, n_patches,
            embed_dim)``.

        Returns
        -------
        torch.Tensor
            :math:`Z^s`, ``(batch, n_latent_channels, n_patches,
            feature_dim)``.
        """
        # Change the width block by block, 192 -> 128 -> 96 -> 64 for
        # GeoNIRS-Base
        # (batch, M, P, D) -> (batch, M, P, F)
        for block in self.blocks:
            x = block(x)
        return x


class _ResidualProjectionBlock(nn.Module):
    """One step of the chromophore projection: a linear skip plus a scaled
    two-layer update, ``(..., input_dim)`` -> ``(..., output_dim)``.

    1. skip: a linear map to ``output_dim``;
    2. update: LayerNorm, linear, activation, dropout and linear;
    3. output: the skip plus ``residual_scale`` times the update.

    Private because no shared block has a learned residual scale on a
    width-changing skip projection.

    Parameters
    ----------
    input_dim : int
        Input width.
    output_dim : int
        Output width.
    drop_prob : float
        Dropout between the two layers of the update.
    init_scale : float
        Initial value of the learnable ``residual_scale``: the block starts
        close to its linear skip.
    activation : type[nn.Module], default=nn.GELU
        Activation between the two layers of the update.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        drop_prob: float,
        init_scale: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.init_scale = init_scale

        # Skip path: a linear map, since the width changes
        self.skip = nn.Linear(input_dim, output_dim)
        # Update path: LayerNorm and a two-layer perceptron
        self.main = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(output_dim, output_dim),
        )
        # Learnable weight of the update path
        self.residual_scale = nn.Parameter(torch.empty(()))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.residual_scale, self.init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., input_dim)`` -> ``(..., output_dim)``."""
        # Linear skip
        # (..., input_dim) -> (..., output_dim)
        skip = self.skip(x)
        # Scaled update
        # (..., input_dim) -> (..., output_dim)
        update = self.main(x)
        update = self.residual_scale * update
        # (..., output_dim)
        x = skip + update
        return x


class _ValidMeanStd(nn.Module):
    """Read-out ``pool="mean_std"``: mean and population standard deviation
    of the tokens over the valid patches of each latent channel.

    For :math:`Z^O` and :math:`Z^R`, ``(batch, 2, n_latent_channels,
    n_patches, feature_dim)``, and the validity ``(batch, n_latent_channels,
    n_patches)``:

    1. concatenate the chromophores along the features, ``(batch,
       n_latent_channels, n_patches, 2 * feature_dim)``;
    2. mean and standard deviation over the valid patches, each ``(batch,
       n_latent_channels, 2 * feature_dim)``;
    3. flatten ``[mean | std]`` of each latent channel, ``(batch, 4 *
       n_latent_channels * feature_dim)``.

    The feature count does not depend on the window length.
    """

    def __init__(self):
        super().__init__()
        # The two chromophores concatenated along the features
        self.join_chromophores = Rearrange(
            "batch chrom latents patches dim -> batch latents patches (chrom dim)"
        )
        # One feature vector per window: [mean | std] of each latent channel
        self.flatten_statistics = Rearrange(
            "stat batch latents dim -> batch (latents stat dim)"
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Pool the tokens over the valid patches.

        Parameters
        ----------
        x : torch.Tensor
            :math:`Z^O` and :math:`Z^R`, ``(batch, 2, n_latent_channels,
            n_patches, feature_dim)``.
        valid : torch.Tensor
            Boolean ``(batch, n_latent_channels, n_patches)``.

        Returns
        -------
        torch.Tensor
            ``(batch, 4 * n_latent_channels * feature_dim)``.
        """
        # Concatenate the chromophores along the features
        # (batch, 2, M, P, F) -> (batch, M, P, 2 * F)
        x = self.join_chromophores(x)

        # Number of valid patches of each latent channel, at least 1 so that
        # a window without a valid patch gives zeros and not NaN
        # (batch, M, P) -> (batch, M, P, 1) -> (batch, M, 1)
        weight = valid[..., None].to(x.dtype)
        count = weight.sum(dim=2)
        count = count.clamp_min(1.0)

        # Mean over the valid patches
        # (batch, M, P, 2 * F) -> (batch, M, 2 * F)
        mean = x * weight
        mean = mean.sum(dim=2)
        mean = mean / count

        # Population standard deviation over the valid patches, as a norm:
        # sqrt of the variance has an infinite gradient at zero (a single
        # valid patch), the norm a zero one
        # (batch, M, P, 2 * F) -> (batch, M, 2 * F)
        deviation = x - mean[:, :, None]
        deviation = deviation * weight
        std = torch.linalg.vector_norm(deviation, dim=2)
        std = std / count.sqrt()

        # [mean | std] of each latent channel, flattened
        # 2 x (batch, M, 2 * F) -> (2, batch, M, 2 * F) -> (batch, 4 * M * F)
        x = torch.stack([mean, std])
        x = self.flatten_statistics(x)
        return x


def _optode_geometry(chs_info, channel_order: str, scale: float) -> np.ndarray:
    """Geometry vector :math:`g_c` of each HbO/HbR pair of ``chs_info``, as
    the authors build it for a downstream dataset.

    Parameters
    ----------
    chs_info : list of dict or None
        MNE channel information; ``loc[3:9]`` holds the source and the
        detector of an fNIRS channel, in metres.
    channel_order : {"stacked", "interleaved"}
        Where the HbO and HbR rows are.
    scale : float
        Reference scale of the coordinates, in metres.

    Returns
    -------
    np.ndarray
        ``(n_pairs, 6)`` float32, ``[midpoint, detector - source]`` of each
        pair, centred and divided by ``scale``; a pair without positions is
        zero. ``(0, 6)`` when no pair has positions.
    """
    empty = np.zeros((0, 6), dtype=np.float32)
    if chs_info is None:
        return empty

    # The two chromophores of a pair share their optodes: read the HbO rows
    channel_index = torch.arange(len(chs_info))[None]
    hbo_index, _ = fnirs_split_hbo_hbr(channel_index, channel_order)

    # Source s_c and detector d_c of each pair, from loc[3:9]
    # (n_pairs, 6); a missing or short loc stays zero
    optodes = np.zeros((hbo_index.shape[1], 6))
    for row, i in zip(optodes, hbo_index[0].tolist()):
        loc = chs_info[i].get("loc")
        loc = np.ravel(loc if loc is not None else [])[3:9]
        row[: loc.size] = loc
    # A pair has positions when they are finite and not all zero
    present = np.isfinite(optodes).all(axis=1) & (optodes != 0).any(axis=1)
    if not present.any():
        return empty

    # Centre on the mean of the optodes that have a position, which removes
    # the translation of the montage, and divide by the reference scale
    present_optodes = np.concatenate([optodes[present, :3], optodes[present, 3:]])
    center = present_optodes.mean(axis=0)
    source = (optodes[:, :3] - center) / scale
    detector = (optodes[:, 3:] - center) / scale

    # g_c = [mid_c, Delta_c], mid_c = (s_c + d_c) / 2, Delta_c = d_c - s_c
    # (n_pairs, 6); a pair without positions is zero
    geometry = np.concatenate([0.5 * (source + detector), detector - source], axis=1)
    geometry[~present] = 0.0
    return geometry.astype(np.float32)

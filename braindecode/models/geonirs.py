# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Adapted from https://github.com/GeoNIRS/GeoNIRS (81f8995; paper arXiv:2610.17845).

from __future__ import annotations

import math
from typing import Literal, Optional

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
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr


class GeoNIRS(EEGModuleMixin, nn.Module):
    r"""GeoNIRS from Li et al. (2026).

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    GeoNIRS is an fNIRS foundation model pretrained on unlabelled recordings
    with different probe layouts :footcite:`li2026geonirs`. Two things make
    such pretraining hard (§1): "channel semantics vary across probe layouts:
    the same channel index may correspond to different cortical locations",
    and "Slowly varying hemodynamic signals make within-signal masked
    reconstruction susceptible to temporal interpolation."

    GeoNIRS answers the first with geometry: it "uses the physical locations
    of source-detector pairs as a shared spatial reference across
    heterogeneous probe layouts" and "compresses variable-size channel sets
    into a fixed-size latent representation". It answers the second by
    rebuilding one chromophore from the other ("bidirectional HbO/HbR
    cross-reconstruction with contiguous temporal masking"). This class is
    the pretrained encoder of Figure 1 of the paper (`arXiv:2610.17845
    <https://arxiv.org/abs/2610.17845>`_) with the paper's linear head; the
    pretraining decoders are not part of it.

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

      "Every patch is projected into a :math:`D`-dimensional token and
      augmented with a learnable temporal positional embedding. Separate HbO
      and HbR temporal encoders then model the token sequence within each
      physical channel". Inside: a linear projection, LayerNorm, an embedding
      of the patch index and a pre-norm Transformer over the patches of one
      channel. All channels share the weights, so "the model does not require
      a fixed :math:`C_d`".

    - ``GeoNIRS.geometry_encoder`` **Geometry encoder** (Appendix B.2)

      "Each physical channel :math:`c` is represented by a six-dimensional
      geometry vector" of its source :math:`s_c` and detector :math:`d_c`:

      .. math::

          g_c = [\mathrm{mid}_c, \Delta_c], \qquad
          \mathrm{mid}_c = \frac{s_c + d_c}{2}, \qquad \Delta_c = d_c - s_c.

      The midpoint gives "the spatial location of the channel" and
      :math:`\Delta_c` "the source-detector separation and orientation".
      Coordinates are centred and divided by "a fixed reference scale of
      25.46 mm"; an MLP and one Transformer layer turn them into :math:`G`.

    - ``GeoNIRS.channel_compressor`` **Geometry-aware channel compressor**
      (§2.3)

      "To transform variable-size physical channel sets into fixed-length
      representations, we introduce :math:`M` learnable latent queries",
      :math:`L`. For each patch :math:`p` they attend over the channels, with
      keys made of the two chromophores fused and the geometry:

      .. math::

          \bar{H}_p = \tfrac{1}{2}\left(H^O_{:,p} + H^R_{:,p}\right), \quad
          Q = W_Q\,\mathrm{LN}(L), \quad
          K_p = W_K\,\mathrm{LN}\left(\bar{H}_p + G\right), \quad
          A_p = \mathrm{softmax}\left(\frac{Q K_p^\top}{\sqrt{d_h}}\right),
          \quad Z^s_p = A_p\,W_V\,\mathrm{LN}\left(H^s_{:,p}\right),
          \quad s \in \{O, R\}.

      "HbO and HbR share the same attention weights but retain separate value
      representations", and "Invalid channels are masked before the softmax
      operation." An output projection and a feed-forward block follow.

    - ``GeoNIRS.hbo_projection``, ``GeoNIRS.hbr_projection``
      **Chromophore projections** (§2.4)

      "a chromophore-specific residual projection that maps the feature
      dimension from :math:`D` to :math:`F` without mixing latent channels or
      temporal patches": three residual blocks, of widths [192, 128, 96, 64]
      for GeoNIRS-Base (Table 6).

    - ``GeoNIRS.final_layer`` **Head** (§2.5, Appendix B.5.2)

      The tokens of the two chromophores are "flattened, standardized using
      training-set statistics, followed by dropout (:math:`p = 0.1`) and a
      single linear classification layer". ``pool="flatten"`` is that head
      without the standardization; ``pool="mean_std"`` reads out the mean and
      standard deviation of :math:`[Z^O \,\|\, Z^R]` over the valid patches.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** A patch is :math:`S = 10` samples, 2 s at 5 Hz, and the
      positional embedding has 15 rows, the 30 s pretraining window. The
      compressor keeps the patch axis: :math:`M` tokens per 2 s patch.
    - **Spatial.** A channel has no index embedding. It is known only through
      its geometry :math:`g_c`, in the keys of the compressor, so any number
      of channels in any layout maps to the same :math:`M` latent channels.
      Zeroing or shuffling the geometry at inference lowers the task-averaged
      balanced accuracy by 2.34 and 3.24 points (Table 3).
    - **Chromophores.** The rows are HbO and HbR concentration changes (no
      HbT), each with its own temporal encoder and projection. The encoder
      has no frequency decomposition.
    - **Validity masks.** "Samples and channels rejected during preprocessing
      quality control, together with patches containing insufficient valid
      samples, are excluded using validity masks" (§2.2). Here a non-finite
      sample is missing, a patch needs ``minimum_patch_valid_fraction`` of
      its samples, and ``ch_mask`` marks padded pairs.

    .. rubric:: Pretrained Weights

    The authors release both backbones in the ``checkpoint/`` folder (Git
    LFS) of `GeoNIRS/GeoNIRS@81f8995
    <https://github.com/GeoNIRS/GeoNIRS/tree/81f8995e5c3aa6b70e55812b194fc04e4e7ea5b7/checkpoint>`_,
    ``geonirs_base.pt`` and ``geonirs_large.pt``. Both use :math:`M = 16` and
    :math:`S = 10` (Table 6); the defaults build GeoNIRS-Base.

    ==============================  ====================  =====================
    Component                       GeoNIRS-Base          GeoNIRS-Large
    ==============================  ====================  =====================
    Pretraining corpus              8 datasets, 383.30 h  22 datasets, 827.04 h
    ``embed_dim`` (:math:`D`)       192                   384
    ``depth`` / ``num_heads``       4 / 8                 6 / 12
    ``feature_dim`` (:math:`F`)     64                    128
    Parameters, without the head    4,443,462             24,394,310
    ==============================  ====================  =====================

    :meth:`load_state_dict` takes the file as released: it drops the
    pretraining decoders and keeps the head's initialisation. The weights
    expect 5 Hz signals standardized per recording and at most 150 samples.

    .. code-block:: python

        import torch
        from braindecode.models import GeoNIRS

        payload = torch.load("geonirs_base.pt", map_location="cpu", weights_only=True)
        model = GeoNIRS(chs_info=raw.info["chs"], n_outputs=2, n_times=60, sfreq=5.0)
        model.load_state_dict(payload)

    .. rubric:: Differences from the reference implementation

    - **Input.** One ``(batch, 2C, n_times)`` tensor, stacked or interleaved
      (the MNE layout), instead of a batch of ``hbo``, ``hbr``, validity
      masks and ``geometry``. A NaN marks a sample the reference zero-fills
      and flags.
    - **Geometry.** Read from ``chs_info`` (``loc[3:9]``), centred and scaled
      as the reference's ``scripts/build_fair_input.py``; a pair without
      positions gets a zero geometry. ``forward(geometry=...)`` replaces it.
    - **Head.** The training-set standardization is left to the caller.
      ``pool="mean_std"`` gives the features of the reference's
      ``mean_std_linear`` head.
    - **Added:** ``on_non_divisible="pad"`` / ``"crop"`` and ``ch_mask``.
      **Left out:** the decoders, the temporal masking and LoRA.

    .. versionadded:: 2.1

    Parameters
    ----------
    patch_size : int, default=10
        Samples per patch, :math:`S` in the paper: 2 s at 5 Hz.
    max_patches : int, default=15
        Rows of the temporal positional embedding, the 15 patches of a 30 s
        pretraining window; raised to the patch count of ``n_times``.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        ``n_times`` not a multiple of ``patch_size``: refuse (as the
        reference); pad with missing samples, to which the validity rule
        applies; or crop.
    embed_dim : int, default=192
        Token width, :math:`D` in the paper ("Embedding width", Table 6).
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
    n_latent_channels : int, default=16
        Learnable latent queries, :math:`M` in the paper.
    compressor_heads : int or None, default=None
        Heads of the compressor's attention; ``None`` is ``num_heads``.
    feature_dim : int, default=64
        Token width after the projection, :math:`F` in the paper (Table 6).
    use_channel_geometry : bool, default=True
        Add the geometry :math:`G` to the keys of the compressor. ``False``
        builds no geometry encoder: the "w/o Geometry" variant of Table 4.
    pool : {"flatten", "mean_std"}, default="flatten"
        Read-out of the ``(batch, 2, M, P, F)`` tokens: all of them flattened
        (the head of the paper), or their mean and standard deviation over
        the valid patches.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or ``[HbO_1, HbR_1, ...]``; ``None``
        reads it from ``chs_info`` (stacked without it).
    drop_prob : float, default=0.1
        Dropout in the backbone, 0.1 in both checkpoints.
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer``, :math:`p = 0.1` in Appendix B.5.2.
    minimum_patch_valid_fraction : float, default=0.8
        Share of measured samples a patch needs to be valid: "at least 80% of
        its time points" (Appendix B.1.2).
    activation : type[nn.Module], default=nn.GELU
        Activation of the backbone; GELU in the reference.
    channel_strategy : str, default="native"
        Only ``"native"``: the input is fNIRS HbO/HbR channels, not electrodes
        the channel layer can map. GeoNIRS reads any optode montage from its
        source-detector geometry (``chs_info`` or ``geometry``), and
        ``ch_mask`` pads montages of different sizes. Any other value raises
        a ``ValueError``.
    channel_strategy_kwargs : dict or None, default=None
        Must be ``None``.

    Notes
    -----
    **Protocol.** For pretraining (Appendix B.1), raw intensity is converted
    to HbO and HbR concentration changes with the modified Beer-Lambert law,
    resampled to 5 Hz and standardized per recording, HbO and HbR separately,
    then cut into 30 s windows; B.1.1 lists no band-pass filter. Downstream
    (§3.1, Appendix B.5-B.6), eight held-out binary tasks in windows of 12 s
    or 10 s are scored by subject-level five-fold cross-validation with seeds
    810 and 811. With linear probing, LoRA and full fine-tuning GeoNIRS
    reaches 63.06, 63.42 and 63.54 task-averaged balanced accuracy (Table 1).

    **Replication.** Linear probe of the frozen GeoNIRS-Base on DS42 (action
    mode, ds005935) and DS19 (pain intensity, ds005777), with the paper's
    folds and seeds. "Authors' pipeline" is their repository at 81f8995 run
    end to end; "Port" is this class with a re-implementation of their probe.

    ===============  ========  ======  =================  ======
    Task             Metric    Paper   Authors' pipeline  Port
    ===============  ========  ======  =================  ======
    DS42             macro-F1  0.7018  0.7000             0.7039
    DS42             AUROC     0.7840  0.7840             0.7840
    DS19             macro-F1  0.5598  0.5599             0.5304
    DS19             AUROC     0.5683  0.5687             0.5549
    ===============  ========  ======  =================  ======

    Both pipelines build the same windows (1628 and 1190), on which this
    class gives the authors' encoder features (max-abs difference 0.0, Base
    and Large, eval mode).

    References
    ----------
    .. footbibliography::
    """

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
        n_latent_channels: int = 16,
        compressor_heads: int | None = None,
        feature_dim: int = 64,
        use_channel_geometry: bool = True,
        pool: Literal["flatten", "mean_std"] = "flatten",
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.1,
        head_drop_prob: float = 0.1,
        minimum_patch_valid_fraction: float = 0.8,
        activation: type[nn.Module] = nn.GELU,
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ) -> None:
        if channel_strategy != "native" or channel_strategy_kwargs:
            raise ValueError(
                f"{type(self).__name__} takes fNIRS HbO/HbR channels, not electrodes "
                "the channel layer can map: only channel_strategy='native' is "
                "supported. It reads any optode montage from its source-detector "
                "geometry (chs_info or geometry), and ch_mask pads montages of "
                "different sizes."
            )
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
        # ``None`` takes the value of the temporal encoders, as in both
        # checkpoints
        self.geometry_hidden_dim = geometry_hidden_dim or embed_dim
        self.geometry_heads = geometry_heads or num_heads
        self.compressor_heads = compressor_heads or num_heads
        # Measured samples a patch needs to be valid ("at least 80% of its
        # time points", Appendix B.1.2): 8 of 10 with the defaults
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
        # Where the HbO and HbR rows are (stacked or interleaved); raises when
        # chs_info does not list HbO/HbR pairs
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        _check_on_non_divisible(on_non_divisible)
        # P, the patches of one window; raises for a window the patches do not
        # tile when on_non_divisible="error"
        self.n_patches = patch_count(self.n_times, patch_size, on_non_divisible)

        # Layers
        # Following paper nomenclature

        # Temporal encoders (§2.2): "Separate HbO and HbR temporal encoders
        # then model the token sequence within each physical channel", giving
        # H^O and H^R. The positional embedding needs a row per patch of the
        # window, so it grows past ``max_patches`` for a longer window.
        self.hbo_encoder = _TemporalEncoder(
            patch_size=self.patch_size,
            max_patches=max(max_patches, self.n_patches),
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
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
            activation=activation,
        )

        # Geometry encoder (Appendix B.2): the source-detector geometry g_c of
        # each channel becomes G, "a shared spatial reference across
        # heterogeneous probe layouts". The "w/o Geometry" variant (Table 4)
        # has none.
        self.geometry_encoder = None
        if use_channel_geometry:
            self.geometry_encoder = _GeometryEncoder(
                embed_dim=self.embed_dim,
                hidden_dim=self.geometry_hidden_dim,
                num_heads=self.geometry_heads,
                drop_prob=drop_prob,
                activation=activation,
            )

        # Geometry-aware channel compressor (§2.3): M latent queries map the
        # C channels of a recording to M latent channels
        self.channel_compressor = _ChannelCompressorAttention(
            embed_dim=self.embed_dim,
            n_latent_channels=self.n_latent_channels,
            num_heads=self.compressor_heads,
            drop_prob=drop_prob,
            activation=activation,
        )

        # Chromophore projections Pi_O and Pi_R (§2.4): D -> F, giving Z^O
        # and Z^R
        self.hbo_projection = _ChromophoreProjection(
            embed_dim=self.embed_dim,
            feature_dim=self.feature_dim,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.hbr_projection = _ChromophoreProjection(
            embed_dim=self.embed_dim,
            feature_dim=self.feature_dim,
            drop_prob=drop_prob,
            activation=activation,
        )

        # Geometry g_c of the pairs of chs_info (Eqs. 4-5), used when forward
        # gets no ``geometry``. Derived from chs_info, so not saved with the
        # weights.
        self.register_buffer(
            "default_geometry",
            torch.from_numpy(_optode_geometry(self._chs_info, self.channel_order)),
            persistent=False,
        )

        # Validity masks: the time axis of the sample mask is cut in patches,
        # and the masks get the axes they are broadcast over
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=self.patch_size,
        )
        self.pair_over_time = Rearrange("batch pairs -> batch pairs 1")
        self.patch_over_latents = Rearrange("batch patches -> batch 1 patches")
        self.patch_over_features = Rearrange("batch patches -> batch 1 patches 1")

        # Read-out of the (batch, 2, M, P, F) tokens. "flatten": every token,
        # the head of the paper (Appendix B.5.2). "mean_std": mean and
        # standard deviation over the valid patches.
        self.flatten_foundation = Rearrange(
            "batch chrom latents patches dim -> batch (chrom latents patches dim)"
        )
        self.mean_std = _ValidMeanStd()
        if self.pool == "flatten":
            n_features = 2 * n_latent_channels * self.n_patches * feature_dim
        else:
            n_features = 4 * n_latent_channels * feature_dim

        # Head (Appendix B.5.2): "dropout (p = 0.1) and a single linear
        # classification layer"
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(n_features, self.n_outputs)

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
        # Backbone of the paper (§2.2-2.4): temporal encoders, geometry
        # encoder, channel compressor and chromophore projections
        # (batch, n_chans, n_times) -> (batch, 2, M, P, F): Z^O and Z^R
        encoded = self.encode(x, geometry=geometry, ch_mask=ch_mask)
        x = encoded["foundation"]

        if self.pool == "flatten":
            # Read-out of the paper: the representation "is flattened"
            # (Appendix B.5.2)
            # (batch, 2, M, P, F) -> (batch, 2 * M * P * F)
            x = self.flatten_foundation(x)
        else:
            # Mean and standard deviation of [Z^O | Z^R] over the valid
            # patches of each latent channel
            # (batch, 2, M, P, F) -> (batch, 4 * M * F)
            x = self.mean_std(x, encoded["valid"])

        # Head (Appendix B.5.2): "dropout (p = 0.1) and a single linear
        # classification layer"
        x = self.head_dropout(x)
        if return_features:
            return {"features": x, "cls_token": None}

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

        The backbone of the paper: temporal encoders, geometry encoder,
        channel compressor and chromophore projections.

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
            ``(batch, n_chans)`` mask of the real channels of a padded batch;
            it must pad whole HbO/HbR pairs.

        Returns
        -------
        dict of torch.Tensor
            ``"foundation"``: :math:`Z^O` and :math:`Z^R`, ``(batch, 2,
            n_latent_channels, n_patches, feature_dim)``. ``"valid"``:
            boolean ``(batch, n_latent_channels, n_patches)``, ``False`` for
            the patches that have no valid channel (their tokens are zero).
        """
        # Padded channels of the batch, as pairs: both chromophores of a
        # padded pair must be flagged
        # (batch, n_chans) -> (batch, n_pairs)
        pair_mask: Optional[torch.Tensor] = None
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None:
            hbo_mask, hbr_mask = fnirs_split_hbo_hbr(ch_mask, self.channel_order)
            if not torch.equal(hbo_mask, hbr_mask):
                raise ValueError("ch_mask needs whole HbO/HbR pairs.")
            pair_mask = hbo_mask

        # The window is cut into P patches of S samples (§2.2)
        # (batch, n_chans, n_times) -> (batch, n_chans, P * S)
        n_times = x.shape[-1]
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)

        # Paired chromophore signals X^O and X^R of the C source-detector pairs
        # (batch, n_chans, P * S) -> 2 x (batch, n_pairs, P * S)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape

        # Validity masks (§2.2). A sample is measured when both chromophores
        # are finite; a pair is valid when it has a measured sample and is
        # not padding.
        hbo_finite = torch.isfinite(hbo)
        hbr_finite = torch.isfinite(hbr)
        # (batch, n_pairs, P * S)
        finite = hbo_finite & hbr_finite
        # The samples added by on_non_divisible="pad" are not measured
        finite[:, :, n_times:] = False
        # (batch, n_pairs)
        channel_valid = finite.any(dim=-1)
        if pair_mask is not None:
            channel_valid = pair_mask & channel_valid
        # (batch, n_pairs) -> (batch, n_pairs, 1)
        channel_valid_over_time = self.pair_over_time(channel_valid)
        # (batch, n_pairs, P * S)
        sample_valid = finite & channel_valid_over_time

        # Missing samples are set to zero, as the reference does
        zeros = torch.zeros_like(hbo)
        hbo = torch.where(sample_valid, hbo, zeros)
        hbr = torch.where(sample_valid, hbr, zeros)

        # A patch is valid when enough of its samples are measured ("at least
        # 80% of its time points", Appendix B.1.2)
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, S) -> (batch, n_pairs, P)
        patch_valid = self.split_patches(sample_valid)
        patch_valid = patch_valid.sum(dim=-1)
        patch_valid = patch_valid >= self.min_valid_samples
        patch_valid = patch_valid & channel_valid_over_time

        # Geometry encoder (Appendix B.2): g_c of each pair -> G
        geometry_embedding: Optional[torch.Tensor] = None
        if self.geometry_encoder is not None:
            if geometry is None:
                # the geometry of chs_info, shared by the whole batch
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
                # one g_c (6 numbers, as default_geometry) per pair of x, for
                # the whole batch or for each window
                pair_geometry = geometry
                pair_shape = [n_pairs, self.default_geometry.shape[-1]]
                if list(pair_geometry.shape[-2:]) != pair_shape:
                    raise ValueError(
                        "geometry and x have different channel counts, or "
                        "geometry is not 6-D: pass (n_pairs, 6) or (batch, "
                        "n_pairs, 6), one row per HbO/HbR pair of x."
                    )
            pair_geometry = pair_geometry.to(device=x.device, dtype=hbo.dtype)
            # (n_pairs, 6) or (batch, n_pairs, 6) -> (batch, n_pairs, 6)
            pair_geometry = pair_geometry.expand(batch_size, -1, -1)
            # (batch, n_pairs, 6) -> (batch, n_pairs, D): G
            geometry_embedding = self.geometry_encoder(pair_geometry, channel_valid)

        # Temporal encoders (§2.2), one per chromophore: the patches of each
        # channel become the tokens H^O and H^R
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, D)
        hbo = self.hbo_encoder(hbo, patch_valid)
        hbr = self.hbr_encoder(hbr, patch_valid)

        # Geometry-aware channel compressor (§2.3): the C channels of each
        # patch are compressed into M latent channels, with one attention map
        # for both chromophores
        # (batch, n_pairs, P, D) -> (batch, M, P, D)
        hbo, hbr = self.channel_compressor(hbo, hbr, geometry_embedding, patch_valid)

        # A patch is kept when at least one pair is valid in it
        # (batch, n_pairs, P) -> (batch, P)
        time_valid = patch_valid.any(dim=1)
        # (batch, P) -> (batch, 1, P, 1)
        valid_scale = self.patch_over_features(time_valid)
        valid_scale = valid_scale.to(hbo.dtype)

        # Chromophore projections Pi_O and Pi_R (§2.4): Z^O and Z^R, zero at
        # the patches that are not kept
        # (batch, M, P, D) -> (batch, M, P, F)
        hbo = self.hbo_projection(hbo)
        hbo = hbo * valid_scale
        hbr = self.hbr_projection(hbr)
        hbr = hbr * valid_scale

        # The two chromophores side by side: the "base representation" of
        # Table 6
        # 2 x (batch, M, P, F) -> (batch, 2, M, P, F)
        foundation = torch.stack([hbo, hbr], dim=1)

        # Validity of the tokens, the same for every latent channel
        # (batch, P) -> (batch, 1, P) -> (batch, M, P)
        valid = self.patch_over_latents(time_valid)
        valid = valid.expand(-1, self.n_latent_channels, -1)
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a state dict, or a checkpoint as the authors release it.

        Parameters
        ----------
        state_dict : dict
            A state dict of this class, or the released file's payload
            (``{"model": ..., "config": ...}``).
        strict : bool, default=True
            As in :meth:`torch.nn.Module.load_state_dict`.
        assign : bool, default=False
            As in :meth:`torch.nn.Module.load_state_dict`.

        Returns
        -------
        NamedTuple
            The missing and unexpected keys.
        """
        # The released files keep the weights under "model"
        state_dict = state_dict.get("model", state_dict)
        # The reconstruction decoders are pretraining-only (§2.5: "the
        # reconstruction decoders are discarded")
        decoders = ("hbo_to_hbr_decoder.", "hbr_to_hbo_decoder.")
        cleaned = {k: v for k, v in state_dict.items() if not k.startswith(decoders)}
        # A backbone checkpoint has no head: the head keeps its initialisation
        if not any(k.startswith("final_layer.") for k in cleaned):
            head = self.final_layer.state_dict(prefix="final_layer.")
            cleaned.update(head)
        return super().load_state_dict(cleaned, strict=strict, assign=assign)


class _TemporalEncoder(nn.Module):
    """Temporal encoder of one chromophore (paper §2.2).

    "Every patch is projected into a D-dimensional token and augmented with a
    learnable temporal positional embedding", then a Transformer models "the
    token sequence within each physical channel". The authors'
    ``TemporalStreamEncoder``; HbO and HbR each have their own.

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
    activation : type[nn.Module], default=nn.GELU
        Activation of the Transformer.
    init_std : float, default=0.02
        Standard deviation of the initial positional embedding (the
        reference's ``time_embedding_init_std``).
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
        activation: type[nn.Module] = nn.GELU,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.init_std = init_std

        # Non-overlapping patches: the time axis becomes (patches, samples)
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_size,
        )
        # Patch projection: the S samples of a patch become one D-wide token
        self.patch_projection = nn.Linear(patch_size, embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        # Learnable temporal positional embedding: one row per patch index
        self.time_embedding = nn.Embedding(max_patches, embed_dim)
        # Temporal Transformer (pre-norm, final LayerNorm). It attends over
        # the patches of ONE channel, so its weights serve any channel count.
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

        # The validity mask gets a feature axis to multiply the tokens
        self.patch_over_features = Rearrange(
            "batch pairs patches -> batch pairs patches 1"
        )
        # Every channel becomes its own sequence of patches for the Transformer
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
            :math:`H^s` of the paper, ``(batch, n_pairs, n_patches,
            embed_dim)``, zero at the invalid patches.
        """
        # Cut each channel into P non-overlapping patches of S samples
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, S)
        x = self.split_patches(x)
        n_patches = x.shape[2]

        # Patch projection and LayerNorm: one D-dimensional token per patch
        # (batch, n_pairs, P, S) -> (batch, n_pairs, P, D)
        x = self.patch_projection(x)
        x = self.patch_norm(x)

        # Learnable temporal positional embedding of the patch index 0..P-1
        # (P, D), broadcast over batch and pairs
        time_position = self.time_embedding.weight[:n_patches]
        x = x + time_position

        # Validity mask: an invalid patch carries no token
        # (batch, n_pairs, P) -> (batch, n_pairs, P, 1)
        weight = self.patch_over_features(patch_valid)
        weight = weight.to(x.dtype)
        x = x * weight
        token_shape = x.shape

        # Which patches each channel may attend to. The reference skips a
        # channel with no valid patch; here such a channel attends to all its
        # patches (an all-padded row would give NaN) and is zeroed below.
        # (batch, n_pairs, P) -> (batch * n_pairs, P)
        rows_valid = self.merge_pair_mask(patch_valid)
        row_active = rows_valid.any(dim=1, keepdim=True)
        attend = rows_valid | ~row_active
        padding = ~attend

        # Temporal Transformer over the patch sequence of each channel
        # (batch, n_pairs, P, D) -> (batch * n_pairs, P, D)
        x = self.merge_pairs(x)
        x = self.temporal_transformer(x, src_key_padding_mask=padding)
        # (batch * n_pairs, P, D) -> (batch, n_pairs, P, D)
        x = x.reshape(token_shape)

        # Output LayerNorm, with the invalid patches zero before and after it
        # (batch, n_pairs, P, D): H^s
        x = x * weight
        x = self.output_norm(x)
        x = x * weight
        return x


class _GeometryEncoder(nn.Module):
    """Geometry encoder (paper Appendix B.2, Table 6).

    Turns the six-dimensional geometry vector :math:`g_c` of each channel
    (midpoint and source-to-detector offset) into its embedding :math:`G`,
    with an MLP and a Transformer over the channels. The authors'
    ``SourceDetectorGeometryEncoder``.

    Parameters
    ----------
    embed_dim : int
        Width of the geometry embedding, :math:`D`.
    hidden_dim : int
        Hidden width of the MLP.
    num_heads : int
        Heads of the Transformer.
    drop_prob : float
        Dropout of the Transformer.
    activation : type[nn.Module], default=nn.GELU
        Activation of the MLP and of the Transformer.
    geometry_dim : int, default=6
        Size of :math:`g_c`: three midpoint and three offset coordinates
        (Eq. 4).
    depth : int, default=1
        Layers of the Transformer ("1 layer", Table 6).
    mlp_ratio : float, default=2.0
        Feed-forward width of the Transformer, over ``embed_dim`` (the
        reference's ``embedding_dim * 2``).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        drop_prob: float,
        activation: type[nn.Module] = nn.GELU,
        geometry_dim: int = 6,
        depth: int = 1,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()

        # Geometry MLP: g_c (6 numbers) -> one D-wide token per channel
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Spatial Transformer (pre-norm, final LayerNorm): each channel sees
        # where the other channels of the montage are
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

        # The channel mask gets a feature axis to multiply the tokens
        self.pair_over_features = Rearrange("batch pairs -> batch pairs 1")

    def forward(self, geometry: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Embed the geometry of each channel.

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
            :math:`G` of the paper, ``(batch, n_pairs, embed_dim)``, zero for
            the invalid pairs.
        """
        # A pair without data has no geometry: its g_c is set to zero
        # (batch, n_pairs) -> (batch, n_pairs, 1)
        valid_over_features = self.pair_over_features(valid)
        zeros = torch.zeros_like(geometry)
        # (batch, n_pairs, 6)
        x = torch.where(valid_over_features, geometry, zeros)

        # Geometry MLP: g_c -> one token per channel
        # (batch, n_pairs, 6) -> (batch, n_pairs, D)
        x = self.geometry_projection(x)
        weight = valid_over_features.to(x.dtype)
        x = x * weight

        # Which channels may be attended to. As in the temporal encoder, a
        # recording with no valid pair attends to all and is zeroed below.
        any_valid = valid.any(dim=1, keepdim=True)
        attend = valid | ~any_valid
        padding = ~attend

        # Spatial Transformer over the channels of the montage
        # (batch, n_pairs, D): G
        x = self.spatial_transformer(x, src_key_padding_mask=padding)
        x = x * weight
        return x


class _ChannelCompressorAttention(nn.Module):
    """Geometry-aware channel compressor (paper §2.3).

    "To transform variable-size physical channel sets into fixed-length
    representations", ``n_latent_channels`` learnable latent queries attend
    over the channels of each patch. The keys are the fused HbO/HbR tokens
    plus the geometry; "HbO and HbR share the same attention weights but
    retain separate value representations". The authors'
    ``BrainOmniChannelCompressor``.

    Private because one softmax over the fused HbO/HbR keys weighs both
    chromophores' values (one dropout mask), and the released checkpoints
    name its projections ``q_proj``/``k_proj``/``v_proj``/``out_proj``.

    Parameters
    ----------
    embed_dim : int
        Token width, :math:`D`.
    n_latent_channels : int
        Learnable latent queries, :math:`M`.
    num_heads : int
        Attention heads, each of width :math:`d_h` = ``embed_dim / num_heads``.
    drop_prob : float
        Dropout on the attention weights and in the feed-forward block.
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward block.
    mlp_ratio : float, default=2.0
        Width of the feed-forward block, over ``embed_dim`` (the reference's
        ``embedding_dim * 2``).
    init_std : float, default=0.02
        Standard deviation of the initial latent queries (the reference's
        ``randn * 0.02``).
    """

    def __init__(
        self,
        embed_dim: int,
        n_latent_channels: int,
        num_heads: int,
        drop_prob: float,
        activation: type[nn.Module] = nn.GELU,
        mlp_ratio: float = 2.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.init_std = init_std

        # Latent queries L: the M channels of the compressed montage, shared
        # by every recording whatever its probe layout
        self.latent_queries = nn.Parameter(torch.empty(n_latent_channels, embed_dim))

        # LayerNorms of the queries (L), the keys (fused tokens + geometry)
        # and the values (tokens of one chromophore)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
        # W_Q, W_K, W_V of §2.3; W_V is shared by HbO and HbR
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # "an output projection and a feed-forward block" (§2.3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Dropout on the attention weights A_p
        self.attn_dropout = nn.Dropout(drop_prob)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, round(embed_dim * mlp_ratio)),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(round(embed_dim * mlp_ratio), embed_dim),
        )

        # The attention runs over the channels of ONE patch: patches join
        # the batch axis, for the tokens and for their validity
        self.patches_first = Rearrange(
            "batch pairs patches dim -> (batch patches) pairs dim"
        )
        self.patches_first_mask = Rearrange(
            "batch pairs patches -> (batch patches) pairs"
        )
        # The geometry is the same for every patch of a window
        self.geometry_over_patches = Rearrange("batch pairs dim -> batch 1 pairs dim")
        self.merge_patches = Rearrange(
            "batch patches pairs dim -> (batch patches) pairs dim"
        )
        # The latent queries are the same for every window and patch
        self.latents_over_batch = Rearrange("latents dim -> 1 latents dim")
        # Multi-head layout of queries, keys and values
        self.split_heads = Rearrange(
            "batch tokens (heads dim) -> batch heads tokens dim", heads=num_heads
        )
        # The channel validity masks the keys, for every head and query
        self.keys_over_queries = Rearrange("batch pairs -> batch 1 1 pairs")
        # Heads are merged back and the two chromophores, attended together
        # as [V^O | V^R], are separated again
        self.merge_heads = Rearrange(
            "batch heads latents (chrom dim) -> chrom batch latents (heads dim)",
            chrom=2,
        )
        self.active_over_features = Rearrange("batch one -> batch one 1")
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
        """Compress the channels of each patch into the latent channels.

        Parameters
        ----------
        hbo, hbr : torch.Tensor
            :math:`H^O` and :math:`H^R` of the temporal encoders, each
            ``(batch, n_pairs, n_patches, embed_dim)``.
        geometry : torch.Tensor or None
            :math:`G` of the geometry encoder, ``(batch, n_pairs,
            embed_dim)``; ``None`` leaves it out of the keys.
        patch_valid : torch.Tensor
            Boolean ``(batch, n_pairs, n_patches)``.

        Returns
        -------
        tuple of torch.Tensor
            The compressed HbO and HbR tokens, each ``(batch,
            n_latent_channels, n_patches, embed_dim)``, zero at the patches
            with no valid channel.
        """
        batch_size, _, n_patches, _ = hbo.shape

        # One attention problem per patch p: its C channels are the tokens
        # (batch, n_pairs, P, D) -> (batch * P, n_pairs, D): H^O_{:,p}, H^R_{:,p}
        hbo = self.patches_first(hbo)
        hbr = self.patches_first(hbr)
        # (batch, n_pairs, P) -> (batch * P, n_pairs)
        valid = self.patches_first_mask(patch_valid)

        # Normalised latent queries LN(L), one copy per patch
        # (M, D) -> (1, M, D) -> (batch * P, M, D)
        latents = self.q_norm(self.latent_queries)
        latents = self.latents_over_batch(latents)
        latents = latents.expand(valid.shape[0], -1, -1)

        # Fused chromophores: \bar{H}_p = (H^O_{:,p} + H^R_{:,p}) / 2
        # (batch * P, n_pairs, D)
        fused = hbo + hbr
        fused = 0.5 * fused
        if geometry is not None:
            # Geometry-aware keys: \bar{H}_p + G, with G repeated over patches
            # (batch, n_pairs, D) -> (batch, 1, n_pairs, D) -> (batch * P, n_pairs, D)
            patch_geometry = self.geometry_over_patches(geometry)
            patch_geometry = patch_geometry.expand(-1, n_patches, -1, -1)
            patch_geometry = self.merge_patches(patch_geometry)
            fused = fused + patch_geometry

        # Queries: Q = W_Q LN(L)
        # (batch * P, M, D) -> (batch * P, heads, M, d_h)
        query = self.q_proj(latents)
        query = self.split_heads(query)

        # Keys: K_p = W_K LN(\bar{H}_p + G)
        # (batch * P, n_pairs, D) -> (batch * P, heads, n_pairs, d_h)
        key = self.k_norm(fused)
        key = self.k_proj(key)
        key = self.split_heads(key)

        # Values, one set per chromophore: V^s_p = W_V LN(H^s_{:,p})
        # (batch * P, n_pairs, D) -> (batch * P, heads, n_pairs, d_h)
        value_hbo = self.v_norm(hbo)
        value_hbo = self.v_proj(value_hbo)
        value_hbo = self.split_heads(value_hbo)
        value_hbr = self.v_norm(hbr)
        value_hbr = self.v_proj(value_hbr)
        value_hbr = self.split_heads(value_hbr)
        # "HbO and HbR share the same attention weights": attending once to
        # [V^O | V^R] applies one A_p (and one dropout mask) to both
        # (batch * P, heads, n_pairs, 2 * d_h)
        value = torch.cat([value_hbo, value_hbr], dim=-1)

        # "Invalid channels are masked before the softmax". A patch with no
        # valid channel attends to all of them (an all-masked row would give
        # NaN) and is zeroed in the output block; the reference skips it.
        # (batch * P, 1)
        active = valid.any(dim=1, keepdim=True)
        attend = valid | ~active
        # (batch * P, n_pairs) -> (batch * P, 1, 1, n_pairs)
        attend = self.keys_over_queries(attend)

        # Z^s_p = A_p V^s_p, with A_p = softmax(Q K_p^T / sqrt(d_h))
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
        # (batch * P, heads, M, 2 * d_h) -> (2, batch * P, M, D): Z^O_p, Z^R_p
        x = self.merge_heads(x)

        # (batch * P, 1) -> (batch * P, 1, 1)
        active_weight = self.active_over_features(active)
        active_weight = active_weight.to(x.dtype)

        # Output projection and feed-forward block, chromophore by chromophore
        # (batch * P, M, D)
        hbo = self._output_block(x[0], latents, active_weight)
        hbr = self._output_block(x[1], latents, active_weight)

        # Patches leave the batch axis; latent channels come before patches
        # (batch * P, M, D) -> (batch, P, M, D) -> (batch, M, P, D)
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
        """The "output projection and a feed-forward block" of §2.3, for the
        attended values of one chromophore.

        Called once per chromophore: running both in one call would draw one
        dropout mask for the two instead of one each.

        Parameters
        ----------
        x : torch.Tensor
            :math:`Z^s_p`, ``(batch * n_patches, n_latent_channels,
            embed_dim)``.
        latents : torch.Tensor
            The normalised latent queries, same shape.
        active_weight : torch.Tensor
            ``(batch * n_patches, 1, 1)``, 1 for the patches that have a
            valid channel and 0 for the others.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        # Output projection W_O, residual on the latent queries, LayerNorm
        # (batch * P, M, D)
        x = self.out_proj(x)
        x = latents + x
        x = self.out_norm(x)

        # Feed-forward block with its residual connection
        update = self.ffn(x)
        x = x + update

        # A patch with no valid channel has no tokens
        x = x * active_weight
        return x


class _ChromophoreProjection(nn.Module):
    r"""Chromophore-specific residual projection :math:`\Pi_s` (paper §2.4).

    "a chromophore-specific residual projection that maps the feature
    dimension from D to F without mixing latent channels or temporal
    patches". The authors' ``FeatureProjection`` / ``ScaledFeatureProjection``;
    HbO and HbR each have their own.

    Parameters
    ----------
    embed_dim : int
        Input width, :math:`D`.
    feature_dim : int
        Output width, :math:`F`.
    drop_prob : float
        Dropout inside each residual block.
    activation : type[nn.Module], default=nn.GELU
        Activation inside each residual block.
    """

    def __init__(
        self,
        embed_dim: int,
        feature_dim: int,
        drop_prob: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        # "Projection dimensions" of Table 6: [192, 128, 96, 64] for
        # GeoNIRS-Base and [384, 192, 96, 128] for GeoNIRS-Large. The
        # reference writes the first by hand and [D, D / 2, D / 4, F] for
        # every other size; we follow it.
        widths = [embed_dim, embed_dim // 2, embed_dim // 4, feature_dim]
        if embed_dim == 192 and feature_dim == 64:
            widths = [192, 128, 96, 64]

        # One residual block per change of width; ``blocks`` is the name the
        # released checkpoints use
        self.blocks = nn.ModuleList(
            [
                _ResidualProjectionBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    drop_prob=drop_prob,
                    activation=activation,
                )
                for input_dim, output_dim in zip(widths[:-1], widths[1:])
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
            :math:`Z^s` of the paper, ``(batch, n_latent_channels, n_patches,
            feature_dim)``.
        """
        # Each block acts on the last axis only, so latent channels and
        # patches are never mixed. The width shrinks block by block, e.g.
        # 192 -> 128 -> 96 -> 64 for GeoNIRS-Base
        # (batch, M, P, D) -> (batch, M, P, F)
        for block in self.blocks:
            x = block(x)
        return x


class _ResidualProjectionBlock(nn.Module):
    """One stage of the chromophore projection: a linear skip plus a scaled
    two-layer update (the authors' ``ResidualProjectionBlock``).

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
    activation : type[nn.Module], default=nn.GELU
        Activation between the two layers of the update.
    init_scale : float, default=0.1
        Initial value of the learnable ``residual_scale`` (0.1 in the
        reference): the block starts close to its linear skip.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        drop_prob: float,
        activation: type[nn.Module] = nn.GELU,
        init_scale: float = 0.1,
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
        # Linear skip: (..., input_dim) -> (..., output_dim)
        skip = self.skip(x)
        # Scaled update: residual_scale * MLP(LayerNorm(x))
        update = self.main(x)
        update = self.residual_scale * update
        # (..., output_dim)
        x = skip + update
        return x


class _ValidMeanStd(nn.Module):
    r"""Read-out ``pool="mean_std"``: mean and standard deviation of
    :math:`Z^{\mathrm{joint}} = [Z^O \,\|\, Z^R]` (paper §2.5) over the valid
    patches of each latent channel.

    The features of the reference's ``mean_std_linear`` head
    (``valid_temporal_mean_std``): the window length no longer sets the
    feature count.
    """

    def __init__(self):
        super().__init__()
        # Z^joint: the two chromophores concatenated along the feature axis
        self.join_chromophores = Rearrange(
            "batch chrom latents patches dim -> batch latents patches (chrom dim)"
        )
        # The validity gets a feature axis to weigh the tokens
        self.valid_over_features = Rearrange(
            "batch latents patches -> batch latents patches 1"
        )
        # The mean gets its patch axis back to be subtracted from the tokens
        self.mean_over_patches = Rearrange("batch latents dim -> batch latents 1 dim")
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
        # Z^joint = [Z^O | Z^R]
        # (batch, 2, M, P, F) -> (batch, M, P, 2 * F)
        x = self.join_chromophores(x)

        # Number of valid patches of each latent channel (at least 1, so a
        # window with no valid patch gives zeros and not NaN)
        # (batch, M, P) -> (batch, M, P, 1) -> (batch, M, 1)
        weight = self.valid_over_features(valid)
        weight = weight.to(x.dtype)
        count = weight.sum(dim=2)
        count = count.clamp_min(1.0)

        # Mean over the valid patches
        # (batch, M, P, 2 * F) -> (batch, M, 2 * F)
        mean = x * weight
        mean = mean.sum(dim=2)
        mean = mean / count

        # Standard deviation over the valid patches (population formula, as
        # the reference): sqrt(sum((z - mean)^2) / count)
        # (batch, M, 2 * F) -> (batch, M, 1, 2 * F)
        centre = self.mean_over_patches(mean)
        std = x - centre
        std = std * weight
        std = std.square()
        # (batch, M, P, 2 * F) -> (batch, M, 2 * F)
        std = std.sum(dim=2)
        std = std / count
        std = std.clamp_min(0.0)
        std = std.sqrt()

        # [mean | std] of each latent channel, flattened
        # 2 x (batch, M, 2 * F) -> (2, batch, M, 2 * F) -> (batch, 4 * M * F)
        x = torch.stack([mean, std])
        x = self.flatten_statistics(x)
        return x


def _optode_geometry(
    chs_info, channel_order: str, scale: float = 25.460582556839242 / 1000.0
) -> np.ndarray:
    """Geometry vector :math:`g_c` of each HbO/HbR pair of ``chs_info``
    (paper Appendix B.2, Eqs. 4-5), as the reference builds it for a
    downstream dataset (``scripts/build_fair_input.py``).

    Parameters
    ----------
    chs_info : list of dict or None
        MNE channel information; ``loc[3:9]`` holds the source and the
        detector of an fNIRS channel, in metres.
    channel_order : {"stacked", "interleaved"}
        Where the HbO and HbR rows are.
    scale : float
        The "fixed reference scale of 25.46 mm" of the paper, in metres
        (the reference's ``GEOMETRY_SCALE_M``).

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

    # "Geometry coordinates are additionally centered to remove global
    # translation differences": the centre is the mean of the optodes that
    # have a position
    present_optodes = np.concatenate([optodes[present, :3], optodes[present, 3:]])
    center = present_optodes.mean(axis=0)
    # Dimensionless coordinates: centred, then divided by the reference scale
    source = (optodes[:, :3] - center) / scale
    detector = (optodes[:, 3:] - center) / scale

    # g_c = [mid_c, Delta_c], mid_c = (s_c + d_c) / 2, Delta_c = d_c - s_c
    geometry = np.concatenate([0.5 * (source + detector), detector - source], axis=1)
    # A pair without positions has a zero geometry, as in the reference
    geometry[~present] = 0.0
    return geometry.astype(np.float32)

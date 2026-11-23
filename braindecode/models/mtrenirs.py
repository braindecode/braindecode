# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
#
# Adapted from https://github.com/microsoft/MtReNIRS (MIT).

from __future__ import annotations

import math
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.functional import (
    apply_rotary,
    check_ch_mask,
    pad_or_crop_to_patches,
    patch_count,
    rotary_angles,
    rotary_frequencies,
    scaled_dot_product_attention,
)
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    fnirs_chromophores_from_chs_info,
)
from braindecode.modules import FeedForwardBlock


class MtReNIRS(EEGModuleMixin, nn.Module, license="mit"):
    r"""MtReNIRS from Chae et al. (2026).

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/mtrenirs.png
       :align: center
       :alt: MtReNIRS preprocessing, montage harmonization, token embedding and pretraining

       Fig. 2 of Chae et al. (2026), arXiv:2610.36159 (`CC BY 4.0
       <https://creativecommons.org/licenses/by/4.0/>`_). This class is the
       encoder of panels (C)-(D) with a linear head.

    MtReNIRS (Montage-aligned transferable Representations for fNIRS) is a
    montage-aware masked autoencoder pretrained on 12 public fNIRS datasets
    (268 subjects, 567 recording hours) :footcite:`microsoft2026mtrenirs`. A
    token is one 10-s window of one chromophore channel, tagged with the
    canonical 10-05 position :footcite:`oostenveld2001five` of its
    source-detector (S-D) pair and with its chromophore, so any montage is a
    set of tokens that Transformer blocks mix, with the window index as their
    only time code.

    .. rubric:: Architecture Overview

    With ``W = n_times // window_samples`` windows, ``C = n_chans`` HbO/HbR
    rows and ``D = embed_dim``:

    1. **Windowing**: ``(batch, C, n_times)`` is cut into non-overlapping
       10-s windows :math:`x_{c,t}`, ``(batch, W, C, window_samples)``; a
       short last window is cropped (``on_non_divisible``).
    2. **Token embedding**: each window becomes a token :math:`z_{c,t}`, the
       sum of its signal embedding and of the spatio-chromophore embedding of
       its channel, ``(batch, W * C, D)``.
    3. **Transformer encoder**: ``depth`` pre-norm blocks attend over all
       channels and windows jointly, then a final LayerNorm; invalid tokens
       are never attended and are zeroed, ``(batch, W, C, D)``.
    4. **Readout and head**: ``pool`` reduces the tokens to
       ``(batch, feature_dim)`` and ``final_layer`` gives
       ``(batch, n_outputs)``.

    .. rubric:: Macro Components

    - **Token embedding** (``embed``, ``_TokenEmbedding``).

      - **Operations.** ``embed.signal_proj``, ``nn.Linear(100, 128)``,
        projects each window :math:`x_{c,t}`. ``embed.spatial_mlp``,
        ``nn.Linear(48, 128)``, GELU, ``nn.Linear(128, 128)``, embeds the
        Fourier features :footcite:`tancik2020fourier` of :math:`u_c = [q_c^\top,
        b_c]^\top`, the canonical coordinate :math:`q_c` of the S-D pair and
        the chromophore :math:`b_c` (0 HbO, 1 HbR):

        .. math::

           \phi(u_c) = \left[\sin(2^k \pi u_{c,j}),\, \cos(2^k \pi u_{c,j})
           \right]_{j = 1, \dots, 4}^{k = 0, \dots, 5} \in \mathbb{R}^{48},
           \qquad z_{c,t} = W_x x_{c,t} + b_x + \mathrm{MLP}(\phi(u_c)).

      - *Role.* Region-specific activity and the coupled, often opposing
        HbO/HbR dynamics are told apart, with no class token and no
        per-channel table.

    - **Canonical coordinates** (``default_coordinates``, or the
      ``coordinates`` argument of :meth:`forward`).

      - **Operations.** :math:`\hat{q}_c = (q_c - \mu) / \sigma`, with
        :math:`\mu` = ``coordinate_center`` and :math:`\sigma` =
        ``coordinate_scale`` taken from the pretraining corpus.
      - *Role.* The paper aligns each montage to the 10-05 system and assigns
        every channel the coordinate of its slot. That alignment and slot
        assignment (the authors' ``mtrenirs.alignment.align``, Sec. III-B,
        Eq. 1) stay outside this class, which reads :math:`q_c` from
        ``loc[:3]`` of ``chs_info``.

    - **Transformer encoder** (``encoder.blocks``, ``encoder.norm``;
      ``_TransformerBlock``, ``_RotarySelfAttention``).

      - **Operations.** Six pre-norm blocks of a 128-dimensional state, four
        heads, a 512-dimensional MLP and 0.1 dropout. With :math:`y` the
        normalized input, a block computes

        .. math::

           h = z + \mathrm{Drop}\big(\mathrm{Attn}(\mathrm{LN}_1(z))\big),
           \qquad z' = h + \mathrm{MLP}(\mathrm{LN}_2(h)),

        .. math::

           \mathrm{Attn}_n = \sum_m \mathrm{softmax}_m\!\left(
           \frac{(\mathcal{R}_{t_n} \mathrm{LN}_q(W_q y_n))^\top
           \mathcal{R}_{t_m} \mathrm{LN}_k(W_k y_m)}{\sqrt{d_h}} + M_m
           \right) W_v y_m,

        where :math:`\mathcal{R}_t` is the rotary rotation
        :footcite:`su2024roformer` of the window index :math:`t` with
        frequencies :math:`\beta^{-2j/d_h}`, :math:`\beta` = ``rope_base``, and
        :math:`M_m = 0` for a valid key and :math:`-\infty` otherwise. In each
        head the queries and keys are layer-normalized and rotated, so a logit
        depends on the difference of two window indices.
      - *Role.* The window order is kept without mixing time with spatial
        identity: all S-D pairs of a window share its index, and the spatial
        identity enters only through the coordinate embedding.

    - **Readout and head** (``pool``, ``final_layer``).

      - **Operations.** ``"temporal_mean"`` (default): mean over the valid
        windows, one ``embed_dim`` vector per channel, concatenated;
        ``"channel_temporal_mean"``: mean over the valid windows and
        channels; ``"flatten"``: all tokens. Then ``nn.Linear``.
      - *Role.* The linear probe of the paper on frozen features
        (Sec. III-D): the temporal average keeps the feature size fixed.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** Within a window, the 100 input weights of the signal
      projection (no convolution); across windows, the rotary encoding of the
      window index only. The weights expect 10 Hz and 10-s windows.
    - **Spatial.** Every row of ``x`` is one chromophore channel, in any order
      and number; the HbO and HbR rows of one S-D pair share :math:`q_c`.
    - **Spectral.** None in the model. The weights expect HbO/HbR band-passed
      to 0.01-0.5 Hz and z-scored per channel.
    - **Chromophores.** Read from MNE's ``coil_type`` or the ``" hbo"`` /
      ``" hbr"`` name suffix; any other channel (e.g. HbT) raises.

    .. rubric:: Additional Mechanisms

    - **Missing data.** A window with a NaN sample, a channel without a
      location (NaN or all-zero ``loc``) and a padded channel (``ch_mask``)
      are left out of the attention and of the pooling.
    - **Scope.** The masked-autoencoder decoder and the 25 % token masking of
      pretraining (Fig. 2D, Sec. III-C) are not part of this class.
      ``final_layer`` and ``pool="channel_temporal_mean"`` are additions (the
      reference returns features only); the reference's ``remainder="overlap"``
      is not offered.

    .. rubric:: Pretrained Weights

    ``mtrenirs_10s.pt`` of `microsoft/MtReNIRS
    <https://github.com/microsoft/MtReNIRS>`_ (commit 16c3082, MIT), the
    paper's epoch-95 checkpoint of the 10-s model, holds the inference encoder
    only. It is on the Hugging Face Hub at `braindecode/mtrenirs-pretrained
    <https://huggingface.co/braindecode/mtrenirs-pretrained>`_; the head is
    not pretrained. It expects 10 Hz HbO/HbR in 100-sample windows, with
    positions on 10-05 slots, preprocessed as in Sec. III-B (Fig. 2A): optical
    density, temporal derivative distribution repair and the modified
    Beer-Lambert law; a 0.01-0.5 Hz band-pass, resampling to 10 Hz and removal
    of flat or low-variance chromophore channels; each chromophore channel
    z-scored over its run and clipped to :math:`[-6, 6]`; downstream windows
    [onset, onset + 10 s) without a hemodynamic-response shift. For the
    band-pass, the authors' released ``from_mne_raw`` uses MNE's default FIR,
    which leaves 43.8 % of the window power above 0.5 Hz; a zero-phase
    third-order Butterworth, ``raw.filter(0.01, 0.5, method="iir",
    iir_params=dict(order=3, ftype="butter"))``, leaves 1.1 % and reproduces
    the paper's numbers (:gh:`1564`). On the released checkpoint, the
    ``"foundation"`` tokens of :meth:`encode` equal the output of the authors'
    ``MtReNIRSEncoder`` (MtReNIRS@16c3082) with max-abs difference 0.0:

    .. code-block:: python

        # raw: MNE Raw with the 10-05 slot positions in loc[:3]
        model = MtReNIRS.from_pretrained(
            "braindecode/mtrenirs-pretrained",
            chs_info=raw.info["chs"],
            n_outputs=3,
            n_times=100,
        )

    `License <https://github.com/microsoft/MtReNIRS/blob/main/LICENSE.txt>`_

    Parameters
    ----------
    window_samples : int, default=100
        Samples per token window :math:`x_{c,t}`: 10 s at 10 Hz.
    embed_dim : int, default=128
        Token and Transformer width.
    depth : int, default=6
        Number of Transformer blocks.
    num_heads : int, default=4
        Attention heads; ``embed_dim // num_heads`` must be even for RoPE.
    mlp_dim : int, default=512
        Hidden width of each block's feed-forward network.
    n_fourier_bands : int, default=6
        Fourier bands :math:`2^k \pi` per coordinate of :math:`u_c`; six give
        48 features.
    spatial_hidden_dim : int, default=128
        Hidden width of the spatio-chromophore MLP (48 -> 128 -> 128 by
        default).
    rope_base : float, default=500.0
        Rotary base :math:`\beta` of the window index.
    qk_norm : bool, default=True
        Per-head LayerNorm of queries and keys.
    pool : {"temporal_mean", "channel_temporal_mean", "flatten"}, default="temporal_mean"
        Readout of the ``(batch, n_windows, n_chans, embed_dim)`` tokens (see
        Macro Components); ``"channel_temporal_mean"`` takes ``ch_mask``.
    on_non_divisible : {"crop", "pad", "error"}, default="crop"
        A short last window is dropped (the reference's ``remainder="drop"``),
        zero-padded or refused.
    coordinate_center : tuple of float, default=(0.00021245056996122003, -0.022675372660160065, 0.020025914534926414)
        Centre :math:`\mu` (m) subtracted from :math:`q_c`; the checkpoint's
        ``data.pos_center``.
    coordinate_scale : float, default=0.09265176206827164
        RMS scale :math:`\sigma` (m) dividing the centred :math:`q_c`; the
        checkpoint's ``data.pos_scale``.
    drop_prob : float, default=0.1
        Dropout of attention weights, attention output and MLP.
    activation : type[nn.Module], default=nn.GELU
        Activation of the spatio-chromophore MLP and the feed-forward blocks.

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: final_layer.weight

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
        window_samples: int = 100,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 512,
        n_fourier_bands: int = 6,
        spatial_hidden_dim: int = 128,
        rope_base: float = 500.0,
        qk_norm: bool = True,
        pool: Literal[
            "temporal_mean", "channel_temporal_mean", "flatten"
        ] = "temporal_mean",
        on_non_divisible: Literal["crop", "pad", "error"] = "crop",
        coordinate_center: tuple[float, float, float] = (
            0.00021245056996122003,
            -0.022675372660160065,
            0.020025914534926414,
        ),
        coordinate_scale: float = 0.09265176206827164,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
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

        # Checkers
        # RoPE rotates pairs of features of each head, so every head needs an
        # even number of features.
        if embed_dim % num_heads or (embed_dim // num_heads) % 2:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads}) with an even head_dim for rotary position "
                "embeddings."
            )
        # The readout is one of the three the paper's heads use; any other
        # name has no defined feature size.
        pools = ("temporal_mean", "channel_temporal_mean", "flatten")
        if pool not in pools:
            raise ValueError(f"pool must be one of {pools}, got {pool!r}.")

        # Parameters
        self.window_samples = window_samples
        self.pool = pool
        self.on_non_divisible = on_non_divisible
        # Number of temporal windows T cut from the recording; raises for a bad
        # on_non_divisible
        self.n_windows = patch_count(self.n_times, window_samples, on_non_divisible)
        # u_c = (x, y, z, chromophore): three standardized optode coordinates
        # and the HbO/HbR flag
        self.n_coordinates = 4
        # Size of the feature vector that each readout hands to the head
        self.feature_dim = {
            "temporal_mean": self.n_chans * embed_dim,
            "channel_temporal_mean": embed_dim,
            "flatten": self.n_windows * self.n_chans * embed_dim,
        }[pool]
        drop_prob = float(drop_prob)

        # Layers
        # Following paper nomenclature

        # Token embedding: z_{c,t} = signal embedding of the window x_{c,t}
        # plus the spatio-chromophore embedding of u_c. One token per
        # (channel, window), so any montage is a variable-length token set.
        self.embed = _TokenEmbedding(
            window_samples=window_samples,
            embed_dim=embed_dim,
            n_coordinates=self.n_coordinates,
            n_fourier_bands=n_fourier_bands,
            spatial_hidden_dim=spatial_hidden_dim,
            activation=activation,
        )

        # Transformer encoder: ``depth`` pre-norm blocks whose attention spans
        # all channels and windows jointly, with QK-norm and a rotary embedding
        # of the window index (the only place where time order enters).
        blocks = [
            _TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                drop_prob=drop_prob,
                rope_base=rope_base,
                qk_norm=qk_norm,
                activation=activation,
            )
            for _ in range(depth)
        ]
        self.encoder = _Encoder(blocks=blocks, embed_dim=embed_dim)

        # u_c of the channels in chs_info, standardized with the pretraining
        # statistics; not persistent, so a checkpoint carries no montage.
        default_coordinates = _coordinates(
            self._chs_info, coordinate_center, coordinate_scale
        )
        self.register_buffer(
            "default_coordinates",
            torch.from_numpy(default_coordinates),
            persistent=False,
        )

        # Shape layers of the tokenization: windows of one channel become tokens
        self.split_windows = Rearrange(
            "batch chans (windows samples) -> batch windows chans samples",
            samples=window_samples,
        )
        self.merge_tokens = Rearrange(
            "batch windows chans samples -> batch (windows chans) samples"
        )
        self.merge_token_mask = Rearrange(
            "batch windows chans -> batch (windows chans)"
        )
        self.merge_coordinates = Rearrange(
            "batch windows chans coords -> batch (windows chans) coords"
        )

        # Shape layers of the readout
        self.flatten_channels = Rearrange("batch chans dim -> batch (chans dim)")
        self.flatten_tokens = Rearrange(
            "batch windows chans dim -> batch (windows chans dim)"
        )

        # Classification head: one linear layer on the pooled features
        self.final_layer = nn.Linear(self.feature_dim, self.n_outputs)
        self._n_times_step = self.window_samples
        self._snaps_in_prepare_inputs = False
        self._min_n_times = 1 if self.on_non_divisible == "pad" else self.window_samples
        if self.pool == "flatten":
            self._time_axis = "fixed"

    def _accepts_any_n_chans(self, args: tuple, kwargs: dict) -> bool:
        if super()._accepts_any_n_chans(args, kwargs):
            return True
        ret_feat = self._call_arg(args, kwargs, "return_features", False)
        coords = self._call_arg(args, kwargs, "coordinates")
        return coords is not None and (
            self.pool == "channel_temporal_mean" or bool(ret_feat)
        )

    def encode(
        self,
        x: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize the recording and encode its (channel, window) tokens.

        Parameters
        ----------
        x : torch.Tensor
            Hemoglobin concentrations, ``(batch, n_chans, n_times)``. A NaN
            sample masks its window.
        coordinates : torch.Tensor or None
            :math:`u_c` of the channels of ``x``, ``(n_chans, 4)`` or
            ``(batch, n_chans, 4)``: standardized position and chromophore
            (0 HbO, 1 HbR). NaN positions mark channels without a location.
            ``None`` uses the coordinates built from ``chs_info``.
        ch_mask : torch.Tensor or None
            ``True`` for real channels, ``(batch, n_chans)``.

        Returns
        -------
        dict of torch.Tensor
            ``"foundation"``, the encoded tokens, ``(batch, n_windows,
            n_chans, embed_dim)``, zero where a token is not valid, and
            ``"valid"``, the validity of each token, ``(batch, n_windows,
            n_chans)``.
        """
        batch_size, n_chans, _ = x.shape

        # Spatio-chromophore coordinates u_c: from chs_info unless given. The
        # model has no channel table, so it cannot run without positions.
        if coordinates is None:
            coord = self.default_coordinates
            if coord.numel() == 0:
                raise ValueError(
                    "MtReNIRS needs channel positions and chromophores: chs_info "
                    "with channel locations (loc[:3]) or the coordinates argument."
                )
            if coord.shape[0] != n_chans:
                raise ValueError(
                    "x and chs_info have different channel counts: pass the "
                    "coordinates of the channels of x."
                )
        else:
            coord = coordinates
            if coord.shape[-2] != n_chans:
                raise ValueError(
                    "coordinates and x have different channel counts: pass one "
                    "row per channel of x."
                )

        # Temporal windowing: the recording is cut into T non-overlapping
        # windows of window_samples; a partial last window is cropped or padded
        # (batch, n_chans, n_times) -> (batch, n_chans, n_windows * window_samples)
        x = pad_or_crop_to_patches(x, self.window_samples, self.on_non_divisible)
        # Windows x_{c,t}, one per channel c and window t
        # -> (batch, n_windows, n_chans, window_samples)
        windows = self.split_windows(x)
        n_windows = windows.shape[1]

        # Validity of each window: a NaN sample (missing or bad segment) masks
        # the whole window
        # (batch, n_windows, n_chans, window_samples)
        finite = torch.isfinite(windows)
        # -> (batch, n_windows, n_chans)
        valid = finite.all(dim=-1)

        # u_c for every sample of the batch
        # (n_chans, 4) or (batch, n_chans, 4) -> (batch, n_chans, 4)
        coord = coord.to(device=x.device, dtype=windows.dtype)
        coord = coord.expand(batch_size, -1, -1)

        # A channel without a location (NaN position) is masked; its u_c is
        # set to 0 so that the Fourier features stay finite
        # (batch, n_chans, 3)
        positions = coord[..., :3]
        positions_finite = torch.isfinite(positions)
        # -> (batch, n_chans)
        located = positions_finite.all(dim=-1)
        coord_zeros = torch.zeros_like(coord)
        # The flag gets a trailing axis, to broadcast over the 4 coordinates
        # (batch, n_chans, 4)
        coord = torch.where(located[..., None], coord, coord_zeros)
        # The flag gets a window axis, to broadcast over the windows
        # (batch, n_chans) -> (batch, 1, n_chans); valid: (batch, n_windows, n_chans)
        valid = valid & located[:, None]

        # Padded channels of a batch that mixes montages are masked too
        if ch_mask is not None:
            # (batch, n_chans) -> (batch, 1, n_chans), over the windows
            ch_mask = ch_mask[:, None].to(torch.bool)
            # (batch, n_windows, n_chans)
            valid = valid & ch_mask

        # Token set: every (window, channel) pair is one token, windows first
        # (batch, n_windows, n_chans) -> (batch, n_windows * n_chans)
        valid_tokens = self.merge_token_mask(valid)
        # NaN samples are replaced by 0 so that masked tokens stay finite
        # (batch, n_windows, n_chans, window_samples)
        window_zeros = torch.zeros_like(windows)
        x = torch.where(finite, windows, window_zeros)
        # -> (batch, n_windows * n_chans, window_samples)
        x = self.merge_tokens(x)

        # u_c of every token: a channel keeps its coordinates in all windows
        # (batch, n_chans, 4) -> (batch, 1, n_chans, 4)
        coord_tokens = coord[:, None]
        # -> (batch, n_windows, n_chans, 4)
        coord_tokens = coord_tokens.expand(-1, n_windows, -1, -1)
        # -> (batch, n_windows * n_chans, 4)
        coord_tokens = self.merge_coordinates(coord_tokens)

        # RoPE position of a token: its window index t, shared by all channels
        # (n_windows,) -> (n_windows * n_chans,)
        window_indices = torch.arange(n_windows, device=x.device)
        window_indices = window_indices.repeat_interleave(n_chans)
        # -> (batch, n_windows * n_chans)
        window_indices = window_indices.expand(batch_size, -1)

        # Token embedding z_{c,t}: signal embedding of x_{c,t} plus the
        # spatio-chromophore embedding of u_c
        # (batch, n_windows * n_chans, window_samples)
        # -> (batch, n_windows * n_chans, embed_dim)
        x = self.embed(x, coord_tokens)

        # Transformer encoder: joint attention over channels and windows,
        # invalid tokens are never attended
        # (batch, n_windows * n_chans, embed_dim), unchanged
        x = self.encoder(x, window_indices, valid_tokens)

        # Invalid tokens are zeroed so that they cannot reach the readout
        # (batch, n_windows * n_chans) -> (batch, n_windows * n_chans, 1)
        token_weight = valid_tokens[..., None].to(x.dtype)
        # (batch, n_windows * n_chans, embed_dim)
        x = x * token_weight

        # Back to the (window, channel) grid for the readout
        # -> (batch, n_windows, n_chans, embed_dim)
        x = x.unflatten(1, [n_windows, n_chans])
        return {"foundation": x, "valid": valid}

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head with a new one of ``n_outputs``."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.feature_dim, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        coordinates: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Hemoglobin concentrations, ``(batch, n_chans, n_times)``.
        return_features : bool, default=False
            Return the pooled features instead of the logits.
        coordinates : torch.Tensor or None
            :math:`u_c` of the channels of ``x``, as in :meth:`encode`.
        ch_mask : torch.Tensor or None
            ``True`` for real channels, ``(batch, n_chans)``
            (:func:`~braindecode.functional.check_ch_mask`). Needs
            ``pool="channel_temporal_mean"``; padded channels are left out of
            the attention and the pooling.

        Returns
        -------
        torch.Tensor or dict
            Logits ``(batch, n_outputs)``, or with ``return_features`` a dict
            with ``features`` ``(batch, feature_dim)`` and ``cls_token``
            (``None``: the model has no class token).
        """
        # A channel mask only makes sense when the readout averages over
        # channels: the other two keep one feature slot per channel.
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None and self.pool != "channel_temporal_mean":
            raise ValueError(
                "ch_mask needs pool='channel_temporal_mean': the other heads "
                "are tied to the channels."
            )

        # Tokenization, token embedding z_{c,t} and Transformer encoder
        # (batch, n_chans, n_times) -> (batch, n_windows, n_chans, embed_dim)
        # valid: (batch, n_windows, n_chans)
        encoded = self.encode(x, coordinates, ch_mask)
        x, valid = encoded["foundation"], encoded["valid"]

        # Readout of the encoded tokens
        if self.pool == "flatten":
            # All tokens are kept and concatenated
            # -> (batch, n_windows * n_chans * embed_dim)
            x = self.flatten_tokens(x)
        else:
            # Mean over the valid windows ("temporal_mean", one vector per
            # channel) or over the valid windows and channels
            # ("channel_temporal_mean", one vector per recording)
            dims: List[int] = [1] if self.pool == "temporal_mean" else [1, 2]
            # (batch, n_windows, n_chans) -> (batch, n_windows, n_chans, 1)
            weight = valid[..., None].to(x.dtype)
            # Number of valid tokens averaged; at least 1, so a channel with
            # no valid window gives zeros and not NaN
            # -> (batch, n_chans, 1) or (batch, 1)
            count = weight.sum(dim=dims)
            count = count.clamp_min(1.0)
            # Sum of the valid tokens
            # -> (batch, n_chans, embed_dim) or (batch, embed_dim)
            x = x * weight
            x = x.sum(dim=dims)
            x = x / count
            if self.pool == "temporal_mean":
                # The per-channel vectors are concatenated
                # (batch, n_chans, embed_dim) -> (batch, n_chans * embed_dim)
                x = self.flatten_channels(x)

        if return_features:
            return {"features": x, "cls_token": None}

        # Classification head
        # (batch, feature_dim) -> (batch, n_outputs)
        x = self.final_layer(x)
        return x


class _TokenEmbedding(nn.Module):
    """Token embedding :math:`z_{c,t}` of every (channel, window) pair.

    Steps, with :math:`u_c` the coordinates and chromophore of the channel:

    1. Signal embedding: a linear projection of the window :math:`x_{c,t}`,
       ``(batch, n_tokens, window_samples)`` -> ``(batch, n_tokens, embed_dim)``.
    2. Fourier features :math:`\\phi(u_c)`: the sine and cosine of
       :math:`2^k \\pi u_{c,j}` for every band :math:`k` and coordinate
       :math:`j`, ``(batch, n_tokens, 2 * n_coordinates * n_fourier_bands)``.
    3. Spatio-chromophore embedding: an MLP of :math:`\\phi(u_c)`,
       ``(batch, n_tokens, embed_dim)``.
    4. The two embeddings are summed, ``(batch, n_tokens, embed_dim)``.
    """

    def __init__(
        self,
        window_samples: int,
        embed_dim: int,
        n_coordinates: int,
        n_fourier_bands: int,
        spatial_hidden_dim: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        # Parameters
        # phi(u_c) holds a sine and a cosine for every coordinate and band
        n_sin_cos = 2
        fourier_dim = n_coordinates * n_fourier_bands * n_sin_cos

        # Layers
        # Signal embedding: a linear projection of the raw window x_{c,t}
        self.signal_proj = nn.Linear(window_samples, embed_dim)

        # Fourier frequencies 2^k pi, k = 0, ..., n_fourier_bands - 1: octave
        # bands let the MLP resolve both coarse and fine optode distances.
        freqs = (2.0 ** torch.arange(n_fourier_bands).float()) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

        # Spatio-chromophore embedding: an MLP of the Fourier features
        # phi(u_c), so the position and the chromophore of a channel are an
        # input and not a learned per-channel table
        self.spatial_mlp = nn.Sequential(
            nn.Linear(fourier_dim, spatial_hidden_dim),
            activation(),
            nn.Linear(spatial_hidden_dim, embed_dim),
        )

        # Shape layer of the Fourier features
        self.merge_bands = Rearrange(
            "batch tokens coords bands -> batch tokens (coords bands)"
        )

    def forward(self, signal: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Embed every (channel, window) token.

        Parameters
        ----------
        signal : torch.Tensor
            Windows :math:`x_{c,t}`, ``(batch, n_tokens, window_samples)``.
        coordinates : torch.Tensor
            :math:`u_c` of each token, ``(batch, n_tokens, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            Tokens :math:`z_{c,t}`, ``(batch, n_tokens, embed_dim)``.
        """
        # Fourier frequencies 2^k pi on the device and dtype of the input
        # (n_fourier_bands,)
        freqs = self.freqs.to(device=coordinates.device, dtype=coordinates.dtype)

        # Angles 2^k pi u_{c,j} of every coordinate j and band k:
        # (batch, n_tokens, n_coordinates, n_fourier_bands)
        angles = coordinates[..., None] * freqs
        # -> (batch, n_tokens, n_coordinates * n_fourier_bands)
        angles = self.merge_bands(angles)

        # Signal embedding of the window x_{c,t}
        # (batch, n_tokens, window_samples) -> (batch, n_tokens, embed_dim)
        signal_embedding = self.signal_proj(signal)

        # Fourier features phi(u_c): all sines, then all cosines
        # -> (batch, n_tokens, 2 * n_coordinates * n_fourier_bands)
        sine = angles.sin()
        cosine = angles.cos()
        fourier = torch.cat([sine, cosine], dim=-1)

        # Spatio-chromophore embedding: MLP of phi(u_c)
        # -> (batch, n_tokens, embed_dim)
        spatial_embedding = self.spatial_mlp(fourier)

        # Token z_{c,t}: the two embeddings are summed
        # -> (batch, n_tokens, embed_dim)
        x = signal_embedding + spatial_embedding
        return x


class _RotarySelfAttention(nn.Module):
    """The authors' ``MultiHeadAttention``: per-head QK LayerNorm, a rotary
    embedding of the window index and a key padding mask.

    Private because the attention of :mod:`braindecode.modules` takes neither
    per-token positions nor a mask; ``qkv`` and ``proj`` keep timm's names, as
    in the released ``mtrenirs_10s.pt``.

    Steps, for tokens ``(batch, n_tokens, embed_dim)``:

    1. One fused projection and a split into heads,
       ``3 x (batch, heads, n_tokens, head_dim)``.
    2. LayerNorm of the queries and keys of each head.
    3. Rotary rotation of the queries and keys by the window index of their
       token, shared by the channels of a window.
    4. Scaled dot-product attention with an additive key mask: ``-inf`` at the
       invalid tokens, unless none is valid, so that the softmax stays finite.
    5. The heads are merged and projected, ``(batch, n_tokens, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_prob: float,
        rope_base: float,
        qk_norm: bool,
    ):
        super().__init__()
        # Parameters
        self.head_dim = embed_dim // num_heads
        self.rope_base = float(rope_base)
        # Query, key and value come out of one fused projection
        n_projections = 3

        # Layers
        # Fused query/key/value projection of every token
        self.qkv = nn.Linear(embed_dim, n_projections * embed_dim, bias=True)
        # Dropout on the attention weights
        self.attn_drop = nn.Dropout(drop_prob)
        # Output projection that mixes the heads back to the token width
        self.proj = nn.Linear(embed_dim, embed_dim)
        # QK-norm: a LayerNorm on each head's queries and keys bounds the
        # attention logits, which keeps pretraining stable.
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        # Shape layers
        self.split_heads = Rearrange(
            "batch tokens (qkv heads dim) -> qkv batch heads tokens dim",
            qkv=n_projections,
            heads=num_heads,
        )
        self.merge_heads = Rearrange(
            "batch heads tokens dim -> batch tokens (heads dim)"
        )

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Self-attention over all (channel, window) tokens.

        Parameters
        ----------
        x : torch.Tensor
            Tokens, ``(batch, n_tokens, embed_dim)``.
        positions : torch.Tensor
            Window index :math:`t` of each token, ``(batch, n_tokens)``.
        valid : torch.Tensor
            ``True`` for tokens that can be attended, ``(batch, n_tokens)``.

        Returns
        -------
        torch.Tensor
            Attended tokens, ``(batch, n_tokens, embed_dim)``.
        """
        # Fused projection, then one query, key and value per head
        # (batch, n_tokens, embed_dim) -> (batch, n_tokens, 3 * embed_dim)
        qkv = self.qkv(x)
        # -> 3 x (batch, heads, n_tokens, head_dim)
        query, key, value = self.split_heads(qkv).unbind(0)

        # QK-norm: LayerNorm of the queries and keys of each head
        # (batch, heads, n_tokens, head_dim), unchanged
        query = self.q_norm(query)
        key = self.k_norm(key)

        # RoPE frequencies base^(-2i / head_dim) of the feature pairs
        # (head_dim // 2,)
        inv_freq = rotary_frequencies(
            self.head_dim, self.rope_base, positions.device, query.dtype
        )
        # RoPE angles t * frequency: the position of a token is its window
        # index t, so channels of one window share a rotation
        # (batch, n_tokens) -> (batch, n_tokens, head_dim)
        positions = positions.to(query.dtype)
        angles = rotary_angles(positions, inv_freq, interleaved=False)
        cosine = angles.cos()
        sine = angles.sin()
        # (batch, n_tokens, head_dim) -> (batch, 1, n_tokens, head_dim)
        cosine = cosine[:, None]
        sine = sine[:, None]

        # Rotate queries and keys (not values): their dot product then
        # depends only on the difference of the two window indices
        # (batch, heads, n_tokens, head_dim), unchanged
        query = apply_rotary(query, cosine, sine, interleaved=False)
        key = apply_rotary(key, cosine, sine, interleaved=False)

        # Key padding mask: invalid tokens (NaN window, channel without a
        # location, padded channel) are never attended. A sample with no valid
        # token attends to all of them, so its softmax stays finite.
        # (batch, n_tokens) -> (batch, 1)
        any_valid = valid.any(dim=-1, keepdim=True)
        # (batch, n_tokens)
        attend = valid | ~any_valid
        # (batch, n_tokens) -> (batch, 1, 1, n_tokens)
        attend = attend[:, None, None]
        # Additive bias: 0 for attended keys, -inf for blocked ones
        # (batch, 1, 1, n_tokens)
        bias = torch.zeros_like(attend, dtype=query.dtype)
        blocked = ~attend
        bias = bias.masked_fill(blocked, float("-inf"))

        # Scaled dot-product attention, softmax(q k^T / sqrt(head_dim) + bias) v
        # (batch, heads, n_tokens, head_dim)
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=dropout_p,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        # Concatenate the heads and project back to the token width
        # (batch, heads, n_tokens, head_dim) -> (batch, n_tokens, embed_dim)
        x = self.merge_heads(x)
        x = self.proj(x)
        return x


class _TransformerBlock(nn.Module):
    """Pre-norm block: QK-norm rotary self-attention, then feed-forward.

    Private because its attention takes per-token positions and a validity mask
    (:class:`~braindecode.modules.PreNormBlock` calls ``attn(x)`` only) and the
    released ``mtrenirs_10s.pt`` keys follow this layout.

    Steps, for tokens ``(batch, n_tokens, embed_dim)``, each with a residual
    connection that leaves the shape unchanged:

    1. LayerNorm, rotary self-attention over all tokens, dropout.
    2. LayerNorm, feed-forward ``embed_dim -> mlp_dim -> embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        drop_prob: float,
        rope_base: float,
        qk_norm: bool,
        activation: type[nn.Module],
    ):
        super().__init__()
        # Pre-norm of the attention sub-layer
        self.norm1 = nn.LayerNorm(embed_dim)
        # Joint spatio-temporal self-attention with QK-norm and RoPE
        self.attn = _RotarySelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_prob=drop_prob,
            rope_base=rope_base,
            qk_norm=qk_norm,
        )
        # Pre-norm of the feed-forward sub-layer
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network: embed_dim -> mlp_dim -> embed_dim
        self.mlp = FeedForwardBlock(
            emb_size=embed_dim,
            expansion=1,
            drop_p=drop_prob,
            activation=activation,
            hidden_features=mlp_dim,
        )
        # Dropout on the attention branch before the residual sum
        self.drop = nn.Dropout(drop_prob)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """One pre-norm Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Tokens, ``(batch, n_tokens, embed_dim)``.
        positions : torch.Tensor
            Window index of each token, ``(batch, n_tokens)``.
        valid : torch.Tensor
            ``True`` for tokens that can be attended, ``(batch, n_tokens)``.

        Returns
        -------
        torch.Tensor
            Tokens, ``(batch, n_tokens, embed_dim)``.
        """
        # Attention sub-layer: x + Dropout(Attention(LayerNorm(x)))
        # (batch, n_tokens, embed_dim), unchanged
        attended = self.norm1(x)
        attended = self.attn(attended, positions, valid)
        attended = self.drop(attended)
        x = x + attended

        # Feed-forward sub-layer: x + MLP(LayerNorm(x))
        # (batch, n_tokens, embed_dim), unchanged
        update = self.norm2(x)
        update = self.mlp(update)
        x = x + update
        return x


class _Encoder(nn.Module):
    """The Transformer blocks and a final LayerNorm.

    Steps, for tokens ``(batch, n_tokens, embed_dim)``: ``depth``
    :class:`_TransformerBlock` in sequence, each given the window index and the
    validity of every token, then a LayerNorm of the tokens. The shape is
    ``(batch, n_tokens, embed_dim)`` throughout.
    """

    def __init__(self, blocks: list[nn.Module], embed_dim: int):
        super().__init__()
        # The stack of pre-norm Transformer blocks
        self.blocks = nn.ModuleList(blocks)
        # Final LayerNorm: pre-norm blocks leave the residual stream
        # unnormalized, so the tokens are normalized once before the readout.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Encode the tokens.

        Parameters
        ----------
        x : torch.Tensor
            Tokens :math:`z_{c,t}`, ``(batch, n_tokens, embed_dim)``.
        positions : torch.Tensor
            Window index of each token, ``(batch, n_tokens)``.
        valid : torch.Tensor
            ``True`` for tokens that can be attended, ``(batch, n_tokens)``.

        Returns
        -------
        torch.Tensor
            Encoded tokens, ``(batch, n_tokens, embed_dim)``.
        """
        # Transformer blocks, each attending over all channels and windows
        # (batch, n_tokens, embed_dim), unchanged
        for block in self.blocks:
            x = block(x, positions, valid)

        # Final LayerNorm of the encoded tokens
        # (batch, n_tokens, embed_dim), unchanged
        x = self.norm(x)
        return x


def _coordinates(chs_info, coordinate_center, coordinate_scale) -> np.ndarray:
    """``(n_chans, 4)`` :math:`u_c`: ``loc[:3]`` minus ``coordinate_center``,
    divided by ``coordinate_scale`` (NaN when unknown or all zero), and 0 for
    HbO, 1 for HbR, as the official ``chromophore_map``; ``(0, 4)`` without
    ``chs_info`` or any location."""
    if chs_info is None:
        return np.zeros((0, 4), np.float32)
    chromophores = fnirs_chromophores_from_chs_info(chs_info)
    if "unknown" in chromophores:
        name = chs_info[chromophores.index("unknown")].get("ch_name")
        raise ValueError(f"MtReNIRS needs HbO/HbR channels, got {name!r}.")
    loc = extract_channel_locations_from_chs_info(chs_info, fill_missing=True)
    loc = np.asarray(loc)
    loc[(loc == 0).all(axis=1)] = np.nan  # MNE without a montage: NaN or zeros
    if not np.isfinite(loc).any():  # forward raises unless coordinates are given
        return np.zeros((0, 4), np.float32)
    center = np.asarray(coordinate_center, dtype=np.float32)
    xyz = (loc - center) / float(coordinate_scale)
    is_hbr = np.equal(chromophores, "hbr")
    return np.c_[xyz, is_hbr].astype(np.float32)

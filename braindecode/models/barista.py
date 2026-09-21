# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
# License: USC academic, non-commercial; see NOTICE.txt.
"""BaRISTA, adapted from https://github.com/ShanechiLab/BaRISTA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import GatedLinearUnit, PatchTokenizer


class BaRISTA(EEGModuleMixin, nn.Module, license="other"):
    r"""BaRISTA from Oganesian, Hashemi and Shanechi (2025) [Oganesian2025]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`
    :bdg-dark-line:`Channel`

    .. versionadded:: 1.8.2

    An intracranial EEG encoder with channel-wise temporal tokenization and
    joint space-time attention. Spatial embeddings can represent electrode
    coordinates, atlas parcels, or lobes.

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

    .. rubric:: Additional Mechanisms

    **Spatial metadata**

    Pass dataset-provided ``spatial_indices`` in input-channel order:
    three coordinate indices per channel for ``"coords"``, or one region
    index per channel for ``"parcels"`` and ``"lobes"``. Brain Treebank's
    NEMAR dataset ``nm000253`` supplies these in ``electrodes.tsv`` as
    ``x, y, z``, ``barista_parcel_index`` and ``barista_lobe_index``.
    Region index 0 is unknown and contributes no spatial embedding.

    If coordinate indices are omitted, ``"coords"`` bins finite, same-frame
    ``chs_info`` positions from metres onto a centred 1 mm grid. This fallback
    is not Brain Treebank's voxel convention; use the dataset's indices for
    that dataset. ``"none"`` disables spatial encoding.

    **Pre-trained weights**

    The reference publishes three checkpoints, one per spatial scale. Loading
    those checkpoints is not supported by this port: parameter names and the
    fused gated projection differ. Dataset-provided indices preserve spatial
    table ordering but do not convert checkpoint parameters. Local checkpoints
    saved by this class can be restored through ``from_pretrained``.

    **License**

    The reference implementation is Copyright (c) 2025 University of Southern
    California and is licensed for educational, research and non-profit use
    only; commercial use requires an agreement with the USC Stevens Center for
    Innovation. This file inherits those terms and is therefore *not* covered by
    braindecode's BSD-3-Clause license.

    .. note::
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
        Spatial embedding scale. Coordinate mode uses ``spatial_indices`` or
        falls back to ``chs_info``; region modes require ``spatial_indices``.
    spatial_indices : list of int or list of list of int, optional
        Dataset-provided embedding indices in input-channel order. Shape
        ``(n_chans, 3)`` for coordinates, with values in ``[0, coord_bins)``;
        shape ``(n_chans,)`` for parcels or lobes, with values in ``[0, 121)``
        or ``[0, 21)`` respectively. Region index 0 denotes unknown. Use the
        dataset's BaRISTA index mapping, not arbitrary atlas label numbers.
    coord_bins : int
        Number of slots per coordinate axis, default 200. Also the grid width
        in millimetres when deriving indices from ``chs_info``.
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
        ``"mean"`` averages them instead. Both modes use the channel and patch
        counts fixed at construction.
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
        spatial_indices: list[int] | list[list[int]] | None = None,
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
        for name, value in {
            "patch_size": patch_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "n_layers": n_layers,
            "mlp_ratio": mlp_ratio,
            "cnn_channels": cnn_channels,
            "cnn_kernel_size": cnn_kernel_size,
            "coord_bins": coord_bins,
            "n_chans": self.n_chans,
            "n_outputs": self.n_outputs,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if cnn_depth < 0:
            raise ValueError(f"cnn_depth must be non-negative, got {cnn_depth}.")
        if self.n_times < patch_size:
            raise ValueError("n_times must contain at least one full patch.")
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
        channels = [1] + [cnn_channels] * cnn_depth + [1]
        self.temporal_encoder = nn.Sequential(
            *[
                _DilatedConvBlock(
                    in_channels=before,
                    out_channels=after,
                    kernel_size=cnn_kernel_size,
                    dilation=2**i,
                    norm_size=patch_size,
                    final=i == cnn_depth,
                )
                for i, (before, after) in enumerate(zip(channels, channels[1:]))
            ]
        )
        self.unfold_grid = Rearrange("batch 1 seq time -> batch seq time")
        self.temporal_pooler = nn.Linear(patch_size, d_model, bias=False)

        self.spatial_emb = self._build_spatial_embedding(spatial_scale, spatial_indices)

        self.backbone = _Transformer(
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
            activation=activation,
            n_patches=self.n_patches,
            n_chans=self.n_chans,
        )

        n_tokens = self.n_patches * self.n_chans
        self.token_pooling = (
            nn.Linear(n_tokens, 1, bias=False) if pooling == "learned" else None
        )
        self.final_layer = nn.Linear(d_model, self.n_outputs)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Match the reference: Kaiming CNN, unit-variance spatial embeddings,
        # unit norm scales, and PyTorch's default Linear initialization.
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _build_spatial_embedding(
        self,
        spatial_scale: str,
        spatial_indices: list[int] | list[list[int]] | None,
    ) -> _SpatialEmbedding | None:
        """Use dataset indices, or bin MNE positions when none were supplied."""
        if spatial_scale == "none":
            return None
        is_coords = spatial_scale == "coords"
        n_slots = (
            self.coord_bins
            if is_coords
            else {"parcels": 121, "lobes": 21}[spatial_scale]
        )
        if spatial_indices is None:
            if not is_coords:
                raise ValueError(f"spatial_indices is required for {spatial_scale!r}.")
            locations = extract_channel_locations_from_chs_info(
                self._chs_info, num_channels=self.n_chans
            )
            if locations is None or len(locations) != self.n_chans:
                raise ValueError(
                    "Provide spatial_indices or chs_info positions for every channel."
                )
            positions = torch.as_tensor(locations, dtype=torch.float32)
            if not positions.isfinite().all():
                raise ValueError("chs_info positions must be finite.")
            if len({ch.get("coord_frame") for ch in self.chs_info}) > 1:
                raise ValueError(
                    "chs_info must use the same coordinate frame for all channels."
                )
            indices = (-1000 * positions).round() + self.coord_bins // 2
            indices = indices.clamp(0, self.coord_bins - 1)
        else:
            indices = torch.as_tensor(spatial_indices)
        expected = (self.n_chans, 3) if is_coords else (self.n_chans,)
        if tuple(indices.shape) != expected:
            raise ValueError(f"spatial_indices must have shape {expected}.")
        if (
            indices.is_complex()
            or indices.dtype == torch.bool
            or not indices.isfinite().all()
            or (indices < 0).any()
            or (indices >= n_slots).any()
            or (indices != indices.long()).any()
        ):
            raise ValueError(
                f"spatial_indices must contain integers in [0, {n_slots})."
            )
        indices = indices.long()
        return _SpatialEmbedding(
            indices.T if is_coords else indices.unsqueeze(0),
            self.d_model,
            n_slots,
            padding_idx=None if is_coords else 0,
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the linear classification head for a new ``n_outputs``."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(
            self.d_model,
            n_outputs,
            device=self.final_layer.weight.device,
            dtype=self.final_layer.weight.dtype,
        ).train(self.training)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Encode an iEEG batch into class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            Return the pooled embedding instead of logits.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape ``(batch, n_outputs)``, or a dictionary containing
            ``features`` of shape ``(batch, d_model)`` and ``cls_token=None``.
        """
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, n_chans, n_times).")
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

        latents = self.backbone(tokens)

        if self.token_pooling is not None:
            features = self.token_pooling(latents.transpose(1, 2)).squeeze(dim=-1)
        else:
            features = latents.mean(dim=1)
        if not torch.jit.is_scripting() and return_features:
            return {"features": features, "cls_token": None}  # nosec B105
        return self.final_layer(features)


class _SpatialEmbedding(nn.Module):
    """Sum of one learned embedding table per spatial dimension."""

    def __init__(
        self,
        indices: torch.Tensor,
        d_model: int,
        n_slots: int,
        padding_idx: int | None,
    ):
        super().__init__()
        self.indices: torch.Tensor
        self.register_buffer("indices", indices, persistent=False)
        self.tables = nn.ModuleList(
            [
                nn.Embedding(n_slots, d_model, padding_idx=padding_idx)
                for _ in range(indices.shape[0])
            ]
        )

    def forward(self) -> torch.Tensor:
        """Return the ``(n_chans, d_model)`` spatial encoding of the montage."""
        return torch.stack(
            [table(self.indices[dim]) for dim, table in enumerate(self.tables)]
        ).sum(dim=0)


class _DilatedConvBlock(nn.Module):
    """Two dilated convolutions over time with a residual stream."""

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
        self.projector: nn.Module
        if in_channels != out_channels or final:
            self.projector = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projector(x)
        x = self.activation(self.norm(self.conv1(x)))
        return self.activation(self.norm(self.conv2(x))) + residual


class _SamePadConv(nn.Module):
    """Dilated convolution along time that preserves the input length."""

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
        # For even kernels, the reference crops the rightmost sample. Native
        # padding="same" pads on the opposite side and changes alignment.
        self.remove = 1 - receptive_field % 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.remove > 0:
            x = x[..., : -self.remove]
        return x


class _Transformer(nn.Module):
    """Stack of pre-norm transformer blocks over the interleaved token sequence."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_heads: int,
        mlp_ratio: int,
        drop_prob: float,
        activation: type[nn.Module],
        n_patches: int,
        n_chans: int,
    ):
        super().__init__()
        head_dim = d_model // num_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(n_patches).float(), inv_freq)
        angles = torch.cat((angles, angles), dim=-1).repeat_interleave(n_chans, dim=0)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)
        self.layers = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_prob=drop_prob,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        return self.norm(x)


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm block: rotary self-attention then a gated feed-forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-8)
        self.self_attn = _RotarySelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            drop_prob=drop_prob,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-8)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * mlp_ratio * d_model),
            GatedLinearUnit(activation),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(drop_prob),
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), cos, sin))
        return x + self.mlp(self.norm2(x))


class _RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings on the patch index."""

    def __init__(self, d_model: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(drop_prob)
        self.split_heads = Rearrange(
            "batch seq (heads dim) -> batch heads seq dim", heads=num_heads
        )
        self.merge_heads = Rearrange("batch heads seq dim -> batch seq (heads dim)")

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        query = _apply_rotary(query, cos, sin)
        key = _apply_rotary(key, cos, sin)

        # Dropout falls on the attention output rather than on the weights,
        # through dropout_p, because the reference disables the latter outright.
        attention = F.scaled_dot_product_attention(query, key, value)
        attention = self.dropout(attention)
        attention = self.merge_heads(attention)
        return self.o_proj(attention)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply the reference's half-split (not interleaved) rotary embedding."""
    first, second = x.chunk(2, dim=-1)
    return x * cos.to(x.dtype) + torch.cat((-second, first), dim=-1) * sin.to(x.dtype)

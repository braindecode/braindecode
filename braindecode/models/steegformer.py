# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)
"""STEEGFormer — ViT-MAE EEG foundation model.

Port of Yang et al. (2026), https://github.com/LiuyinYang1101/STEEGFormer
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    DropPath,
    FeedForwardBlock,
    MultiHeadAttention,
    PatchTokenizer,
)

# Shared montage vocabulary of the official ST-EEGFormer checkpoints: the
# learned channel embedding has one slot per entry, in this order (the slot
# index of an electrode is its position in this list). Loaded from
# ``steegformer_channels.json`` (the authors' ``channels_mapping`` in
# ``LiuyinYang1101/STEEGFormer``, MIT). Used by small/base/large (vocab 145);
# the HBN ``largeV2`` model uses a different, larger vocabulary.
with (Path(__file__).parent / "steegformer_channels.json").open() as _f:
    STEEGFORMER_CHANNEL_ORDER: list[str] = json.load(_f)
_STEEGFORMER_CHANNEL_INDEX = {
    name.upper(): i for i, name in enumerate(STEEGFORMER_CHANNEL_ORDER)
}


class STEEGFormer(EEGModuleMixin, nn.Module):
    r"""STEEGFormer from Yang et al. (2026) [Yang2026]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. figure:: https://raw.githubusercontent.com/LiuyinYang1101/STEEGFormer/main/assets/graphic_overview.png
        :align: center
        :width: 1000px

        ST-EEGFormer architecture, reproduced from the official repository
        (``LiuyinYang1101/STEEGFormer``).

    .. versionadded:: 1.6.1

    .. rubric:: Architecture Overview

    ViT-based EEG foundation model, pre-trained with a Masked Autoencoder
    (MAE) objective on raw EEG. Each channel is cut into non-overlapping
    temporal patches that are linearly embedded into tokens, augmented with
    temporal and channel positional information, prepended with a learned CLS
    token, encoded by a stack of pre-norm Transformer blocks, and read out by a
    linear head.

    .. rubric:: Macro Components

    - **Patch + token embedding** (``STEEGFormer.patch_embed``).
      *Operations:* cut each channel into ``seq = n_times // patch_size``
      non-overlapping patches of ``patch_size`` samples and linearly project
      each to an ``embed_dim`` token. *Role:* turn a ``C``-channel segment into
      ``C * seq`` tokens (one per (channel, time-patch) pair).
    - **Positional embeddings** (``STEEGFormer.temporal_pos``,
      ``STEEGFormer.channel_pos``). *Operations:* add a fixed sinusoidal
      temporal encoding over the ``seq`` patches and a learned channel
      embedding drawn from a shared montage vocabulary. *Role:* mark when (in
      time) and on which electrode each token sits, so the same electrode
      shares its embedding across datasets with different channel sets.
    - **Transformer encoder** (``STEEGFormer.encoder``). *Operations:*
      ``depth`` pre-norm ViT blocks (multi-head self-attention + MLP), reusing
      braindecode's :class:`~braindecode.modules.MultiHeadAttention` and
      :class:`~braindecode.modules.FeedForwardBlock`. *Role:* mix information
      across all (channel, time-patch) tokens.
    - **Read-out + head** (``STEEGFormer.norm``, ``STEEGFormer.final_layer``).
      *Operations:* ``"avg"`` mean-pools the patch tokens (CLS excluded);
      ``"cls"`` layer-normalises the sequence and takes the CLS token; a linear
      layer maps to ``n_outputs``. *Role:* produce the class logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* non-overlapping temporal patches with a fixed sinusoidal
      position encoding over the ``seq`` patches.
    - *Spatial (channels):* a learned channel embedding indexed through a
      shared montage vocabulary of standard electrode positions.
    - *Spectral:* none explicit; frequency content is learned implicitly by
      the patch projection and self-attention.

    .. rubric:: Additional Mechanisms

    A learned CLS token (sequence position 0) summarises the sequence for the
    ``"cls"`` read-out. Optional stochastic depth (``drop_path``) and dropout
    (``drop_prob``) regularise training; both default to ``0`` to match the
    released checkpoints.

    .. rubric:: Variants

    The released variants differ in width/depth and, for ``largeV2``, the
    channel-vocabulary size (``patch_size=16``, ``mlp_ratio=4`` throughout):

    .. list-table::
       :header-rows: 1

       * - Variant
         - ``embed_dim``
         - ``depth``
         - ``num_heads``
         - ``n_chans_pos``
       * - small
         - 512
         - 8
         - 8
         - 145
       * - base
         - 768
         - 12
         - 12
         - 145
       * - large
         - 1024
         - 24
         - 16
         - 145
       * - largeV2
         - 1024
         - 24
         - 16
         - 256

    .. rubric:: Pre-trained weights

    Ready-to-use checkpoints are re-hosted on the Hugging Face Hub under the
    braindecode organization. These repos convert the official MAE encoder
    checkpoints to braindecode's key names and include ``config.json`` plus
    ``model.safetensors``/``pytorch_model.bin``:

    .. list-table::
       :header-rows: 1

       * - Variant
         - Hub repo
         - Notes
       * - small
         - ``braindecode/STEEGFormer-small``
         - 145-slot channel vocabulary
       * - base
         - ``braindecode/STEEGFormer-base``
         - 145-slot channel vocabulary
       * - large
         - ``braindecode/STEEGFormer-large``
         - 145-slot channel vocabulary
       * - largeV2
         - ``braindecode/STEEGFormer-largeV2``
         - 256-slot HBN channel vocabulary

    Use the regular Hub API to load a re-hosted checkpoint::

        model = STEEGFormer.from_pretrained(
            "braindecode/STEEGFormer-small", n_outputs=4, n_chans=22
        )

    The re-hosted repos save complete braindecode model files, so they include a
    classification head tensor for serialization. Only the encoder weights are
    from the official MAE pretraining; pass ``n_outputs`` for the downstream
    task so the head is rebuilt as needed.

    To regenerate the re-hosted files from the official GitHub checkpoints, run
    the standalone ``convert_steegformer_checkpoints.py`` archived in each Hub
    repo; the model itself loads braindecode-format state dicts, so
    ``from_pretrained`` needs no conversion.

    .. note::
        Numerical equivalence of the encoder features with the reference
        implementation has been verified on the released checkpoints. The
        channel-to-vocabulary mapping is resolved from the electrode names
        in ``chs_info`` (looked up in :data:`STEEGFORMER_CHANNEL_ORDER`, the
        BENDR/LaBraM convention); when ``chs_info`` is absent or a name is
        unknown, it falls back to the identity mapping (channel ``i`` -> slot
        ``i``) with a warning. Pass ``chan_pos_idx`` to override explicitly.

    Parameters
    ----------
    patch_size : int
        Temporal patch size (unfold), default 16.
    embed_dim : int
        Token embedding dimension (512 / 768 / 1024 across variants).
    depth : int
        Number of Transformer encoder blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden-to-embedding ratio of the MLP blocks.
    drop_prob : float
        Dropout rate.
    drop_path : float
        Stochastic-depth rate (max of a linear schedule over ``depth``), default
        ``0`` (disabled, matching the released checkpoints).
    activation : type[nn.Module]
        Activation layer class used in the feed-forward blocks, default
        :class:`~torch.nn.GELU`.
    global_pool : str
        Token aggregation before the head (``"avg"`` or ``"cls"``).
    n_chans_pos : int
        Size of the shared montage vocabulary the channel embedding is drawn
        from (145 for small/base/large, 256 for ``largeV2``), default 145.
    chan_pos_idx : array-like of int, optional
        Montage-vocabulary slot of each input channel, shape ``(n_chans,)``.
        If omitted, it is resolved from ``chs_info`` electrode names (falling
        back to ``range(n_chans)``).

    References
    ----------
    .. [Yang2026] Yang, L., Sun, Q., Li, A. & Van Hulle, M. M. (2026). Are EEG
       foundation models worth it? Comparative evaluation with traditional
       decoders in diverse BCI tasks. The Fourteenth International Conference
       on Learning Representations (ICLR 2026).
       https://openreview.net/forum?id=5Xwm8e6vbh
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
        # --- model hyperparameters (defaults: "small" variant) ---
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        drop_path: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        global_pool: str = "avg",
        n_chans_pos: int = 145,
        chan_pos_idx=None,
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

        if global_pool not in ("avg", "cls"):
            raise ValueError(
                f"global_pool must be 'avg' or 'cls', got {global_pool!r}."
            )
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be even for the sinusoidal temporal encoding, "
                f"got {embed_dim}."
            )
        if mlp_ratio != int(mlp_ratio):
            # FeedForwardBlock expands by an integer factor; reject non-integer
            # ratios rather than silently truncating them.
            raise ValueError(f"mlp_ratio must be a whole number, got {mlp_ratio}.")
        mlp_ratio = int(mlp_ratio)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_prob = drop_prob
        self.drop_path = drop_path
        self.activation = activation
        self.global_pool = global_pool
        self.n_chans_pos = n_chans_pos

        # Map each input channel to its slot in the shared montage vocabulary.
        # Priority: explicit ``chan_pos_idx`` wins; otherwise resolve from the
        # electrode names in ``chs_info`` (BENDR/LaBraM convention); if neither
        # is usable, fall back to the identity mapping (channel i -> slot i).
        if chan_pos_idx is not None:
            chan_pos_idx = torch.as_tensor(chan_pos_idx, dtype=torch.long)
        else:
            chan_pos_idx = self._chan_pos_idx_from_chs_info()
        if chan_pos_idx.shape != (self.n_chans,):
            raise ValueError(
                f"chan_pos_idx must have shape ({self.n_chans},), got "
                f"{tuple(chan_pos_idx.shape)}."
            )
        if int(chan_pos_idx.max()) >= n_chans_pos or int(chan_pos_idx.min()) < 0:
            raise ValueError(
                f"chan_pos_idx values must be in [0, {n_chans_pos}), got range "
                f"[{int(chan_pos_idx.min())}, {int(chan_pos_idx.max())}]."
            )
        # Non-persistent: the channel->vocab-slot selection is montage-specific
        # and is recomputed from chan_pos_idx at construction, so it must NOT be
        # baked into a pushed checkpoint (it would clobber or shape-mismatch a
        # different montage on from_pretrained).
        self.register_buffer("channel_indices", chan_pos_idx, persistent=False)

        # Patch embedding + positional embeddings + CLS token.
        self.patch_embed = PatchTokenizer(
            patch_size=patch_size,
            n_times=self.n_times,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible="crop",
            projection="linear",
            output_order="patch_channel",
        )
        self.temporal_pos = _TemporalPositionalEncoding(embed_dim)
        self.channel_pos = _ChannelPositionalEmbed(n_chans_pos, embed_dim)
        self.flatten_tokens = Rearrange(
            "batch seq n_chans embed_dim -> batch (seq n_chans) embed_dim"
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)

        # Transformer encoder (reuses braindecode's attention/FFN blocks).
        # Stochastic depth follows timm's linear schedule from 0 to drop_path.
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.encoder = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    embed_dim, num_heads, drop_prob, int(mlp_ratio), dpr[i], activation
                )
                for i in range(depth)
            ]
        )

        # Classification head.
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

        # Generic trunc-normal init for every Linear/LayerNorm, then the
        # faithful-port special: a trunc-normal CLS token (the channel
        # embedding keeps its zero init from _ChannelPositionalEmbed).
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Trunc-normal init for ``Linear`` weights, unit/zero for ``LayerNorm``."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs):
        """Replace the linear classification head for a new ``n_outputs``.

        Called by :meth:`from_pretrained` when the requested number of outputs
        differs from the pre-trained checkpoint (whose head is discarded).
        """
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)

    def _chan_pos_idx_from_chs_info(self) -> torch.Tensor:
        """Resolve montage-vocabulary slots from the ``chs_info`` electrode names.

        Looks each electrode name up in :data:`STEEGFORMER_CHANNEL_ORDER`
        (case-insensitive). Falls back to the identity mapping -- and warns --
        when ``chs_info`` is absent or carries names outside the vocabulary.
        """
        try:
            chs_info = self.chs_info
        except ValueError:
            chs_info = None
        if not chs_info:
            return torch.arange(self.n_chans)
        names = [ch["ch_name"] for ch in chs_info]  # type: ignore[index]
        idx = [_STEEGFORMER_CHANNEL_INDEX.get(n.upper()) for n in names]
        missing = [n for n, j in zip(names, idx) if j is None]
        if missing:
            shown = ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
            warnings.warn(
                f"STEEGFormer: {len(missing)} channel name(s) absent from the "
                f"montage vocabulary ({shown}); falling back to the identity "
                f"channel mapping. Pass `chan_pos_idx` explicitly to align an "
                f"arbitrary montage with the pre-trained channel embedding.",
                UserWarning,
                stacklevel=2,
            )
            return torch.arange(self.n_chans)
        return torch.tensor(idx, dtype=torch.long)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Encode an EEG batch into class logits (or encoder features).

        Parameters
        ----------
        x : torch.Tensor
            EEG input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If ``True``, return the encoder tokens as
            ``{"features": patch_tokens, "cls_token": cls_token}`` instead of
            the class logits (the unified braindecode foundation-model API).

        Returns
        -------
        torch.Tensor | dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            ``{"features", "cls_token"}`` when ``return_features`` is set.
        """
        seq = x.shape[-1] // self.patch_size
        if seq < 1:
            raise ValueError(
                f"STEEGFormer requires at least one full temporal patch of "
                f"{self.patch_size} samples, got input with {x.shape[-1]} samples."
            )
        # Tokens + positional embeddings, kept on the (seq, channel) grid.
        tokens = self.patch_embed(x)  # (batch, seq, n_chans, embed_dim)
        tokens = (
            tokens + self.temporal_pos(seq) + self.channel_pos(self.channel_indices)
        )
        tokens = self.flatten_tokens(tokens)

        # Prepend the CLS token (combined with the temporal encoding at pos. 0).
        cls = self.cls_token + self.temporal_pos.cls_token_encoding()
        cls = cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.pos_drop(x)

        x = self.encoder(x)

        if return_features:
            # Unified foundation-model API: CLS at index 0, patch tokens after.
            return {"features": x[:, 1:, :], "cls_token": x[:, 0, :]}

        # Aggregate tokens, then classify. Mirrors the reference: average
        # pooling discards the CLS token and applies no final norm, whereas the
        # CLS read-out normalises the sequence first and then takes the CLS token.
        if self.global_pool == "avg":
            x = x[:, 1:].mean(dim=1)  # mean over tokens, excluding CLS
        else:  # "cls"
            x = self.norm(x)[:, 0]
        return self.final_layer(x)


class _TemporalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding over temporal patches.

    Standard sine/cosine encoding (Vaswani et al., 2017). Position ``0`` is
    reserved for the CLS token, so the ``seq`` temporal patches use positions
    ``1..seq`` (matching the released checkpoints).

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension (must be even).
    max_len : int
        Maximum number of positions (CLS token included).
    """

    def __init__(self, embed_dim: int, max_len: int = 2048):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Non-persistent: deterministic, regenerated identically at init.
        self.register_buffer("pe", pe, persistent=False)

    def cls_token_encoding(self) -> torch.Tensor:
        # Encoding at position 0, shape (embed_dim,).
        return self.pe[0]

    def forward(self, seq: int) -> torch.Tensor:
        if seq + 1 > self.max_len:
            raise ValueError(
                f"Too many temporal patches ({seq}) for max_len={self.max_len}; "
                "increase max_len."
            )
        # Positions 1..seq (position 0 is reserved for the CLS token) ->
        # (1, seq, 1, embed_dim) to broadcast over batch and channels.
        return rearrange(self.pe[1 : seq + 1], "seq embed_dim -> 1 seq 1 embed_dim")


class _ChannelPositionalEmbed(nn.Module):
    """Learned channel positional embedding over a shared montage vocabulary.

    The reference model does not learn one vector *per input channel* but one
    vector per entry of a fixed **montage vocabulary** of ``n_vocab`` standard
    electrode positions (145 in Yang et al.). Each input channel is mapped to
    its slot in that vocabulary, so the same electrode shares its embedding
    across datasets with different channel sets. Zero initialisation means the
    model starts as if there were no channel embedding and learns it from data.

    Parameters
    ----------
    n_vocab : int
        Size of the shared montage vocabulary (number of known electrode slots).
    embed_dim : int
        Token embedding dimension.
    """

    def __init__(self, n_vocab: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(channel_indices)  # (n_chans, embed_dim)
        # -> (1, 1, n_chans, embed_dim) to broadcast over batch and patches.
        return rearrange(emb, "n_chans embed_dim -> 1 1 n_chans embed_dim")


class _ResidualAdd(nn.Module):
    """Residual connection with optional stochastic depth (``x + drop_path(fn(x))``)."""

    def __init__(self, fn: nn.Module, drop_path: float = 0.0):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.fn(x))


class _TransformerEncoderBlock(nn.Sequential):
    """Pre-norm Transformer encoder block (ViT-style).

    Same layout as the block of :class:`~braindecode.models.EEGConformer`: a
    residual multi-head self-attention sub-block followed by a residual
    feed-forward sub-block, each with pre-LayerNorm. Reuses braindecode's
    :class:`~braindecode.modules.MultiHeadAttention` and
    :class:`~braindecode.modules.FeedForwardBlock`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_prob: float,
        mlp_ratio: int,
        drop_path: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    MultiHeadAttention(embed_dim, num_heads, drop_prob),
                    nn.Dropout(drop_prob),
                ),
                drop_path,
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    FeedForwardBlock(
                        embed_dim,
                        expansion=mlp_ratio,
                        drop_p=drop_prob,
                        activation=activation,
                    ),
                    nn.Dropout(drop_prob),
                ),
                drop_path,
            ),
        )

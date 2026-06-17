# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)
"""STEEGFormer — WIP draft (see #1040).

Port of Yang et al. (2026), https://github.com/LiuyinYang1101/STEEGFormer
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, MultiHeadAttention


class _PatchEmbedEEG(nn.Module):
    """Split each channel into temporal patches and embed them as tokens.

    The input EEG is cut, **per channel**, into contiguous non-overlapping
    temporal patches of ``patch_size`` samples; each patch is linearly
    projected to ``embed_dim``. This is the EEG counterpart of the patch
    embedding of a Vision Transformer: one token per (temporal patch, channel)
    pair.

    Parameters
    ----------
    patch_size : int
        Number of time samples per temporal patch.
    embed_dim : int
        Token embedding dimension.

    Notes
    -----
    ``n_times`` must be an exact multiple of ``patch_size``; the number of
    temporal patches is ``seq = n_times // patch_size``. The (temporal,
    channel) structure is kept in the output so that temporal and channel
    positional embeddings can be added before the tokens are flattened.
    """

    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_chans, n_times) -> (batch, seq, n_chans, patch_size)
        patches = rearrange(x, "b c (seq p) -> b seq c p", p=self.patch_size)
        # -> (batch, seq, n_chans, embed_dim)
        return self.proj(patches)


class _TemporalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding over temporal patches.

    Standard sine/cosine encoding (Vaswani et al., 2017). Position ``0`` is
    reserved for the CLS token, so the ``seq`` temporal patches use positions
    ``1..seq``.

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
        self.register_buffer("pe", pe)

    def cls_token_encoding(self) -> torch.Tensor:
        # Encoding at position 0, shape (embed_dim,).
        return self.pe[0]

    def forward(self, seq: int) -> torch.Tensor:
        if seq + 1 > self.max_len:
            raise ValueError(
                f"Too many temporal patches ({seq}) for max_len={self.max_len}; "
                "increase max_len."
            )
        # Positions 1..seq -> (1, seq, 1, embed_dim) to broadcast over
        # batch and channels.
        return rearrange(self.pe[1 : seq + 1], "seq d -> 1 seq 1 d")


class _ChannelPositionalEmbed(nn.Module):
    """Learned per-channel positional embedding, zero-initialised.

    One learned vector per EEG channel, added to every temporal patch of that
    channel. Zero initialisation means the model starts as if there were no
    channel embedding and learns it from data.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    embed_dim : int
        Token embedding dimension.
    """

    def __init__(self, n_chans: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_chans, embed_dim)
        nn.init.zeros_(self.embedding.weight)
        self.register_buffer("channel_indices", torch.arange(n_chans))

    def forward(self) -> torch.Tensor:
        emb = self.embedding(self.channel_indices)  # (n_chans, embed_dim)
        # -> (1, 1, n_chans, embed_dim) to broadcast over batch and patches.
        return rearrange(emb, "c d -> 1 1 c d")


class _ResidualAdd(nn.Module):
    """Wrap a module in a residual connection (``x + fn(x)``)."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


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
        drop_rate: float,
        mlp_ratio: int,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    MultiHeadAttention(embed_dim, num_heads, drop_rate),
                    nn.Dropout(drop_rate),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    FeedForwardBlock(
                        embed_dim,
                        expansion=mlp_ratio,
                        drop_p=drop_rate,
                        activation=activation,
                    ),
                    nn.Dropout(drop_rate),
                )
            ),
        )


class STEEGFormer(EEGModuleMixin, nn.Module):
    r"""STEEGFormer from Yang et al. (2026) [Yang2026]_.

    :bdg-info:`Attention/Transformer` :bdg-warning:`Foundation / Self-supervised`

    .. rubric:: Architectural Overview

    ViT-based foundation model, pre-trained with a Masked Autoencoder (MAE)
    objective on raw EEG. Segments (<= 6 s @ 128 Hz) are split into temporal
    patches per channel, embedded as tokens, enriched with temporal/channel
    positional embeddings and a CLS token, processed by a Transformer encoder,
    and read out by an average-pooling classification head.

    .. note::
        Work in progress (draft, see #1040). The architecture is implemented
        from scratch with braindecode building blocks. Two simplifications vs
        the reference: the per-channel embedding is a plain
        :class:`~torch.nn.Embedding` over the model's own channels (not the
        shared 145-channel montage vocabulary), and loading the official
        pre-trained (MAE) weights is not supported yet.

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
    mlp_ratio : float
        Hidden-to-embedding ratio of the MLP blocks.
    drop_rate : float
        Dropout rate.
    global_pool : str
        Token aggregation before the head (``"avg"`` or ``"cls"``).

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
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # --- model hyperparameters (defaults: "small" variant) ---
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        global_pool: str = "avg",
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

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.global_pool = global_pool

        # Patch embedding + positional embeddings + CLS token.
        self.patch_embed = _PatchEmbedEEG(patch_size, embed_dim)
        self.temporal_pos = _TemporalPositionalEncoding(embed_dim)
        self.channel_pos = _ChannelPositionalEmbed(self.n_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer encoder (reuses braindecode's attention/FFN blocks).
        self.encoder = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    embed_dim, num_heads, drop_rate, int(mlp_ratio)
                )
                for _ in range(depth)
            ]
        )

        # Classification head.
        self.norm = nn.LayerNorm(embed_dim)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_chans, n_times)
        seq = x.shape[-1] // self.patch_size
        # Crop the tail so n_times is an exact multiple of patch_size
        # (mirrors the non-overlapping patching of the reference).
        x = x[..., : seq * self.patch_size]

        # Tokens + positional embeddings, kept on the (seq, channel) grid.
        tokens = self.patch_embed(x)  # (batch, seq, n_chans, embed_dim)
        tokens = tokens + self.temporal_pos(seq) + self.channel_pos()
        tokens = rearrange(tokens, "b seq c d -> b (seq c) d")

        # Prepend the CLS token (combined with the temporal encoding at pos. 0).
        cls = self.cls_token + self.temporal_pos.cls_token_encoding()
        cls = cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.pos_drop(x)

        x = self.encoder(x)

        # Aggregate tokens, then classify.
        if self.global_pool == "avg":
            x = x[:, 1:].mean(dim=1)  # mean over tokens, excluding CLS
        else:  # "cls"
            x = x[:, 0]
        x = self.norm(x)
        return self.final_layer(x)

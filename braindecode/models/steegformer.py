# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)
"""STEEGFormer — ViT-MAE EEG foundation model.

Port of Yang et al. (2026), https://github.com/LiuyinYang1101/STEEGFormer
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict

import torch
from einops import rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, MultiHeadAttention

# Shared montage vocabulary of the official ST-EEGFormer checkpoints: the
# learned channel embedding has one slot per entry, in this order (the slot
# index of an electrode is its position in this list). Taken from the authors'
# ``channels_mapping`` (``pretrain/senloc_file/sen_chan_idx.pkl``,
# ``LiuyinYang1101/STEEGFormer``, MIT). Used by small/base/large (vocab 145);
# the HBN ``largeV2`` model uses a different, larger vocabulary.
STEEGFORMER_CHANNEL_ORDER: list[str] = [
    "C1",
    "Pz",
    "C4",
    "F6",
    "FTT8h",
    "Oz",
    "Fp1",
    "FCC5h",
    "TPP8h",
    "CPP6h",
    "C2",
    "F4",
    "OI2h",
    "AF4",
    "FCz",
    "CCP6h",
    "TP8",
    "POO10h",
    "FC1",
    "FC6",
    "C5",
    "P8",
    "FT8",
    "P6",
    "P9",
    "Fz",
    "AFF1",
    "TPP10h",
    "AFF2",
    "P10",
    "CPP2h",
    "M1",
    "FCC6h",
    "FTT7h",
    "FC2",
    "PPO2",
    "AFp3h",
    "AF7",
    "PO10",
    "AF8",
    "CPP1h",
    "P7",
    "F1",
    "AFp4h",
    "PO9",
    "FT9",
    "CP2",
    "Iz",
    "FCC1h",
    "FC5",
    "T5",
    "CP5",
    "CP6",
    "FFC5h",
    "F2",
    "M2",
    "POO9h",
    "AFF5h",
    "PO4",
    "POO3h",
    "Fp2",
    "T3",
    "CP4",
    "POz",
    "TTP7h",
    "T7",
    "A2",
    "CCP4h",
    "T8",
    "PPO10h",
    "FC3",
    "F3",
    "F5",
    "A1",
    "P3",
    "FC4",
    "FCC2h",
    "FFC6h",
    "FFT8h",
    "CCP2h",
    "CPP4h",
    "T6",
    "FTT9h",
    "PPO6h",
    "CP3",
    "CP1",
    "AF3",
    "FT10",
    "OI1h",
    "TPP9h",
    "P5",
    "I2",
    "CCP1h",
    "T4",
    "CCP3h",
    "O1",
    "PO5",
    "PPO9h",
    "PPO5h",
    "P1",
    "AFz",
    "PO6",
    "PO3",
    "O2",
    "CPP5h",
    "FFC1h",
    "FCC4h",
    "FFT7h",
    "FFC2h",
    "FFC4h",
    "Cz",
    "TP7",
    "Fpz",
    "FTT10h",
    "PO7",
    "CPP3h",
    "P4",
    "P2",
    "F8",
    "CPz",
    "FCC3h",
    "FFC3h",
    "FT7",
    "I1",
    "TTP8h",
    "AFF6h",
    "CCP5h",
    "C6",
    "PPO1",
    "PO8",
    "C3",
    "POO4h",
    "TPP7h",
    "F7",
    "T9",
    "TP9",
    "T10",
    "TP10",
    "POO1",
    "POO2",
    "PPO1h",
    "PPO2h",
]
_STEEGFORMER_CHANNEL_INDEX = {
    name.upper(): i for i, name in enumerate(STEEGFORMER_CHANNEL_ORDER)
}


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
        drop_prob: float,
        mlp_ratio: int,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    MultiHeadAttention(embed_dim, num_heads, drop_prob),
                    nn.Dropout(drop_prob),
                )
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
                )
            ),
        )


class STEEGFormer(EEGModuleMixin, nn.Module):
    r"""STEEGFormer from Yang et al. (2026) [Yang2026]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. figure:: https://raw.githubusercontent.com/LiuyinYang1101/STEEGFormer/main/assets/graphic_overview.png
        :align: center
        :width: 1000px

        ST-EEGFormer architecture, reproduced from the official repository
        (``LiuyinYang1101/STEEGFormer``).

    .. rubric:: Architectural Overview

    ViT-based EEG foundation model, pre-trained with a Masked Autoencoder
    (MAE) objective on raw EEG. The pipeline is, end to end:

    - **Patch + token embedding.** Each channel is cut into non-overlapping
      temporal patches of ``patch_size`` samples; every patch is linearly
      projected to a ``embed_dim`` token. A segment of ``C`` channels and
      ``seq = n_times // patch_size`` patches therefore yields ``C * seq``
      tokens (one per (channel, time-patch) pair).
    - **Positional information.** A *fixed sinusoidal* encoding marks the
      temporal position of each patch, and a *learned* embedding, drawn from a
      shared **montage vocabulary** of standard electrode positions, marks the
      channel — so the same electrode shares its embedding across datasets with
      different channel sets. A learned CLS token is prepended.
    - **Transformer encoder.** ``depth`` pre-norm ViT blocks (multi-head
      self-attention + MLP, reusing braindecode's
      :class:`~braindecode.modules.MultiHeadAttention` and
      :class:`~braindecode.modules.FeedForwardBlock`).
    - **Read-out.** ``"avg"`` mean-pools the patch tokens (CLS excluded);
      ``"cls"`` layer-normalises the sequence and takes the CLS token. A linear
      head maps to ``n_outputs``.

    .. rubric:: Variants

    The three released variants differ only in width/depth (``patch_size=16``,
    ``mlp_ratio=4`` throughout):

    .. list-table::
       :header-rows: 1

       * - Variant
         - ``embed_dim``
         - ``depth``
         - ``num_heads``
       * - small
         - 512
         - 8
         - 8
       * - base
         - 768
         - 12
         - 12
       * - large
         - 1024
         - 24
         - 16

    .. rubric:: Pre-trained weights

    The official MAE checkpoints (GitHub releases of
    ``LiuyinYang1101/STEEGFormer``) can be loaded directly:
    :meth:`load_state_dict` detects the upstream ``timm`` format, keeps the
    encoder, and remaps it to this module (see that method). The pre-trained
    checkpoint carries **no classification head** (it is re-initialised), so it
    provides a feature extractor to be fine-tuned downstream. The montage
    vocabulary size depends on the checkpoint (145 for small/base/large, 256
    for the HBN ``largeV2`` model): set ``n_chans_pos`` accordingly.

    .. note::
        Numerical equivalence of the encoder features with the reference
        implementation has been verified on the released checkpoints. The
        channel-to-vocabulary mapping is resolved
        from the electrode names in ``chs_info`` (looked up in
        :data:`STEEGFORMER_CHANNEL_ORDER`, the BENDR/LaBraM convention); when
        ``chs_info`` is absent or a name is unknown, it falls back to the
        identity mapping (channel ``i`` -> slot ``i``) with a warning. Pass
        ``chan_pos_idx`` to override the mapping explicitly.

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
    drop_prob : float
        Dropout rate.
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
        drop_prob: float = 0.0,
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

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_prob = drop_prob
        self.activation = activation
        self.global_pool = global_pool
        self.n_chans_pos = n_chans_pos

        # Map each input channel to its slot in the shared montage vocabulary.
        # Priority: an explicit ``chan_pos_idx`` wins; otherwise the mapping is
        # resolved from the electrode names in ``chs_info`` (the BENDR/LaBraM
        # convention); if neither is usable, fall back to the identity mapping
        # (channel i -> slot i), as in the reference example.
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
        self.register_buffer("channel_indices", chan_pos_idx)

        # Patch embedding + positional embeddings + CLS token.
        self.patch_embed = _PatchEmbedEEG(patch_size, embed_dim)
        self.temporal_pos = _TemporalPositionalEncoding(embed_dim)
        self.channel_pos = _ChannelPositionalEmbed(n_chans_pos, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(drop_prob)

        # Transformer encoder (reuses braindecode's attention/FFN blocks).
        self.encoder = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    embed_dim, num_heads, drop_prob, int(mlp_ratio), activation
                )
                for _ in range(depth)
            ]
        )

        # Classification head.
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    def _chan_pos_idx_from_chs_info(self) -> torch.Tensor:
        """Resolve montage-vocabulary slots from the ``chs_info`` electrode names.

        Looks each electrode name up in :data:`STEEGFORMER_CHANNEL_ORDER`
        (case-insensitive), the BENDR/LaBraM convention. Falls back to the
        identity mapping -- and warns -- when ``chs_info`` is absent or carries
        names outside the montage vocabulary.
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

    def reset_head(self, n_outputs):
        """Replace the linear classification head for a new ``n_outputs``.

        Called by :meth:`from_pretrained` when the requested number of outputs
        differs from the pre-trained checkpoint (whose head is discarded).
        """
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load weights, remapping the official checkpoint if needed.

        The official ST-EEGFormer checkpoints (GitHub releases of
        ``LiuyinYang1101/STEEGFormer``) are MAE pre-training checkpoints built
        on ``timm``: the encoder weights are wrapped under a top-level
        ``"model"`` key, named in the ``timm`` convention, and accompanied by
        decoder weights used only for the pre-training objective. This override
        detects that format and converts it to this module's attribute names:

        1. unwrap the top-level ``"model"`` entry;
        2. drop the MAE decoder (``decoder_*``, ``dec_*``, ``mask_token``) and
           the downstream-specific keys absent from the encoder (``pos_embed``,
           ``fc_norm``, ``head``);
        3. rename the ``timm`` blocks to this module's ``encoder`` blocks and
           **split the fused ``attn.qkv``** into separate ``queries``/``keys``/
           ``values`` (the braindecode attention keeps them apart).

        A checkpoint already in this module's format is loaded unchanged.
        """
        if isinstance(state_dict.get("model"), dict):
            state_dict = state_dict["model"]
        is_official = any("attn.qkv" in k for k in state_dict)
        if not is_official:
            return super().load_state_dict(state_dict, *args, **kwargs)

        e = self.embed_dim
        # ``enc_temporal_emd.pe`` is a fixed sinusoidal buffer regenerated at
        # init (and shaped differently), so it is regenerated, not loaded.
        drop_prefixes = ("decoder_", "dec_", "mask_token", "enc_temporal_emd")
        drop_exact = {
            "pos_embed",
            "fc_norm.weight",
            "fc_norm.bias",
            "head.weight",
            "head.bias",
        }
        remapped: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(drop_prefixes) or key in drop_exact:
                continue
            if key == "enc_channel_emd.channel_transformation.weight":
                remapped["channel_pos.embedding.weight"] = value
            elif key.startswith("blocks."):
                _, idx, rest = key.split(".", 2)
                d = f"encoder.{idx}."
                if rest.startswith("norm1."):
                    remapped[d + "0.fn.0." + rest[6:]] = value
                elif rest in ("attn.qkv.weight", "attn.qkv.bias"):
                    suffix = rest.rsplit(".", 1)[1]  # "weight" or "bias"
                    remapped[d + "0.fn.1.queries." + suffix] = value[:e]
                    remapped[d + "0.fn.1.keys." + suffix] = value[e : 2 * e]
                    remapped[d + "0.fn.1.values." + suffix] = value[2 * e :]
                elif rest.startswith("attn.proj."):
                    remapped[d + "0.fn.1.projection." + rest[10:]] = value
                elif rest.startswith("norm2."):
                    remapped[d + "1.fn.0." + rest[6:]] = value
                elif rest.startswith("mlp.fc1."):
                    remapped[d + "1.fn.1.0." + rest[8:]] = value
                elif rest.startswith("mlp.fc2."):
                    remapped[d + "1.fn.1.3." + rest[8:]] = value
            else:
                # cls_token, patch_embed.proj.*, norm.* share the same name.
                remapped[key] = value

        # The official encoder has no classification head and the temporal
        # encoding / channel index are non-persistent buffers regenerated at
        # init, so a non-strict load is required.
        kwargs.setdefault("strict", False)
        return super().load_state_dict(remapped, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an EEG batch into class logits.

        Parameters
        ----------
        x : torch.Tensor
            EEG input of shape ``(batch, n_chans, n_times)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, n_outputs)``.
        """
        seq = x.shape[-1] // self.patch_size
        # Crop the tail so n_times is an exact multiple of patch_size
        # (mirrors the non-overlapping patching of the reference).
        x = x[..., : seq * self.patch_size]

        # Tokens + positional embeddings, kept on the (seq, channel) grid.
        tokens = self.patch_embed(x)  # (batch, seq, n_chans, embed_dim)
        tokens = (
            tokens + self.temporal_pos(seq) + self.channel_pos(self.channel_indices)
        )
        tokens = rearrange(tokens, "b seq c d -> b (seq c) d")

        # Prepend the CLS token (combined with the temporal encoding at pos. 0).
        cls = self.cls_token + self.temporal_pos.cls_token_encoding()
        cls = cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.pos_drop(x)

        x = self.encoder(x)

        # Aggregate tokens, then classify. Mirrors the reference: average
        # pooling discards the CLS token and applies no final norm, whereas the
        # CLS read-out normalises the sequence first and then takes the CLS
        # token.
        if self.global_pool == "avg":
            x = x[:, 1:].mean(dim=1)  # mean over tokens, excluding CLS
        else:  # "cls"
            x = self.norm(x)[:, 0]
        return self.final_layer(x)

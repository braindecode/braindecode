# Authors: Rukuang Huang, SungJun Cho and the OHBA Analysis Group (original implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/OHBA-analysis/MEG-GPT (MIT) and
# https://github.com/OHBA-analysis/EphysTokenizer (MIT).
"""MEG-GPT: a data-driven tokenizer and a time-attention GPT for source-space MEG."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Union

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.functional import pad_or_crop_to_patches
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock


class MEGGPT(EEGModuleMixin, nn.Module, license="mit"):
    r"""MEG-GPT from Huang et al. (2026) [huang2026meggpt]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-secondary:`Recurrent`

    .. rubric:: Architecture Overview

    1. The EphysTokenizer encoder [ephystokenizer]_ turns each channel into a
       stream of discrete tokens, one per sample: a GRU runs over the channel,
       a linear layer and a layer norm give one logit per codebook entry, and
       the argmax is remapped to a token id through the released vocabulary.
    2. The token stream is cut into segments of ``n_patches[0] * patch_lens[0]``
       tokens (100, i.e. 0.4 s at 250 Hz). Token, time and channel embeddings
       are summed.
    3. A stack of causal time-attention layers runs on every channel. In layer
       ``i`` the keys and values are pooled into ``n_patches[i]`` patches of
       ``patch_lens[i]`` steps and the output keeps the last steps only, so the
       sequence shrinks to ``n_last_steps`` outputs per segment.
    4. The outputs are averaged over time and segments into
       ``(batch, n_chans, embed_dim)`` features (Section 2.4 of the paper),
       flattened and classified by a linear layer.

    .. rubric:: Macro Components

    ``MEGGPT.tokenizer``
        **Operations:** per-channel GRU, linear layer, layer norm, argmax and
        vocabulary lookup.

        **Role:** discretise each sample of each channel. The argmax has no
        gradient, so the tokenizer is frozen (its parameters do not require
        gradients).

    ``MEGGPT.layers``
        **Operations:** pre-norm causal attention over time with patched keys
        and values (a linear layer mixes the heads and steps of a patch), a
        banded mask in layers with a ``patch_band``, and a feed-forward block.

        **Role:** model the temporal dependencies of each brain region.

    ``MEGGPT.final_layer``
        **Operations:** flatten and linear projection.

        **Role:** map the time-averaged features to class logits.

    .. rubric:: Differences from the released code

    - The released model also adds a learned embedding of the Cam-CAN session
      (621 sessions). It is specific to the pre-training data, and the paper
      discards it for downstream use (Section 2.5.4), so this class has none:
      the transplanted checkpoint drops those weights, and the forward equals
      the released model called without session labels.
    - The next-token prediction head is replaced by ``final_layer``.
    - Only the parts of the released configuration are ported: time attention
      with patching. The reference code's optional channel and cross attention,
      unpatched keys, rotary or sinusoidal positions and batch/group norms are
      off in the released configuration and not implemented.
    - The tokenizer encodes the whole input window in one GRU pass; the
      reference tokenizes recordings in 200-sample windows, which gives the
      same tokens for windows of up to 200 samples.

    .. rubric:: Input

    Source-space MEG: 52 parcels of the Glasser 52 parcellation at 250 Hz,
    each parcel z-scored over the recording (``tokenize_session(standardize=True)``
    in the reference). The model does not normalise its input. A window that is
    not a multiple of the segment length is padded or cropped as set by
    ``on_non_divisible``.

    .. rubric:: Pre-trained weights

    The released checkpoints (MEG-GPT ``models/meg-gpt/model_state.pt`` and the
    tokenizer ``models/tokenizer/{model_state.pt,vocab.pkl}`` of the original
    repository) map onto this class key by key, without the session embedding,
    the next-token head and the tokenizer's signal decoder; every weight but
    the classification head is then pre-trained.

    .. versionadded:: 1.9

    Parameters
    ----------
    n_tokens : int, default=66
        Size of the GPT vocabulary (65 used codebook entries plus id 0 for the
        unused ones in the released vocabulary).
    n_codes : int, default=128
        Codebook size of the tokenizer.
    rnn_hidden_size : int, default=128
        Hidden size of the tokenizer GRU.
    embed_dim : int, default=256
        Embedding and model dimension.
    num_heads : int, default=4
        Number of attention heads.
    feedforward_dim : int, default=1024
        Hidden width of the feed-forward blocks.
    n_patches : sequence of int, default=(100, 25, 20, 10, 8)
        Number of key/value patches in each layer.
    patch_lens : sequence of int, default=(1, 4, 4, 8, 8)
        Steps per key/value patch in each layer. Layer ``i`` reads
        ``n_patches[i] * patch_lens[i]`` steps and outputs as many steps as the
        next layer reads; ``n_patches[0] * patch_lens[0]`` is the segment length.
    n_last_steps : int, default=16
        Number of steps the last layer outputs per segment.
    patch_band : sequence of int or None, default=(8, None, None, None, None)
        Half-width, in patches, of the banded attention of each layer; ``None``
        for no band.
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward blocks.
    drop_prob : float, default=0.2
        Dropout after the embeddings, the attention and the feed-forward blocks.
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        How a window that is not a multiple of the segment length is handled:
        right zero-padded, cropped, or rejected.
    return_features : bool, default=False
        Whether ``forward`` returns the feature dictionary by default.

    Input shape
    -----------
    ``(batch, n_chans, n_times)``.

    Output shape
    ------------
    By default ``(batch, n_outputs)`` logits. With ``return_features=True``,
    ``{"features": (batch, n_chans, embed_dim), "cls_token": None}``.

    References
    ----------
    .. [huang2026meggpt] Huang, R., Cho, S., Gohil, C., Parker Jones, O., &
       Woolrich, M. (2026). MEG-GPT: A transformer-based foundation model for
       magnetoencephalography data. Imaging Neuroscience, 4, IMAG.a.1301.
       https://doi.org/10.1162/imag.a.1301 (preprint arXiv:2510.18080).
    .. [ephystokenizer] EphysTokenizer, released implementation:
       https://github.com/OHBA-analysis/EphysTokenizer

    `License <https://github.com/OHBA-analysis/MEG-GPT/blob/main/LICENSE>`_
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # model-specific parameters
        n_tokens: int = 66,
        n_codes: int = 128,
        rnn_hidden_size: int = 128,
        embed_dim: int = 256,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        n_patches: Sequence[int] = (100, 25, 20, 10, 8),
        patch_lens: Sequence[int] = (1, 4, 4, 8, 8),
        n_last_steps: int = 16,
        patch_band: Sequence[Optional[int]] = (8, None, None, None, None),
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.2,
        on_non_divisible: str = "pad",
        return_features: bool = False,
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
        _check_on_non_divisible(on_non_divisible)
        self.segment_len = n_patches[0] * patch_lens[0]
        self.embed_dim = embed_dim
        self.on_non_divisible = on_non_divisible
        self.return_features = return_features

        self.tokenizer = _EphysTokenizer(n_codes, n_tokens, rnn_hidden_size)
        self.to_segments = Rearrange("b c (s t) -> (b s) t c", t=self.segment_len)
        self.token_embedding = nn.Embedding(n_tokens, embed_dim)
        self.time_embedding = nn.Parameter(torch.empty(self.segment_len, embed_dim))
        self.channel_embedding = nn.Parameter(torch.empty(self.n_chans, embed_dim))
        self.drop = nn.Dropout(drop_prob)
        # Each layer outputs as many steps as the next one reads.
        n_outs = [n * p for n, p in zip(n_patches[1:], patch_lens[1:])]
        self.layers = nn.ModuleList(
            _DecoderLayer(
                embed_dim,
                num_heads,
                feedforward_dim,
                n,
                p,
                n_out,
                band,
                activation,
                drop_prob,
            )
            for n, p, n_out, band in zip(
                n_patches, patch_lens, n_outs + [n_last_steps], patch_band
            )
        )
        self.final_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(self.n_chans * embed_dim, self.n_outputs)
        )

        # Reference initialisation (meg_gpt/optim/initializer.py).
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.05, 0.05)
        nn.init.xavier_uniform_(self.time_embedding)
        nn.init.xavier_uniform_(self.channel_embedding)

    def forward(
        self, x: Tensor, return_features: Optional[bool] = None
    ) -> Union[Tensor, Dict[str, Optional[Tensor]]]:
        batch_size = x.shape[0]
        x = pad_or_crop_to_patches(x, self.segment_len, self.on_non_divisible)
        tokens = self.to_segments(self.tokenizer(x))  # (batch * segments, time, chans)
        h = self.token_embedding(tokens)
        h = h + self.time_embedding[:, None]
        h = h + self.channel_embedding
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = h.reshape(batch_size, -1, h.shape[-2], h.shape[-1])
        features = h.mean(dim=1)
        if return_features is None:
            return_features = self.return_features
        if return_features:
            out: Dict[str, Optional[Tensor]] = {"features": features, "cls_token": None}
            return out
        return self.final_layer(features)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        head = self.final_layer[1]
        self.final_layer[1] = nn.Linear(
            head.in_features,
            n_outputs,
            device=head.weight.device,
            dtype=head.weight.dtype,
        )


class _EphysTokenizer(nn.Module):
    """EphysTokenizer encoder: ``(batch, chans, time)`` floats to token ids.

    A GRU runs over each channel separately; the argmax of the normalised
    logits is mapped through ``vocab`` (codebook entry to token id, frequency
    ranked; unused entries map to 0 in the released vocabulary).
    """

    def __init__(self, n_codes: int, n_tokens: int, rnn_hidden_size: int):
        super().__init__()
        self.rnn = nn.GRU(1, rnn_hidden_size, batch_first=True)
        self.proj = nn.Linear(rnn_hidden_size, n_codes)
        self.norm = nn.LayerNorm(n_codes)
        # Placeholder until the released vocabulary is loaded.
        self.register_buffer("vocab", torch.arange(n_codes) % n_tokens)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_chans, n_times = x.shape
        h, _ = self.rnn(x.reshape(batch_size * n_chans, n_times, 1))
        # Released scaling; kept so that ties break as in the reference.
        logits = self.norm(self.proj(h)) / 0.1
        codes = logits.argmax(dim=-1)
        return self.vocab[codes].reshape(batch_size, n_chans, n_times)


class _DecoderLayer(nn.Module):
    """Pre-norm patched time attention and feed-forward block.

    The attention outputs the last ``n_out`` steps; the residual keeps the
    same steps of the input.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        n_patches: int,
        patch_len: int,
        n_out: int,
        patch_band: Optional[int],
        activation: type[nn.Module],
        drop_prob: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _PatchedTimeAttention(
            embed_dim, num_heads, n_patches, patch_len, n_out, patch_band
        )
        self.drop = nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(
            embed_dim,
            1,
            0.0,
            activation,
            hidden_features=feedforward_dim,
            output_drop_p=drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.attn(self.norm1(x))
        h = self.drop(h)
        x = h + x[:, -h.shape[1] :]
        return x + self.mlp(self.norm2(x))


class _PatchedTimeAttention(nn.Module):
    """Causal attention over time, per channel, on ``(batch, time, chans, dim)``.

    Keys and values are pooled into patches by one linear layer over the
    ``num_heads * patch_len`` (head, step) pairs of a patch, the layout of the
    released weights, which no shared attention module has. Queries are the
    last ``n_out`` steps. A query that sees no patch outputs zeros, as in the
    reference.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_patches: int,
        patch_len: int,
        n_out: int,
        patch_band: Optional[int],
    ):
        super().__init__()
        self.n_out = n_out
        self.scale = math.sqrt(embed_dim // num_heads)
        self.to_patches = Rearrange(
            "b (p l) c (h k) -> b p c k (h l)", h=num_heads, l=patch_len
        )
        self.from_patches = Rearrange("b p c k h -> b p c (h k)")
        pool = num_heads * patch_len
        self.patch_k = nn.Linear(pool, num_heads) if patch_len > 1 else nn.Identity()
        self.patch_v = nn.Linear(pool, num_heads) if patch_len > 1 else nn.Identity()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.split_heads = Rearrange("b t c (h k) -> b h c t k", h=num_heads)
        self.split_heads_t = Rearrange("b t c (h k) -> b h c k t", h=num_heads)
        self.merge_heads = Rearrange("b h c t k -> b t c (h k)")

        query = torch.arange(n_out)[:, None]
        patch = torch.arange(n_patches)
        # A patch is visible once its last step is at or before the query.
        visible = (patch + 1 - n_patches) * patch_len + n_out - 1 <= query
        if patch_band is not None:
            visible &= (patch - query // patch_len).abs() <= patch_band
        self.register_buffer("mask", ~visible, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        k = self.k_proj(self.from_patches(self.patch_k(self.to_patches(x))))
        v = self.v_proj(self.from_patches(self.patch_v(self.to_patches(x))))
        q = self.q_proj(x[:, -self.n_out :])
        q = self.split_heads(q)
        k = self.split_heads_t(k)
        v = self.split_heads(v)
        scores = q @ k / self.scale
        scores = scores.masked_fill(self.mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        weights = weights.nan_to_num(nan=0.0)
        out = self.merge_heads(weights @ v)
        return self.out_proj(out)

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
"""MVPFormer: a foundation model with multi-variate parallel attention.

Reimplementation of MVPFormer (Carzaniga et al., 2026), "A foundation model
with multi-variate parallel attention to generate neuronal activity". The
architecture is transcribed from the authors' reference implementation
(Copyright IBM Corp. 2024-2025), released under the Apache License,
Version 2.0; this file is therefore distributed under Apache-2.0
(https://www.apache.org/licenses/LICENSE-2.0). The braindecode reimplementation
is pure-PyTorch and CPU-runnable (no Triton / DeepSpeed).

Original Authors: Carzaniga et al., IBM Corp.
Braindecode Adaptation: Bruno Aristimunha
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from torch import nn

from braindecode.functional import daubechies_filters, wavelet_decomposition
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import PatchTokenizer


class MVPFormer(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""MVPFormer from Carzaniga et al. (2026) [Carzaniga2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://ar5iv.labs.arxiv.org/html/2506.20354/assets/x2.png
        :align: center
        :alt: MVPFormer model overview
        :width: 1000px

        MVPFormer model overview from [Carzaniga2026]_.

    .. figure:: https://ar5iv.labs.arxiv.org/html/2506.20354/assets/x1.png
        :align: center
        :alt: Multi-variate parallel attention mechanism
        :width: 1000px

        Multi-variate parallel attention (MVPA) from [Carzaniga2026]_.

    .. versionadded:: 1.6.1

    A foundation model with multi-variate parallel attention for heterogeneous
    multi-variate iEEG. The raw signal is tokenised channel-wise into continuous
    wavelet embeddings and processed by a decoder-only (Llama2-style) transformer
    whose self-attention is decomposed into content, time-relative and
    channel-relative terms over a ``(segment, channel)`` token grid
    [Carzaniga2026]_.

    .. rubric:: Architecture Overview

    MVPFormer combines three ideas to model heterogeneous multi-variate iEEG
    [Carzaniga2026]_:

    1. **Continuous wavelet tokenization.** A db4 wavelet encoder maps each
       fixed-length signal segment (per channel) to a continuous embedding,
       rather than a discrete vocabulary, preserving fine signal structure.
    2. **Multi-variate parallel attention (MVPA).** Self-attention over the 2D
       ``(segment, channel)`` token grid is decomposed into three additive terms
       -- content, time-relative and channel-relative -- so the temporal and
       spatial axes are modelled jointly at the attention level.
    3. **Llama2-style decoder.** Parallel attention and MLP blocks, grouped-query
       attention and causal masking in time, trained to predict the next-in-time
       embedding (generative pre-training).

    .. rubric:: Macro Components

    ``MVPFormer.patch_tokenizer`` + ``MVPFormer.patch_embed`` (Wavelet encoder)
        **Operations.** :class:`~braindecode.modules.PatchTokenizer` splits the
        raw signal :math:`\mathbf{x} \in \mathbb{R}^{C \times T}` into
        non-overlapping segments of ``segment_len`` samples. Each segment passes
        independently through a fixed db4 discrete wavelet decomposition, is
        RMS-normalised, and is linearly projected to a continuous embedding
        :math:`\mathbf{e}_{c,t} \in \mathbb{R}^{d}`
        (:class:`~braindecode.models.mvpformer._WaveletPatchEmbed`). The wavelet
        transform has no learnable parameters; only the norm and projection are
        trained.
        **Role.** Turns the heterogeneous raw signal into a ``(segment, channel)``
        grid of continuous tokens whose frequency content is exposed by the
        wavelet sub-bands.

    ``MVPFormer.blocks`` (MVPA decoder stack)
        **Operations.** A stack of
        :class:`~braindecode.models.mvpformer._MVPABlock` modules in the parallel
        attention/MLP configuration :math:`\mathbf{o} = \mathbf{o} +
        \text{attn}(\text{ln}(\mathbf{o})) + \text{mlp}(\text{ln}(\mathbf{o}))`
        (Megatron-LM style) with RMSNorm and a SwiGLU MLP. Each block computes
        multi-variate parallel attention
        (:class:`~braindecode.models.mvpformer._MVPAttention`) between tokens at
        grid positions :math:`(c,t)` and :math:`(c',t')` as a sum of three terms:

        .. math::
            \mathbf{a}^{\text{MVPA}}_{c,t,c',t'}
            &= \mathbf{x}_{c,t}^\top W_q^\top W_{k_e}\,\mathbf{x}_{c',t'}
               + \mathbf{u}^\top W_{k_e}\,\mathbf{x}_{c',t'}
               && \text{(content)} \\
            &\;+ \mathbf{x}_{c,t}^\top W_q^\top W_{k_t}\,\mathcal{T}_{t-t'}
               + \mathbf{v}^\top W_{k_t}\,\mathcal{T}_{t-t'}
               && \text{(time-relative)} \\
            &\;+ \mathbf{x}_{c,t}^\top W_q^\top W_{k_c}\,\mathcal{C}_{c-c'}
               + \mathbf{w}^\top W_{k_c}\,\mathcal{C}_{c-c'}
               && \text{(channel-relative)}

        where :math:`\mathcal{T}` and :math:`\mathcal{C}` are learnable time and
        channel positional codebooks and :math:`\mathbf{u}, \mathbf{v},
        \mathbf{w}` are learnable bias terms. Attention is causal in time and
        uses grouped-query attention.
        **Role.** Models content, temporal and spatial dependencies jointly,
        producing one output embedding per input segment.

    ``MVPFormer.final_layer`` (Classification head)
        **Operations.** The last time-segment of the decoder output is pooled
        over channels (``pooling="mean"`` or ``"concat"``) and passed through a
        linear layer.
        **Role.** Maps the pooled representation to ``n_outputs`` class logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** Time is the first grid axis. The time-relative attention term
      encodes the signed distance :math:`t-t'` between segments (shared across
      channels) in Transformer-XL fashion, and attention is causally masked so a
      segment only attends to the past. Content attention is restricted to a
      local window of ``local_window`` segments while the relative term still
      spans the full context.
    - **Spatial:** Channels are the second grid axis. The channel-relative
      attention term encodes the relative distance :math:`c-c'` between channels
      (shared across time); being relative, it is montage-agnostic and can
      recover a connectivity map from random initialisation, independent of the
      absolute electrode layout.
    - **Spectral:** Frequency content is captured at tokenization: the db4
      discrete wavelet decomposition of each segment yields multi-resolution
      sub-band coefficients before projection, so spectral structure is embedded
      in every token rather than learned by explicit band-pass filters.

    .. rubric:: Additional Mechanisms

    - **Grouped-query attention.** ``n_head_kv`` key/value heads are shared across
      ``n_heads`` query heads to reduce memory and compute.
    - **Relative terms and content window.** The time- and channel-relative
      terms are quadratic in one axis and constant in the other (computed once
      and broadcast), and a relative-shift trick yields all relative offsets in
      one pass. The local content window bounds each query's *effective*
      context, but this reference implementation still materialises the full
      ``(segment*channel)``-squared content-attention matrix before masking; the
      Triton ``FlashMVPA`` kernel that avoids that cost is not ported, so memory
      scales quadratically with the total number of tokens.
    - **Generative pre-training (reference only).** Upstream, MVPFormer is
      pre-trained to predict the next-in-time embedding with a contrastive loss
      and fine-tuned with LoRA. This braindecode implementation provides the
      architecture; the Triton ``FlashMVPA`` kernel and the contrastive
      pre-training loop are not included.

    Parameters
    ----------
    segment_len : int
        Length of each temporal segment in samples.
    d_model : int
        Token embedding dimension.
    n_layers : int
        Number of decoder blocks.
    n_heads : int
        Number of attention (query) heads.
    n_head_kv : int
        Number of key/value heads (grouped-query attention).
    d_inner : int
        Hidden dimension of the SwiGLU MLP.
    local_window : int
        Size, in segments, of the causal local content-attention lookback window.
    global_att : bool
        Whether to use the global content-attention term.
    max_segments : int
        Size of the segment (time) embedding table; must be >= number of
        segments produced from the input.
    max_channels : int
        Size of the channel embedding table; must be >= ``n_chans``.
    drop_prob : float
        Dropout rate (embedding, attention and residual).
    activation : type[nn.Module]
        Activation class used in the SwiGLU MLP (default :class:`torch.nn.SiLU`).
    pooling : {"mean", "concat"}
        How to pool the last segment over channels before the head. ``"mean"``
        (default, as used for seizure detection) is montage-agnostic;
        ``"concat"`` flattens channels and ties the head to ``n_chans``.

    Notes
    -----
    The defaults are a small, CI-friendly configuration. The published model
    sizes are **MVPFormer-S** (75M, ``d_model=768, n_layers=12, n_heads=12,
    n_head_kv=4, d_inner=1728``) and **MVPFormer-M** (1.2B, ``d_model=2048,
    n_layers=24, n_heads=16, n_head_kv=8, d_inner=5632``); both use 5-second
    segments at 512 Hz, i.e. ``segment_len=2560``.

    References
    ----------
    .. [Carzaniga2026] Carzaniga, F. S., Hersche, M., Sebastian, A.,
       Schindler, K., & Rahimi, A. (2026). A foundation model with
       multi-variate parallel attention to generate neuronal activity. In
       The Fourteenth International Conference on Learning Representations.
       https://openreview.net/forum?id=5M1YOW3bRq
    """

    def __init__(
        self,
        # braindecode signal parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # model-specific parameters
        *,
        segment_len: int = 2560,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_head_kv: int = 4,
        d_inner: int = 512,
        local_window: int = 10,
        global_att: bool = True,
        max_segments: int = 110,
        max_channels: int = 128,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.SiLU,
        pooling: str = "mean",
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ):
        if not isinstance(segment_len, int) or segment_len < 1:
            raise ValueError(
                f"segment_len must be a positive integer number of samples, "
                f"got {segment_len!r}."
            )

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            channel_strategy=channel_strategy,
            channel_strategy_kwargs=channel_strategy_kwargs,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if pooling not in ("mean", "concat"):
            raise ValueError(f"pooling must be 'mean' or 'concat', got {pooling!r}.")

        self.segment_len = segment_len
        self.d_model = d_model
        self.pooling = pooling
        # Ceil division because PatchTokenizer pads the last partial segment.
        self.n_segments = -(-self.n_times // self.segment_len)
        if self.n_segments > max_segments:
            raise ValueError(
                f"{self.n_segments} segments exceed max_segments ({max_segments}); "
                "increase max_segments or reduce n_times / segment_len."
            )
        if self.n_chans > max_channels:
            raise ValueError(
                f"n_chans ({self.n_chans}) exceeds max_channels ({max_channels})."
            )

        self.patch_tokenizer = PatchTokenizer(
            patch_size=self.segment_len, n_times=self.n_times, learnable=False
        )
        self.patch_embed = _WaveletPatchEmbed(self.segment_len, d_model)
        self.positional_embedding = nn.Embedding(max_segments, d_model)
        self.channel_embedding = nn.Embedding(max_channels, d_model)
        self.drop = nn.Dropout(drop_prob)
        self.blocks = nn.ModuleList(
            [
                _MVPABlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_head_kv=n_head_kv,
                    d_inner=d_inner,
                    global_att=global_att,
                    local_window=local_window,
                    attn_drop=drop_prob,
                    resid_drop=drop_prob,
                    layer_idx=i,
                    activation=activation,
                )
                for i in range(n_layers)
            ]
        )
        self.ln_f = nn.RMSNorm(d_model, eps=1e-5)
        head_in = d_model if pooling == "mean" else self.n_chans * d_model
        self.final_layer = nn.Linear(head_in, self.n_outputs, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Normal init (std=0.02), matching the reference ``initializer_range``."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def reset_head(self, n_outputs):
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)
        head_in = (
            self.d_model if self.pooling == "mean" else self.n_chans * self.d_model
        )
        self.final_layer = nn.Linear(head_in, n_outputs, bias=False)
        # Match fresh construction (std=0.02), not the default Linear init.
        self._init_weights(self.final_layer)

    def forward(
        self, x, return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        # x: (batch, n_chans, n_times)
        patches = self.patch_tokenizer(x)  # (batch, channel, segment, segment_len)
        embeds = self.patch_embed(patches)  # (batch, channel, segment, d_model)
        embeds = embeds.permute(0, 2, 1, 3)  # (batch, segment, channel, d_model)
        _, n_segments, n_channels, _ = embeds.shape
        position_embeds = self.positional_embedding(
            torch.arange(n_segments, device=x.device)
        ).unsqueeze(0)  # (1, segment, d_model)
        channel_embeds = self.channel_embedding(
            torch.arange(n_channels, device=x.device)
        ).unsqueeze(0)  # (1, channel, d_model)
        hidden = self.drop(embeds)
        for block in self.blocks:
            hidden = block(hidden, position_embeds, channel_embeds)
        hidden = self.ln_f(hidden)  # (batch, segment, channel, d_model)
        pooled = hidden[:, -1]  # last segment: (batch, channel, d_model)
        if self.pooling == "mean":
            pooled = pooled.mean(dim=1)  # (batch, d_model)
        else:
            pooled = pooled.flatten(1)  # (batch, channel * d_model)
        if return_features:
            out: Dict[str, Optional[torch.Tensor]] = {
                "features": pooled,
                "cls_token": None,
            }
            return out
        return self.final_layer(pooled)


class _WaveletPatchEmbed(nn.Module):
    """db4 wavelet patch embedding (MVPFormer signal encoder).

    Maps each time segment ``(..., segment_len)`` to a continuous embedding
    ``(..., d_model)``: a full-depth db4 discrete wavelet decomposition
    (:func:`~braindecode.functional.wavelet_decomposition`, periodic boundary)
    followed by RMSNorm and a linear projection. The db4 filters are computed
    from first principles (:func:`~braindecode.functional.daubechies_filters`),
    so no external wavelet library is needed. Only ``ln`` and ``proj`` are
    learnable.

    Parameters
    ----------
    segment_len : int
        Number of raw time samples per segment.
    d_model : int
        Output embedding dimension.
    """

    def __init__(self, segment_len: int, d_model: int):
        super().__init__()
        # db4 = Daubechies wavelet with 4 vanishing moments (8 taps).
        self.register_buffer("_filters", daubechies_filters(4), persistent=False)
        # Output length of the (fixed-depth) periodic decomposition for this segment.
        self.dwt_size = wavelet_decomposition(
            torch.zeros(1, segment_len), self._filters
        ).shape[-1]
        self.ln = nn.RMSNorm(self.dwt_size, eps=1e-5)
        self.proj = nn.Linear(self.dwt_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.promote_types(x.dtype, torch.float32))
        feats = wavelet_decomposition(x, self._filters)
        feats = self.ln(feats)
        # Match the projection weight dtype (robust across AMP / float16 inputs).
        return self.proj(feats.to(self.proj.weight.dtype))


def _repeat_segment_mask(mask: torch.Tensor, n_channels: int) -> torch.Tensor:
    """``(query_segment, key_segment)`` -> ``(query_segment * n_channels,
    key_segment * n_channels)``, each segment entry repeated per channel."""
    n_query, n_key = mask.shape
    mask = mask[:, None, :, None].expand(-1, n_channels, -1, n_channels)
    return mask.reshape(n_query * n_channels, n_key * n_channels)


def _finfo_min(dtype: torch.dtype) -> float:
    """``torch.finfo(dtype).min`` of a float dtype (TorchScript has no finfo)."""
    if dtype == torch.float16:
        return -65504.0
    if dtype == torch.bfloat16:
        return -3.3895313892515355e38
    if dtype == torch.float64:
        return -1.7976931348623157e308
    return -3.4028234663852886e38


class _MVPAttention(nn.Module):
    r"""Multi-variate parallel attention (MVPA).

    Attention over a 2D ``(segment, channel)`` token grid, decomposed into three
    additive terms (Carzaniga et al., 2026, eqs. 2-4):

    - **content** — query/key, restricted to a local window of ``local_window``
      segments, plus an optional **global** term over the whole context;
    - **time-relative** — Transformer-XL style, depends only on the segment
      distance, shared across channels;
    - **channel-relative** — depends only on the channel distance, shared across
      segments.

    Uses grouped-query attention (``n_head_kv`` <= ``n_heads``) and a causal mask
    over segments. This is the pure-PyTorch path; the Triton ``FlashMVPA`` kernel
    of the reference implementation is intentionally omitted.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_heads : int
        Number of attention (query) heads.
    n_head_kv : int
        Number of key/value heads (grouped-query attention).
    global_att : bool
        Whether to add the global content-attention term.
    local_window : int
        Size, in segments, of the causal local content-attention lookback window.
    attn_drop, resid_drop : float
        Dropout on attention weights and on the output projection.
    scale_attn : bool
        Scale logits by ``1 / sqrt(head_dim)``.
    scale_by_layer_idx : bool
        Additionally scale logits by ``1 / (layer_idx + 1)``.
    layer_idx : int or None
        Layer index (only used when ``scale_by_layer_idx``).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_head_kv: int,
        global_att: bool = True,
        local_window: int = 10,
        attn_drop: float = 0.0,
        resid_drop: float = 0.0,
        scale_attn: bool = True,
        scale_by_layer_idx: bool = False,
        layer_idx: int | None = None,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if n_heads % n_head_kv:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_head_kv ({n_head_kv})."
            )
        self.n_heads = n_heads
        self.n_head_kv = n_head_kv
        self.n_kv_groups = n_heads // n_head_kv
        self.head_dim = d_model // n_heads
        self.kv_dim = self.head_dim * n_head_kv
        self.global_att = global_att
        self.local_window = local_window
        self.scale_attn = scale_attn
        self.scale_by_layer_idx = scale_by_layer_idx
        self.layer_idx = layer_idx

        self.q_attn = nn.Linear(d_model, d_model, bias=False)
        self.c_attn = nn.Linear(d_model, 2 * self.kv_dim, bias=False)  # K | V
        self.position_net = nn.Linear(d_model, self.kv_dim, bias=False)  # W_{k,R}
        self.channel_net = nn.Linear(d_model, self.kv_dim, bias=False)  # W_{k,C}
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_bias = nn.Parameter(torch.empty(3 * self.kv_dim))  # u | v | w
        nn.init.normal_(self.attn_bias, std=0.02)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

    # -- shape helpers (segment = "time"/position axis, channel axis) ----------
    # The shape helpers below run the ops einops ran (reshape, permute,
    # unsqueeze + expand + reshape), so outputs and gradients are unchanged.
    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # grouped-query attention: replicate each kv head into ``n_rep`` query heads,
        # (batch, kv_head, ...) -> (batch, kv_head * group, ...).
        # Repeat a contiguous copy: on Intel Gaudi (HPU) the einops repeat of the
        # permuted singleton-axis relative keys returns scrambled values.
        x = x.contiguous()
        batch, kv_heads, segment, channel, head_dim = x.shape
        x = x.unsqueeze(2).expand(-1, -1, n_rep, -1, -1, -1)
        return x.reshape(batch, kv_heads * n_rep, segment, channel, head_dim)

    @staticmethod
    def _repeat_channel(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # broadcast a per-segment (time-relative) score across every channel:
        # (batch, head, query, key_segment) -> (..., key_segment * channel)
        return x.unsqueeze(-1).expand(-1, -1, -1, -1, n_rep).flatten(3)

    @staticmethod
    def _repeat_time(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # broadcast a per-channel score across every time segment:
        # (batch, head, query, key_channel) -> (..., segment * key_channel)
        return x.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).flatten(3)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        # Transformer-XL relative shift along the segment axis.
        batch, head, n_query, n_key = x.shape
        zero_pad = torch.zeros((batch, head, n_key, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x.view(batch, head, n_key, n_query)], dim=-1)
        x_padded = x_padded.view(batch, head, n_query + 1, n_key)
        return x_padded[..., 1:, :].view_as(x)

    @staticmethod
    def _rel_shift_chan(x):
        # Relative shift along the channel axis (symmetric distance), as a single
        # ``torch.gather`` whose backward is a ``scatter_add``. Advanced indexing
        # would backpropagate through ``index_put_(accumulate=True)``, which Intel
        # Gaudi (HPU) runs on the host, slowly and with wrong gradients. Index
        # tensors are built on x.device with long dtype.
        device = x.device
        chan_size = x.shape[-1]
        if chan_size > 1:
            upper_val = torch.cat(
                [
                    torch.arange(1, chan_size - i, dtype=torch.long, device=device)
                    for i in range(chan_size - 1)
                ]
            )
        else:
            upper_val = torch.zeros((0,), dtype=torch.long, device=device)
        idxes = torch.triu_indices(chan_size, chan_size, offset=1, device=device)
        shifting_idxes = torch.zeros(
            (chan_size, chan_size), dtype=torch.long, device=device
        )
        shifting_idxes[idxes[0], idxes[1]] = upper_val
        lower = shifting_idxes.transpose(-2, -1)  # a view: writes into shifting_idxes
        lower[idxes[0], idxes[1]] = upper_val
        shifting_idxes = (chan_size - 1 - shifting_idxes).repeat(
            x.shape[-2] // chan_size, 1
        )
        return torch.gather(
            x,
            -1,
            shifting_idxes.expand(list(x.shape[:-2]) + list(shifting_idxes.shape)),
        )

    def _split_heads(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        # (batch, segment, channel, head * head_dim)
        # -> (batch, head, segment, channel, head_dim)
        return tensor.unflatten(-1, (num_heads, -1)).permute(0, 3, 1, 2, 4)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        # (batch, head, segment, channel, head_dim)
        # -> (batch, segment, channel, head * head_dim)
        return tensor.permute(0, 2, 3, 1, 4).flatten(3)

    def _as_bias(self, bias: torch.Tensor) -> torch.Tensor:
        # (kv_dim,) -> (1, n_heads, 1, 1, head_dim)
        bias = bias.reshape(1, self.n_head_kv, 1, 1, -1)
        return self._repeat_kv(bias, self.n_kv_groups)

    def _rel_attn(
        self,
        query: torch.Tensor,
        content_key: torch.Tensor,
        time_key: torch.Tensor,
        channel_key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, _, n_segments, n_channels, _ = query.size()
        global_bias, time_bias, channel_bias = self.attn_bias.split(self.kv_dim, dim=0)

        time_bias = self._as_bias(time_bias)
        channel_bias = self._as_bias(channel_bias)

        # Relative keys: the time key depends only on the segment, the channel key
        # only on the channel (the broadcast singleton axis is squeezed out).
        # (batch, head, segment, 1, head_dim) -> (batch, head, head_dim, segment)
        time_key = self._repeat_kv(time_key, self.n_kv_groups).squeeze(3)
        time_key = time_key.permute(0, 1, 3, 2)
        # (batch, head, 1, channel, head_dim) -> (batch, head, head_dim, channel)
        channel_key = self._repeat_kv(channel_key, self.n_kv_groups).squeeze(2)
        channel_key = channel_key.permute(0, 1, 3, 2)

        # (batch, head, segment, channel, head_dim)
        # -> (batch, head, segment * channel, head_dim)
        time_q = (query + time_bias).flatten(2, 3)
        channel_q = (query + channel_bias).flatten(2, 3)
        time_att = self._rel_shift(torch.matmul(time_q, time_key))
        channel_att = self._rel_shift_chan(torch.matmul(channel_q, channel_key))
        attn_weights = self._repeat_channel(time_att, n_channels) + self._repeat_time(
            channel_att, n_segments
        )

        if self.global_att:
            global_bias = self._as_bias(global_bias)
            # (batch, head, segment, channel, head_dim)
            # -> (batch, head, head_dim, segment * channel)
            global_key = self._repeat_kv(content_key, self.n_kv_groups)
            global_key = global_key.permute(0, 1, 4, 2, 3).flatten(3)
            global_q = (query + global_bias).flatten(2, 3)
            global_att = torch.matmul(global_q, global_key)
            window = self.local_window
            ones = torch.ones(
                (n_segments, n_segments), device=query.device, dtype=torch.bool
            )
            window_mask = torch.logical_and(
                torch.tril(ones, diagonal=window),
                torch.triu(ones, diagonal=-window),
            )
            # (query_segment, key_segment)
            # -> (query_segment * query_channel, key_segment * key_channel)
            window_mask = _repeat_segment_mask(window_mask, n_channels).clone()
            window_mask[-n_channels:] = 1
            attn_weights = attn_weights + global_att.masked_fill(~window_mask, 0.0)

        if self.scale_attn:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)
        layer_idx = self.layer_idx
        if self.scale_by_layer_idx and layer_idx is not None:
            attn_weights = attn_weights / float(layer_idx + 1)

        causal_mask = _repeat_segment_mask(
            torch.tril(
                torch.ones(
                    (n_segments, n_segments), device=query.device, dtype=torch.bool
                )
            ),
            n_channels,
        )
        if torch.jit.is_scripting():  # torch.finfo is not scriptable
            min_value = _finfo_min(attn_weights.dtype)
        else:
            min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~causal_mask, min_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        value = self._repeat_kv(value, self.n_kv_groups)
        attn_output = torch.matmul(attn_weights, value.flatten(2, 3))
        # (batch, head, segment * channel, head_dim)
        # -> (batch, head, segment, channel, head_dim)
        return attn_output.unflatten(2, (-1, n_channels))

    def forward(
        self,
        hidden_states,
        position_embeds,
        channel_embeds,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        query = self.q_attn(hidden_states)
        content_key, value = self.c_attn(hidden_states).split(self.kv_dim, dim=-1)
        # add the broadcast axis (channel for the time key, segment for the channel key)
        time_key = self.position_net(position_embeds).unsqueeze(2)
        channel_key = self.channel_net(channel_embeds).unsqueeze(1)
        query = self._split_heads(query, self.n_heads)
        content_key = self._split_heads(content_key, self.n_head_kv)
        value = self._split_heads(value, self.n_head_kv)
        time_key = self._split_heads(time_key, self.n_head_kv)
        channel_key = self._split_heads(channel_key, self.n_head_kv)
        attn_output = self._rel_attn(
            query, content_key, time_key, channel_key, value, attention_mask
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        return self.resid_dropout(attn_output)


class _SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward (Llama MLP): ``down(act(gate(x)) * up(x))``."""

    def __init__(
        self, d_model: int, d_inner: int, activation: type[nn.Module] = nn.SiLU
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _MVPABlock(nn.Module):
    """MVPFormer decoder block: parallel attention + MLP (Megatron/Wang2021).

    ``out = x + attn(ln_1(x)) + mlp(ln_1(x))`` — a single pre-norm feeds both
    branches (matching the upstream block, whose ``ln_2`` is instantiated but
    unused; it is dropped here).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_head_kv: int,
        d_inner: int,
        global_att: bool = True,
        local_window: int = 10,
        attn_drop: float = 0.0,
        resid_drop: float = 0.0,
        eps: float = 1e-5,
        scale_attn: bool = True,
        scale_by_layer_idx: bool = False,
        layer_idx: int | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.ln_1 = nn.RMSNorm(d_model, eps=eps)
        self.attn = _MVPAttention(
            d_model,
            n_heads,
            n_head_kv,
            global_att,
            local_window,
            attn_drop,
            resid_drop,
            scale_attn,
            scale_by_layer_idx,
            layer_idx,
        )
        self.mlp = _SwiGLUMLP(d_model, d_inner, activation=activation)

    def forward(
        self,
        hidden_states,
        position_embeds,
        channel_embeds,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        normed = self.ln_1(hidden_states)
        attn_output = self.attn(normed, position_embeds, channel_embeds, attention_mask)
        mlp_output = self.mlp(normed)
        return hidden_states + mlp_output + attn_output

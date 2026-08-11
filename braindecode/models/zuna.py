# Authors: Chris Warner, Jonas Mago, Jon Huml
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.nn import functional

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import PatchTokenizer


class ZUNA(EEGModuleMixin, nn.Module):
    r"""ZUNA from Warner et al (2026) [Warner2026]_.

    :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel` :bdg-info:`Attention/Transformer`

    .. versionadded:: 1.7

    .. figure:: ../_static/model/zuna_arch.png
       :align: center
       :alt: ZUNA encoder-decoder architecture
       :width: 1000px

    ZUNA is a position-aware diffusion autoencoder for EEG superresolution.

    Architecture defaults follow the published ``Zyphra/ZUNA1.1`` encoder.
    Inputs default to five-second windows sampled at 256 Hz. Channel
    coordinates are read once from ``chs_info`` with
    :func:`braindecode.models.util.extract_channel_locations_from_chs_info` and
    encoded in the model's fixed rotary-position buffers.

    :meth:`forward` returns ``(batch, n_outputs)`` logits by default, or a
    dict of intermediate latents when ``return_features=True``.

    .. rubric:: Architecture Overview

    Each channel's window is cut into non-overlapping patches of
    ``fine_time_pts`` samples (0.125 s at 256 Hz), giving a sequence of
    ``n_chans * (n_times // fine_time_pts)`` tokens. Tokens are linearly
    embedded, interleaved with learned register tokens, and processed by a
    stack of ``n_layers`` transformer blocks whose attention is rotated by a
    4D rotary embedding over the token's discretised scalp coordinates
    ``(x, y, z)`` and its coarse-time index. The register slots are projected
    to per-token latents, mean-pooled over time per channel, and classified.

    .. rubric:: Macro Components

    - ``ZUNA.encoder`` (:class:`torch.nn.Module`)

      **Operations**: ``tok_embeddings`` (linear patch embedding) →
      interleave ``registers`` → ``n_layers`` × ``_TransformerBlock``
      (RMS-normed multi-head attention with 4D RoPE and QK-norm, SwiGLU
      feed-forward, sandwich norm) → ``norm`` → ``output`` linear projection
      to ``latent_dim`` per token.

      **Role**: pretrained, position-aware EEG token encoder.

    - ``ZUNA.final_layer`` (:class:`torch.nn.Sequential`)

      **Operations**: flatten the ``(n_chans, latent_dim)`` channel embedding
      → linear map to ``n_outputs``.

      **Role**: randomly initialised classification head fine-tuned on the
      downstream task.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal**: 0.125-s patch tokens plus a coarse-time rotary axis; the
      time dimension is mean-pooled after encoding.
    - **Spatial**: three rotary axes carry each channel's bucketed 3D scalp
      coordinates, making the encoder montage-agnostic.
    - **Spectral**: no explicit frequency decomposition; spectral structure
      is learned from the raw 256 Hz patches.

    .. rubric:: Additional Mechanisms

    - **Register tokens**: one learned register interleaved per data token;
      only register slots are read out, decoupling readout from input tokens.
    - **Scaled dot-product attention**: PyTorch SDPA selects the available
      optimized attention kernel for the current device and dtype.
    - **Fixed positional grid**: channel and coarse-time rotary positions are
      computed once during construction, keeping the forward graph fully
      tensor-based.

    Parameters
    ----------
    dim : int
        Transformer embedding dimension of the encoder.
    n_layers : int
        Number of transformer blocks in the encoder.
    n_heads : int
        Number of attention heads per block.
    head_dim : int
        Dimension of each attention head. Must be divisible by eight.
    fine_time_pts : int
        Number of fine time points per token (the encoder input dimension).
        ``n_times`` must be divisible by this value. The ZUNA1.1
        encoder uses ``32`` samples, equivalent to 0.125-second coarse-time
        tokens at 256 Hz.
    latent_dim : int
        Per-token latent dimension produced by the encoder (the encoder
        output dimension).
    max_seqlen : int
        Size of the rotary-frequency table. It must cover ``pos_bins`` and
        ``n_times // fine_time_pts``.
    rope_theta : float
        Base period of the rotary positional embedding.
    pos_bins : int
        Number of discretisation bins per spatial axis for channel
        coordinates.
    pos_half_range : float
        Half-range (in metres) used to normalise channel coordinates before
        bucketing (scalp-radius normalisation).
    norm_eps : float
        Epsilon of the RMS normalisation layers.
    multiple_of : int
        Feed-forward hidden dimension is rounded up to a multiple of this
        value.
    ffn_dim_multiplier : float | None
        Optional multiplier applied to the feed-forward hidden dimension.
    sandwich_norm : bool
        Whether to apply the ZUNA1.1 post-attention and post-FFN RMS norms.
    qk_norm : bool
        Whether to apply ZUNA1.1 query/key RMS norms inside attention.
    activation : type[nn.Module]
        Feed-forward activation. The default is :class:`torch.nn.SiLU`, as
        used by the pretrained encoder.

    References
    ----------
    .. [Warner2026] Warner, C., Mago, J., Huml, J.R. and Millidge, B.,
       2026. ZUNA1.1: A more flexible EEG foundation model for Denoising and
       Super-resolution. arXiv preprint arXiv:2607.27308.
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        chs_info: Optional[list[dict]] = None,
        n_times: Optional[int] = None,
        input_window_seconds: Optional[float] = None,
        sfreq: Optional[float] = None,
        # model-specific parameters
        *,
        dim: int = 1024,
        n_layers: int = 16,
        n_heads: int = 8,
        head_dim: int = 64,
        fine_time_pts: int = 32,
        latent_dim: int = 32,
        max_seqlen: int = 256,
        rope_theta: float = 10000.0,
        pos_bins: int = 50,
        pos_half_range: float = 0.12,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ):
        sfreq = 256.0 if sfreq is None else sfreq
        if n_times is None and input_window_seconds is None:
            n_times = 1280

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.num_channels = self.n_chans
        self.latent_dim = latent_dim
        self.patch_embedding = nn.Sequential(
            PatchTokenizer(
                patch_size=fine_time_pts,
                n_times=self.n_times,
                on_non_divisible="error",
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        channel_locations = extract_channel_locations_from_chs_info(
            self._chs_info, num_channels=self.num_channels
        )
        if channel_locations is None:
            raise ValueError("ZUNA requires channel locations in chs_info.")
        channel_positions = torch.as_tensor(channel_locations, dtype=torch.float32)
        if (
            channel_positions.shape != (self.num_channels, 3)
            or not torch.isfinite(channel_positions).all()
        ):
            raise ValueError("ZUNA requires finite 3D locations for every channel.")

        coarse_time_points = self.n_times // fine_time_pts
        normalized_positions = (channel_positions + pos_half_range) / (
            2 * pos_half_range
        )
        channel_position_indices = (
            (normalized_positions * pos_bins).long().clamp(0, pos_bins - 1)
        )
        channel_position_indices = channel_position_indices.repeat_interleave(
            coarse_time_points, dim=0
        )
        coarse_time_indices = torch.arange(coarse_time_points).repeat(self.num_channels)
        token_position_indices = torch.cat(
            (channel_position_indices, coarse_time_indices.unsqueeze(1)), dim=1
        )

        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by eight for 4D RoPE.")
        rotary_axis_dim = head_dim // 4
        rotary_embedding = RotaryEmbedding(
            # rotary_embedding_torch cannot construct dim=2 directly because
            # of its theta-rescaling formula; dim=4 followed by slicing is
            # equivalent for the single-frequency dim=2 case.
            dim=max(rotary_axis_dim, 4),
            theta=rope_theta,
            cache_if_possible=False,
        )
        with torch.no_grad():
            rotary_frequency_table = rotary_embedding(
                torch.arange(max_seqlen, dtype=torch.float32)
            )[:, :rotary_axis_dim]
        token_rotary_frequencies = rotary_frequency_table[
            token_position_indices
        ].flatten(1, 2)
        token_rotary_frequencies = token_rotary_frequencies.repeat_interleave(2, dim=0)

        self.encoder = _ZUNAEncoder(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            input_dim=fine_time_pts,
            output_dim=latent_dim,
            rotary_frequencies=token_rotary_frequencies,
            norm_eps=norm_eps,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            sandwich_norm=sandwich_norm,
            qk_norm=qk_norm,
            activation=activation,
        )
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_channels * self.latent_dim, self.n_outputs),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        return_features: bool = False,
    ):
        patch_tokens = self.patch_embedding(input_tensor)
        token_latents = self.encoder(patch_tokens)
        structured_latents = token_latents.reshape(
            token_latents.shape[0], self.num_channels, -1, self.latent_dim
        )
        features = structured_latents.mean(dim=2)
        logits = self.final_layer(features)

        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {
                "features": features,
                "cls_token": None,  # nosec B105
            }
        return logits

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_channels * self.latent_dim, n_outputs),
        )


class _RotaryPositionEmbedding(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rotary_cosine: torch.Tensor,
        rotary_sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_cosine = rotary_cosine.unsqueeze(0).unsqueeze(2)
        rotary_sine = rotary_sine.unsqueeze(0).unsqueeze(2)

        batch_size, sequence_length, num_heads, head_dim = query.shape
        query_pairs = query.float().reshape(
            batch_size, sequence_length, num_heads, head_dim // 2, 2
        )
        key_pairs = key.float().reshape(
            batch_size, sequence_length, num_heads, head_dim // 2, 2
        )
        rotated_query = torch.stack(
            (-query_pairs[..., 1], query_pairs[..., 0]), dim=-1
        ).flatten(-2)
        rotated_key = torch.stack(
            (-key_pairs[..., 1], key_pairs[..., 0]), dim=-1
        ).flatten(-2)

        query = (query.float() * rotary_cosine + rotated_query * rotary_sine).type_as(
            query
        )
        key = (key.float() * rotary_cosine + rotated_key * rotary_sine).type_as(key)
        return query, key


class _RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    ``torch.nn.RMSNorm`` is only available from PyTorch 2.4, but braindecode
    supports ``torch>=2.0``; this shippable equivalent (same approach as
    :class:`~braindecode.models.REVE` and ``CodeBrain``) keeps the model
    importable on older PyTorch while preserving the ``.weight`` parameter
    name for state-dict compatibility.
    """

    def __init__(self, dimension: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized = input_tensor.float() * torch.rsqrt(
            input_tensor.float().pow(2).mean(-1, keepdim=True) + self.epsilon
        )
        return normalized.type_as(self.weight) * self.weight


class _Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, embedding_dim, bias=False)
        self.q_norm = _RMSNorm(head_dim, epsilon=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(head_dim, epsilon=norm_eps) if qk_norm else nn.Identity()
        self.rotary_embedding = _RotaryPositionEmbedding()

    def forward(
        self,
        input_tensor: torch.Tensor,
        rotary_cosine: torch.Tensor,
        rotary_sine: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = input_tensor.shape
        attention_shape = (
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )
        query = self.q_norm(self.wq(input_tensor).reshape(attention_shape))
        key = self.k_norm(self.wk(input_tensor).reshape(attention_shape))
        value = self.wv(input_tensor).reshape(attention_shape)
        query, key = self.rotary_embedding(query, key, rotary_cosine, rotary_sine)

        attention_output = functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.n_heads * self.head_dim
        )
        return self.wo(attention_output)


class _FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        hidden_dim = int(8 * embedding_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)
        self.w1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.activation = activation()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(input_tensor)) * self.w3(input_tensor))


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        head_dim: int,
        norm_eps: float,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.attention = _Attention(
            embedding_dim,
            n_heads,
            head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
        )
        self.feed_forward = _FeedForward(
            embedding_dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            activation=activation,
        )
        self.attention_norm = _RMSNorm(embedding_dim, epsilon=norm_eps)
        self.ffn_norm = _RMSNorm(embedding_dim, epsilon=norm_eps)
        self.attention_norm_post = (
            _RMSNorm(embedding_dim, epsilon=norm_eps)
            if sandwich_norm
            else nn.Identity()
        )
        self.ffn_norm_post = (
            _RMSNorm(embedding_dim, epsilon=norm_eps)
            if sandwich_norm
            else nn.Identity()
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        rotary_cosine: torch.Tensor,
        rotary_sine: torch.Tensor,
    ) -> torch.Tensor:
        input_tensor = input_tensor.float()
        hidden_states = input_tensor + self.attention_norm_post(
            self.attention(
                self.attention_norm(input_tensor), rotary_cosine, rotary_sine
            ).float()
        )
        return hidden_states + self.ffn_norm_post(
            self.feed_forward(self.ffn_norm(hidden_states)).float()
        )


class _ZUNAEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        input_dim: int,
        output_dim: int,
        rotary_frequencies: torch.Tensor,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.tok_embeddings = nn.Linear(input_dim, dim)
        self.registers = nn.Parameter(torch.zeros(1, input_dim))
        self.layers = nn.ModuleList(
            _TransformerBlock(
                dim,
                n_heads,
                head_dim,
                norm_eps,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                sandwich_norm=sandwich_norm,
                qk_norm=qk_norm,
                activation=activation,
            )
            for _ in range(n_layers)
        )
        self.norm = _RMSNorm(dim, epsilon=norm_eps)
        self.output = nn.Linear(dim, output_dim, bias=False)
        self.register_buffer(
            "rotary_cosine",
            rotary_frequencies.cos(),
            persistent=False,
        )
        self.register_buffer(
            "rotary_sine",
            rotary_frequencies.sin(),
            persistent=False,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, patch_size = patch_tokens.shape
        register_tokens = self.registers.expand(batch_size, sequence_length, -1)
        interleaved_tokens = torch.stack(
            (register_tokens, patch_tokens), dim=2
        ).reshape(batch_size, 2 * sequence_length, patch_size)

        hidden_states = self.tok_embeddings(interleaved_tokens)
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.rotary_cosine, self.rotary_sine)
        register_latents = hidden_states.reshape(batch_size, sequence_length, 2, -1)[
            :, :, 0
        ]
        return self.output(self.norm(register_latents))

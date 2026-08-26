# Authors: Chris Warner, Jonas Mago, Jon Huml
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import PatchTokenizer


class ZUNA(EEGModuleMixin, nn.Module):
    r"""ZUNA from Warner et al. (2026) [Warner2026ZUNA]_.

    :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel` :bdg-info:`Attention/Transformer`

    .. versionadded:: 1.7

    .. figure:: ../_static/model/zuna_arch.png
       :align: center
       :alt: ZUNA pretraining encoder-decoder architecture
       :width: 1000px

    ZUNA was introduced as a diffusion autoencoder for masked EEG channel
    reconstruction and super-resolution [Warner2026ZUNA]_. ZUNA1.1 retains
    that objective and adds query-key normalization, sandwich normalization,
    pretraining windows from 0.5 to 30 seconds, and eight channel and time
    dropout schemes [Warner2026ZUNA11]_.

    This Braindecode class contains the ZUNA1.1 encoder followed by a
    classification head. It does not contain the diffusion decoder, channel
    masking, or reconstruction sampler shown in the figure. :meth:`forward`
    returns logits of shape ``(batch, n_outputs)``. With
    ``return_features=True``, it returns the per-channel encoder features used
    by the classification head.

    Signal size and sampling frequency follow the standard Braindecode model
    arguments. Supply either ``n_times`` or both ``input_window_seconds`` and
    ``sfreq``; ZUNA does not set these values implicitly. ZUNA1.1 was trained
    at 256 Hz with 32-sample tokens. This implementation does not
    resample, filter, or normalize the input. Channel coordinates are read from
    ``chs_info`` when the model is constructed and stored in fixed rotary
    position buffers.

    .. rubric:: Architecture Overview

    :class:`~braindecode.modules.PatchTokenizer` splits every channel into
    non-overlapping patches of ``fine_time_pts`` samples. The resulting
    ``n_chans * (n_times // fine_time_pts)`` patches are serialized in
    channel-major order. A learned register is placed before each patch, and
    both are projected to ``dim`` before entering the transformer blocks.

    Self-attention is bidirectional. Four-dimensional rotary positions encode
    each token's discretized scalp coordinates ``(x, y, z)`` and coarse-time
    index. The encoder reads the register positions, projects them to
    ``latent_dim``, restores the channel and patch axes, and averages over the
    patch axis. A linear layer maps the concatenated channel features to
    ``n_outputs``.

    .. rubric:: Macro Components

    - ``ZUNA.patch_embedding`` (:class:`torch.nn.Sequential`)

      **Operations**: split ``(batch, channel, time)`` into patches with
      :class:`~braindecode.modules.PatchTokenizer`, then rearrange
      ``(channel, temporal_patch)`` into one token axis.

      **Role**: produce one continuous-valued token for every channel and time
      patch.

    - ``ZUNA.encoder`` (:class:`torch.nn.Module`)

      **Operations**: ``tok_embeddings`` (linear patch embedding) →
      interleave ``registers`` → ``n_layers`` × ``_TransformerBlock``
      (RMS-normed multi-head attention with 4D RoPE and QK-norm, SwiGLU
      feed-forward, sandwich norm) → ``norm`` → ``output`` linear projection
      to ``latent_dim`` per token.

      **Role**: encode channel-time patches while retaining their spatial and
      temporal coordinates.

    - ``ZUNA.final_layer`` (:class:`torch.nn.Sequential`)

      **Operations**: rearrange the ``(n_chans, latent_dim)`` channel features
      into one axis → linear map to ``n_outputs``.

      **Role**: produce task logits from the pooled encoder features.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal**: fixed-size patches provide local samples, and the fourth
      rotary axis identifies each patch's coarse-time index. The patch axis is
      averaged after encoding.
    - **Spatial**: the first three rotary axes contain bucketed 3D coordinates
      from ``chs_info``. A model instance uses the montage supplied at
      construction.
    - **Spectral**: there is no Fourier or filter-bank stage. The encoder
      receives raw time-domain patches.

    .. rubric:: Additional Mechanisms

    - **Register tokens**: one learned register is paired with every patch.
      Only register outputs enter the latent projection.
    - **ZUNA1.1 normalization**: query-key RMS normalization and optional
      post-attention and post-feed-forward RMS normalization match the changes
      introduced in ZUNA1.1.
    - **Fixed positional grid**: channel and coarse-time rotary values are
      computed during construction. Inputs passed to :meth:`forward` must have
      the configured channel count and window length.

    Parameters
    ----------
    dim : int, optional
        Transformer embedding dimension. The default is ``1024``.
    n_layers : int, optional
        Number of transformer blocks. The default is ``16``.
    n_heads : int, optional
        Number of attention heads per block. The default is ``8``.
    head_dim : int, optional
        Dimension of each attention head. It must be divisible by eight. The
        default is ``64``.
    fine_time_pts : int, optional
        Number of fine time points per token (the encoder input dimension).
        ``n_times`` must be divisible by this value. The default is ``32``, or
        0.125 seconds for data sampled at 256 Hz.
    latent_dim : int, optional
        Per-token output dimension of the encoder. The default is ``32``.
    max_seqlen : int, optional
        Length of the rotary frequency table. It must be at least
        ``max(pos_bins, n_times // fine_time_pts)``. The default is ``256``.
    rope_theta : float, optional
        Base period of the rotary positional embedding. The default is
        ``10000.0``.
    pos_bins : int, optional
        Number of buckets per spatial coordinate. The default is ``50``.
    pos_half_range : float, optional
        Half-range (in metres) used to normalise channel coordinates before
        bucketing. Coordinates at or beyond this range are clipped to the first
        or last bucket. The default is ``0.12``.
    norm_eps : float, optional
        Epsilon of the RMS normalization layers. The default is ``1e-5``.
    multiple_of : int, optional
        Feed-forward hidden dimension is rounded up to a multiple of this
        value. The default is ``256``.
    ffn_dim_multiplier : float | None, optional
        Multiplier applied before rounding the feed-forward hidden dimension.
        The default is ``None``.
    sandwich_norm : bool, optional
        Apply RMS normalization after attention and the feed-forward layer. The
        default is ``True``.
    qk_norm : bool, optional
        Apply RMS normalization to queries and keys. The default is ``True``.
    activation : type[nn.Module], optional
        Feed-forward activation class. The default is
        :class:`torch.nn.SiLU`.

    Notes
    -----
    The full ZUNA1.1 pretraining model has an encoder and a rectified-flow
    decoder. Its encoder latent is regularized with a maximum mean discrepancy
    loss. The decoder and both pretraining losses are outside this class.

    In the paper's downstream experiments, token latents were averaged and the
    encoder was fine-tuned with the classifier. A frozen encoder with a linear
    head performed worse. The authors also found that checkpoints with lower
    reconstruction error were not necessarily better for classification
    [Warner2026ZUNA11]_.

    References
    ----------
    .. [Warner2026ZUNA] Warner, C., Mago, J., Huml, J. R., Osman, M., and
       Millidge, B. (2026). ZUNA: Flexible EEG Superresolution with
       Position-Aware Diffusion Autoencoders. arXiv:2602.18478.
       https://arxiv.org/abs/2602.18478
    .. [Warner2026ZUNA11] Warner, C., Mago, J., Huml, J. R., and Millidge, B.
       (2026). ZUNA1.1: A More Flexible EEG Foundation Model for Denoising and
       Super-Resolution. arXiv:2607.27308.
       https://arxiv.org/abs/2607.27308
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
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Variables
        self.num_channels = self.n_chans
        self.latent_dim = latent_dim
        coarse_time_points = self.n_times // fine_time_pts
        rotary_axis_dim = head_dim // 4

        # Checks
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
        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by eight for 4D RoPE.")

        # Layers
        self.patch_embedding = nn.Sequential(
            PatchTokenizer(
                patch_size=fine_time_pts,
                n_times=self.n_times,
                on_non_divisible="error",
            ),
            Rearrange(
                "batch channel temporal_patch sample "
                "-> batch (channel temporal_patch) sample"
            ),
        )

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

        rotary_frequency_table = _build_rotary_frequency_table(
            torch.arange(max_seqlen, dtype=torch.float32),
            axis_dim=rotary_axis_dim,
            theta=rope_theta,
        )
        token_rotary_frequencies = rearrange(
            rotary_frequency_table[token_position_indices],
            "token coordinate rotary_frequency -> token (coordinate rotary_frequency)",
        )
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
            Rearrange("batch channel latent -> batch (channel latent)"),
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
            Rearrange("batch channel latent -> batch (channel latent)"),
            nn.Linear(self.num_channels * self.latent_dim, n_outputs),
        )


def _build_rotary_frequency_table(
    positions: torch.Tensor, *, axis_dim: int, theta: float
) -> torch.Tensor:
    """Build ZUNA's per-axis rotary frequencies with native PyTorch."""
    embedding_dim = max(axis_dim, 4)
    inverse_frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                embedding_dim,
                2,
                device=positions.device,
                dtype=torch.float32,
            )
            / embedding_dim
        )
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse_frequencies
    return angles.repeat_interleave(2, dim=-1)[:, :axis_dim]


class _RotaryPositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.merge_rotary_pairs = Rearrange(
            "batch sequence head frequency_pair complex_component "
            "-> batch sequence head (frequency_pair complex_component)"
        )

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
        rotated_query = self.merge_rotary_pairs(
            torch.stack((-query_pairs[..., 1], query_pairs[..., 0]), dim=-1)
        )
        rotated_key = self.merge_rotary_pairs(
            torch.stack((-key_pairs[..., 1], key_pairs[..., 0]), dim=-1)
        )

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

        # Variables
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Layers
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

        # Variables
        hidden_dim = int(8 * embedding_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)

        # Layers
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

        # Layers
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

        # Layers
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

        # Buffers
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

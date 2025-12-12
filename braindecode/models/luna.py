"""This implementation is adapted from ETH Zurich's BioFoundation repository.

Döner, B., Ingolfsson, T. M., Benini, L., & Li, Y. (2025).
LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis.
The Thirty-Ninth Annual Conference on Neural Information Processing Systems, NeurIPS.
Retrieved from https://openreview.net/forum?id=uazfjnFL0G

Original Authors: Berkay Döner, Thorir Mar Ingolfsson
Braindecode Adaptation: Bruno Aristimunha

the LICENSE Of this file is APACHE-2.0.
"""

import math
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules.layers import DropPath


class LUNA(EEGModuleMixin, nn.Module):
    """LUNA from Döner et al. [LUNA]_.

    :bdg-success:`Convolution` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    .. figure:: https://arxiv.org/html/2510.22257v1/x1.png
        :align: center
        :alt: LUNA Architecture.

    LUNA is a topology-invariant EEG model that processes signals from varying
    numbers of channels using a channel-unification mechanism with learned queries.

    The architecture consists of:
    1. Patch Feature Extraction (temporal CNN + FFT-based features)
    2. Channel-Unification Module (cross-attention with learned queries)
    3. Patch-wise Temporal Encoder (RoPE-based transformer)
    4. Decoder Heads (classification or reconstruction)

    Parameters
    ----------
    patch_size : int
        Number of time samples per patch. Default: 40.
    num_queries : int
        Number of learned queries for channel unification.
        Paper uses: 4 (Base), 6 (Large), 8 (Huge). Default: 4.
    embed_dim : int
        Embedding dimension for patch features.
        Paper uses: 64 (Base), 96 (Large), 128 (Huge). Default: 64.
    depth : int
        Number of transformer encoder blocks.
        Paper uses: 8 (Base), 10 (Large), 24 (Huge). Default: 8.
    num_heads : int
        Number of attention heads in channel unification.
        Default: 2.
    mlp_ratio : float
        Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
    norm_layer : nn.Module
        Normalization layer class. Default: nn.LayerNorm.
    drop_path : float
        Stochastic depth rate. Default: 0.0.

    References
    ----------
    .. [LUNA] Döner, B., Ingolfsson, T. M., Benini, L., & Li, Y. (2025).
        LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis.
        The Thirty-Ninth Annual Conference on Neural Information Processing Systems - NeurIPS.
        Retrieved from https://openreview.net/forum?id=uazfjnFL0G
    """

    def __init__(
        self,
        # Braindecode EEGModuleMixin parameters
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        n_times: Optional[int] = None,
        sfreq: Optional[float] = None,
        chs_info: Optional[Any] = None,
        input_window_seconds: Optional[float] = None,
        # Model-specific parameters
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        drop_path: float = 0.0,
        drop_prob_chan: float = 0.0,
        attn_drop: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Mapping for loading pretrained weights
        self.mapping = {
            "channel_location_embedder.0.fc1.weight": "channel_location_embedder.fc1.weight",
            "channel_location_embedder.0.fc1.bias": "channel_location_embedder.fc1.bias",
            "channel_location_embedder.0.fc2.weight": "channel_location_embedder.fc2.weight",
            "channel_location_embedder.0.fc2.bias": "channel_location_embedder.fc2.bias",
            "channel_location_embedder.0.norm.weight": "channel_location_embedder.norm.weight",
            "channel_location_embedder.0.norm.bias": "channel_location_embedder.norm.bias",
        }

        # Model parameters
        self.num_classes = self.n_outputs if self.n_outputs else 0
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.patch_embed_size = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.drop_path = drop_path
        self.attn_drop = attn_drop
        self.drop_prob_chan = drop_prob_chan
        self.mlp_ratio = mlp_ratio
        self.activation = activation

        # Layers
        self.patch_embed = _PatchEmbedNetwork(
            embed_dim=self.embed_dim, patch_size=self.patch_size
        )
        self.freq_embed = _FrequencyFeatureEmbedder(
            embed_dim=self.embed_dim, patch_size=self.patch_size
        )
        # For weight loading, we omit the normalization here to match parameter count
        self.channel_location_embedder = _Mlp(
            in_features=int(self.patch_embed_size),
            out_features=int(self.patch_embed_size),
            hidden_features=int(self.patch_embed_size * 2),
            act_layer=self.activation,
            drop=self.drop_prob_chan,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cross_attn = _CrossAttentionBlock(
            num_queries=self.num_queries,
            input_embed_dim=self.embed_dim,
            output_embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=int(self.mlp_ratio * self.embed_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _RotaryTransformerBlock(
                    dim=int(self.embed_dim * self.num_queries),
                    num_heads=int(self.num_heads * self.num_queries),
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=self.drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(int(self.embed_dim * self.num_queries))

        self._channel_location_cache: Dict[int, torch.Tensor] = {}

        if self.num_classes == 0:
            self.decoder_head = _PatchReconstructionHeadWithQueries(
                input_dim=self.patch_size,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_queries=self.num_queries,
            )
            self.channel_emb = _ChannelEmbeddings(self.embed_dim)
        else:
            self.final_layer = _ClassificationHeadWithQueries(
                input_dim=self.patch_size,
                num_queries=self.num_queries,
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
                num_heads=self.num_heads,
            )
            self.mask_token.requires_grad = (
                False  # no use of mask token for classification
            )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.cross_attn.initialize_weights()
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def prepare_tokens(
        self,
        x_signal: torch.Tensor,
        channel_locations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_channels = channel_locations.shape[1]
        num_patches_per_channel = x_signal.shape[-1] // self.patch_size
        x_patched = self.patch_embed(x_signal)
        freq_embed = self.freq_embed(x_signal)
        x_patched = x_patched + freq_embed
        x_masked = x_patched.clone()  # (B, N, D), N = C * num_patches_per_channel

        if mask is not None:
            mask_tokens = self.mask_token.repeat(
                x_masked.shape[0], x_masked.shape[1], 1
            )  # (B, N, D) N = C * num_patches_per_channel
            mask = rearrange(
                mask, "B C (S P) -> B (C S) P", P=self.patch_size
            )  # (B, C, T) -> (B, N, P)
            mask = (
                (mask.sum(dim=-1) > 0).unsqueeze(-1).float()
            )  # (B, N, 1), since a patch is either fully masked or not
            x_masked = torch.where(mask.bool(), mask_tokens, x_masked)

        channel_min = torch.min(channel_locations, dim=1, keepdim=True)[0]
        channel_max = torch.max(channel_locations, dim=1, keepdim=True)[0]
        channel_locations = (channel_locations - channel_min) / (
            channel_max - channel_min + 1e-8
        )

        if mask is not None:
            channel_locations = (
                channel_locations + torch.randn_like(channel_locations) * 0.02
            )

        channel_locations = nerf_positional_encoding(
            channel_locations, self.patch_embed_size
        )
        channel_locations_emb = self.channel_location_embedder(channel_locations)

        x_tokenized = rearrange(x_masked, "B (C t) D -> (B t) C D", C=num_channels)
        channel_locations_emb = channel_locations_emb.repeat(
            num_patches_per_channel, 1, 1
        )
        x_tokenized = x_tokenized + channel_locations_emb

        return x_tokenized, channel_locations_emb

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        channel_locations: Optional[torch.Tensor] = None,
        channel_names: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x_signal = X
        B, C, _ = x_signal.shape

        if channel_locations is None:
            channel_locations = self.get_default_channel_locations(
                batch_size=B,
                num_channels=C,
                device=x_signal.device,
                dtype=x_signal.dtype,
            )

        x, channel_locations_emb = self.prepare_tokens(
            x_signal, channel_locations, mask=mask
        )
        x, _ = self.cross_attn(x)
        x = rearrange(x, "(B t) Q D -> B t (Q D)", B=B)
        num_patches = x.shape[1]

        for blk in self.blocks:
            x = blk(x)
        x_latent = self.norm(x)

        if self.num_classes > 0:
            return self.final_layer(x_latent)

        if channel_names is None:
            raise ValueError("channel_names must be provided for reconstruction tasks.")
        channel_emb = self.channel_emb(channel_names)
        channel_emb = channel_emb.repeat(num_patches, 1, 1)
        decoder_queries = channel_locations_emb + channel_emb
        return self.decoder_head(x_latent, decoder_queries)

    def get_default_channel_locations(
        self,
        batch_size: int,
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_channels not in self._channel_location_cache:
            template = self.build_channel_location_template(num_channels)
            self._channel_location_cache[num_channels] = template
        template = self._channel_location_cache[num_channels].to(
            device=device, dtype=dtype
        )
        return template.unsqueeze(0).repeat(batch_size, 1, 1)

    def build_channel_location_template(self, num_channels: int) -> torch.Tensor:
        """Build channel location template for the model.

        Attempts to extract channel locations from chs_info. Falls back to a default
        linear spacing along the x-axis if real locations are unavailable.

        Parameters
        ----------
        num_channels : int
            Number of channels to generate locations for.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_channels, 3) with channel locations in 3D space.
        """
        # Try to extract channel locations from chs_info using the unified utility
        channel_info = getattr(self, "_chs_info", None)
        if channel_info is not None:
            locs = extract_channel_locations_from_chs_info(
                channel_info, num_channels=num_channels
            )
            if locs is not None and len(locs) == num_channels:
                return torch.from_numpy(locs).float()

        # Fallback: generate default linear spacing along x-axis
        positions = torch.linspace(-1.0, 1.0, steps=num_channels, dtype=torch.float32)
        zeros = torch.zeros_like(positions)
        locs_tensor = torch.stack([positions, zeros, zeros], dim=-1)
        return locs_tensor

    def _get_default_channel_locations(
        self,
        batch_size: int,
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.get_default_channel_locations(
            batch_size=batch_size,
            num_channels=num_channels,
            device=device,
            dtype=dtype,
        )

    def _build_channel_location_template(self, num_channels: int) -> torch.Tensor:
        return self.build_channel_location_template(num_channels)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    """
    coords: (N, C, 3)
    Returns: (N, C, embed_size)
    """
    N, C, dim = coords.shape
    device = coords.device
    freqs = embed_size // (2 * dim)
    leftover = embed_size - freqs * 2 * dim
    freq_bands = 2.0 ** torch.arange(freqs, device=device).float()
    scaled_coords = coords.unsqueeze(-1) * freq_bands.view(
        1, 1, 1, -1
    )  # (N, C, dim, freqs)
    sin_enc = torch.sin(scaled_coords)  # (N, C, dim, freqs)
    cos_enc = torch.cos(scaled_coords)  # (N, C, dim, freqs)
    encoded = (
        torch.stack([sin_enc, cos_enc], dim=-1)
        .permute(0, 1, 3, 2, 4)
        .reshape(N, C, freqs * dim * 2)
    )
    if leftover > 0:
        pad = torch.zeros(N, C, leftover, device=device, dtype=coords.dtype)
        encoded = torch.cat([encoded, pad], dim=-1)
    return encoded


class _ChannelEmbeddings(nn.Module):
    """
    This class creates embeddings for each EEG channel based on a predefined
    mapping of channel names to indices.

    The number of unique channels is determined by the union of channels
    from SEED Pretraining, TUEG, and Siena datasets.

    Parameters
    ----------
    embed_dim : int
        Dimension of the channel embeddings.
    number_channels : int
        Number of unique EEG channels. Default is 90.

    """

    def __init__(self, embed_dim: int, number_channels=90) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(number_channels, embed_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embeddings(indices)

    def initialize_weights(self) -> None:
        torch.init.normal_(self.embeddings.weight, std=2.0)


class _FrequencyFeatureEmbedder(nn.Module):
    """
    This class takes data that is of the form (B, C, T) and patches it
    along the time dimension (T) into patches of size P (patch_size).
    The output is of the form (B, C, S, P) where S = T // P.
    """

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        in_features = 2 * (patch_size // 2 + 1)
        self.frequency_to_embed = _Mlp(
            in_features=in_features,
            hidden_features=int(4 * in_features),
            out_features=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.size()
        S = T // self.patch_size
        # There is a chance that the input tensor is not divisible by the patch size
        # In this case we need to pad the tensor with zeros
        if T % self.patch_size != 0:
            # Pad last dimension with zeros to make it divisible by patch size
            pad_size = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, pad_size))
            T = x.size(-1)
            S = T // self.patch_size
        x = x.view(B, C, S, self.patch_size)

        freq_representation = fft.rfft(
            x, dim=-1
        )  # (B, C, num_patches, patch_size // 2 + 1)
        magnitude = torch.abs(freq_representation)
        phase = torch.angle(freq_representation)

        # Concatenate magnitude and phase along the frequency axis (last dimension)
        freq_features = torch.cat((magnitude, phase), dim=-1)
        # Map frequency features to embedding dimension
        embedded = self.frequency_to_embed(
            freq_features
        )  # (B, C, num_patches, embed_dim)
        embedded = rearrange(embedded, "B C t D -> B (C t) D")
        return embedded


class _RotarySelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim, learned_freq=False)

        self.scale = qk_scale or head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.attn_drop_fn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (K, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        # Calculate attention scores
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Apply softmax to get attention probabilities
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply dropout
        attn_weights = self.attn_drop_fn(attn_weights)

        # Apply attention weights to values
        attn = attn_weights @ v  # (B, H, N, D)
        attn = rearrange(attn, "B H N D -> B N (H D)")
        return self.proj_drop(self.proj(attn))


class _FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class _RotaryTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _FeedForwardBlock(
            dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class _PatchReconstructionHeadWithQueries(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_queries: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.reconstruction_shape = self.input_dim
        self.num_queries = num_queries
        # Projection from embed space to pixel space, according to type of input
        self.decoder_pred = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embed_dim,
                num_heads,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                dim_feedforward=int(embed_dim * 4),
                norm_first=True,
            ),
            num_layers=1,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_linear = _Mlp(
            embed_dim, int(embed_dim * 4), input_dim, act_layer=nn.GELU, drop=0.0
        )  # nn.Linear(embed_dim, input_dim, bias=True)

    def forward(self, enc: torch.Tensor, decoder_queries: torch.Tensor) -> torch.Tensor:
        """
        enc: [B, num_patches, embed_dim], embed_dim = Q*D
        decoder_queries: [B*num_patches, num_channels, embed_dim]
        """

        B, num_patches, embed_dim = enc.shape
        enc = rearrange(enc, "B t (Q D) -> (B t) Q D", Q=self.num_queries)
        out = self.decoder_pred(decoder_queries, enc)  # (B*t, C, D)
        out = self.norm(out)
        out = self.decoder_linear(out)  # (B*t, C, patch_size)
        out = rearrange(out, "(B t) C P -> B C (t P)", B=B)
        return out


class _ClassificationHeadWithQueries(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_queries: int = 8,
        num_heads: int = 8,
        num_classes: int = 2,
        drop_decoder: float = 0.15,
        drop_ffn: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(embed_dim * num_queries)
        self.reconstruction_shape = self.input_dim
        self.decoder_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads, batch_first=True, dropout=drop_decoder
        )
        self.decoder_ffn = _Mlp(
            in_features=self.embed_dim,
            hidden_features=int(self.embed_dim * 4),
            out_features=num_classes,
            act_layer=nn.GELU,
            drop=drop_ffn,
        )

        self.learned_agg = nn.Parameter(
            torch.randn(1, 1, self.embed_dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Output shape:
            [B, num_tokens, in_chans, input_dim]
        Args:
            x: [B, num_tokens+1, embed_dim]
            channel_embeddings: [B, in_chans, embed_dim]
        """
        B, num_patches, embed_dim = x.shape
        decoder_queries = self.learned_agg.repeat(x.shape[0], 1, 1)

        x = self.decoder_attn(query=decoder_queries, key=x, value=x)[0]
        x = x[:, 0, :]
        x = self.decoder_ffn(x)
        return x


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        num_queries: int,
        input_embed_dim: int,
        output_embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        ff_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.dropout_p = dropout_p
        self.query_embed = nn.Parameter(
            torch.randn(1, num_queries, input_embed_dim), requires_grad=True
        )  # Learnable queries
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_embed_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.ffn = _Mlp(
            input_embed_dim,
            ff_dim,
            output_embed_dim,
            act_layer=nn.GELU,
            drop=dropout_p,
        )
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        self.queries_norm = nn.LayerNorm(input_embed_dim)
        self.query_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_embed_dim,
                nhead=num_heads,
                activation="gelu",
                dim_feedforward=ff_dim,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )

    def initialize_weights(self) -> None:
        torch.nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is the input with shape (batch_size*num_patches, num_channels, embed_dim)
        batch_size, _, _ = x.size()
        queries = self.query_embed.repeat(batch_size, 1, 1)
        queries = self.queries_norm(queries)
        keys = self.keys_norm(x)
        values = self.values_norm(x)

        attention_out, attention_scores = self.cross_attention(
            query=queries, key=keys, value=values
        )  # Shape: (batch_size*num_patches, num_queries, embed_dim)
        attention_out = self.ffn(attention_out) + attention_out

        attention_out = self.query_self_attn(attention_out)
        return (
            attention_out,
            attention_scores,
        )  # Shape: (batch_size*num_patches, num_queries, embed_dim)


class _PatchEmbedNetwork(nn.Module):
    def __init__(self, embed_dim: int = 64, patch_size: int = 40) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = 1
        self.out_channels = int(embed_dim // 4)
        self.groups = 4
        self.kernel_size = int(patch_size // 2)
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, self.kernel_size - 1),
                stride=(1, self.kernel_size // 2),
                padding=(0, self.kernel_size // 2 - 1),
            ),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        output: (B, C*S, D) where S = T//patch_size, D = embed_dim
        """
        x = rearrange(x, "B C (S P) -> B (C S) P", P=self.patch_size)
        x = x.unsqueeze(1)
        x = self.proj_in(x)
        x = rearrange(x, "B E CS D -> B CS (D E)")
        return x


class _Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    Code copied from timm.models.mlp.Mlp
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Thorir Mar Ingolfsson <thoriri@ethz.ch>
#
# License: Apache-2.0

"""
LUNA (Latent Unified Network Architecture) module.

Implementation of the topology-agnostic EEG foundation model from Döner et al. (2024).
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_ as torch_trunc_normal_

from braindecode.models.base import EEGModuleMixin

# External dependencies
try:
    from rotary_embedding_torch import RotaryEmbedding

    HAS_ROPE = True
except ImportError:
    HAS_ROPE = False
    RotaryEmbedding = None

try:
    from timm.models.layers import DropPath
    from timm.models.layers import Mlp as TimmMlp

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    TimmMlp = None
    DropPath = None


# =============================================================================
# Helper Functions
# =============================================================================


def trunc_normal_(tensor, mean=0.0, std=1.0):
    """Wrapper for truncated normal initialization."""
    torch_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    """
    NeRF-style positional encoding for 3D electrode coordinates.

    Parameters
    ----------
    coords : torch.Tensor
        3D electrode coordinates, shape (N, C, 3) where:
        - N: batch size
        - C: number of channels
        - 3: (x, y, z) coordinates
    embed_size : int
        Size of the positional embedding.

    Returns
    -------
    torch.Tensor
        Positional embeddings, shape (N, C, embed_size).

    Notes
    -----
    This function applies sinusoidal encoding to 3D spatial coordinates using
    multiple frequency bands. The encoding helps the model understand spatial
    relationships between electrodes.

    The formula is:
    PE(pos, 2i) = sin(pos * 2^i)
    PE(pos, 2i+1) = cos(pos * 2^i)

    Applied to each coordinate dimension (x, y, z) separately.
    """
    N, C, dim = coords.shape
    device = coords.device

    # Calculate number of frequency bands
    freqs = embed_size // (2 * dim)
    leftover = embed_size - freqs * 2 * dim

    # Create frequency bands: [1, 2, 4, 8, ...]
    freq_bands = 2.0 ** torch.arange(freqs, device=device).float()

    # Expand coordinates for broadcasting
    # coords: (N, C, 3) -> (N, C, 3, 1)
    # freq_bands: (freqs,) -> (1, 1, 1, freqs)
    coords_expanded = coords.unsqueeze(-1)
    freq_bands_expanded = freq_bands.view(1, 1, 1, -1)

    # Apply frequency encoding: coords * [1, 2, 4, 8, ...]
    # Result shape: (N, C, 3, freqs)
    scaled_coords = coords_expanded * freq_bands_expanded

    # Apply sin and cos
    sin_enc = torch.sin(scaled_coords)  # (N, C, 3, freqs)
    cos_enc = torch.cos(scaled_coords)  # (N, C, 3, freqs)

    # Interleave sin and cos: [sin, cos, sin, cos, ...]
    # Stack along last dimension and flatten
    encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # (N, C, 3, freqs, 2)
    encoding = encoding.flatten(2)  # (N, C, 3*freqs*2)

    # Handle leftover dimensions with zeros
    if leftover > 0:
        padding = torch.zeros(N, C, leftover, device=device)
        encoding = torch.cat([encoding, padding], dim=-1)

    return encoding


# =============================================================================
# Component 1: Patch Feature Extraction
# =============================================================================


class PatchEmbedNetwork(nn.Module):
    """
    Temporal patch embedding using 2D CNN.

    Extracts features from temporal patches using a 3-layer convolutional network
    with GroupNorm and GELU activation. Uses 2D convolutions to process all
    channel-patch combinations simultaneously.

    Parameters
    ----------
    embed_dim : int
        Dimension of the output embeddings.
    patch_size : int
        Size of each temporal patch.

    Notes
    -----
    Architecture:
    - Conv2d (1→16 channels), kernel=(1,19), stride=(1,10), GroupNorm(4), GELU
    - Conv2d (16→16), kernel=(1,3), stride=(1,1), GroupNorm(4), GELU
    - Conv2d (16→16), kernel=(1,3), stride=(1,1), GroupNorm(4), GELU

    Output: (B, C*S, D*16) where S = num_patches, D = reduced temporal dim
    """

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = 1
        self.out_channels = int(embed_dim // 4)
        self.groups = 4
        self.kernel_size = int(patch_size // 2)

        # 3-layer 2D CNN with GroupNorm and GELU
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
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape (B, C, T).

        Returns
        -------
        torch.Tensor
            Patch embeddings, shape (B, C*S, D*E) where:
            - S = T // patch_size (number of patches)
            - D = reduced temporal dimension
            - E = self.out_channels
        """
        B, C, T = x.shape

        # Reshape into patches: (B, C, T) -> (B, C*S, P)
        # where S = T // patch_size, P = patch_size
        x = rearrange(x, "B C (S P) -> B (C S) P", P=self.patch_size)

        # Add channel dimension for Conv2d: (B, C*S, P) -> (B, 1, C*S, P)
        x = x.unsqueeze(1)

        # Apply 2D convolutions
        x = self.proj_in(x)  # (B, E, C*S, D)

        # Rearrange: (B, E, C*S, D) -> (B, C*S, D*E)
        x = rearrange(x, "B E CS D -> B CS (D E)")

        return x


class FrequencyFeatureEmbedder(nn.Module):
    """
    FFT-based frequency feature embedding.

    Extracts frequency features using FFT on each patch, then passes magnitude
    and phase through an MLP to produce embeddings.

    Parameters
    ----------
    patch_size : int
        Size of each temporal patch.
    embed_dim : int
        Dimension of the output embeddings.

    Notes
    -----
    Uses torch.fft.rfft to compute real FFT, extracts magnitude and phase,
    concatenates them, and passes through an MLP.

    Output shape: (B, C*S, embed_dim) where S = number of patches
    """

    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # FFT produces (patch_size // 2 + 1) frequency bins
        # Factor of 2 because we concatenate magnitude and phase
        in_features = 2 * (patch_size // 2 + 1)

        # MLP: (2 * num_freq_bins) -> embed_dim
        if HAS_TIMM and TimmMlp is not None:
            self.frequency_to_embed = TimmMlp(
                in_features=in_features,
                hidden_features=int(4 * in_features),
                out_features=embed_dim,
                act_layer=nn.GELU,
            )
        else:
            self.frequency_to_embed = nn.Sequential(
                nn.Linear(in_features, int(4 * in_features)),
                nn.GELU(),
                nn.Linear(int(4 * in_features), embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape (B, C, T).

        Returns
        -------
        torch.Tensor
            Frequency embeddings, shape (B, C*S, embed_dim) where
            S = T // patch_size (number of patches).
        """
        B, C, T = x.shape
        S = T // self.patch_size

        # Handle non-divisible lengths
        if T % self.patch_size != 0:
            pad_size = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, pad_size))
            T = x.shape[-1]
            S = T // self.patch_size

        # Reshape into patches: (B, C, T) -> (B, C, S, patch_size)
        x = x.view(B, C, S, self.patch_size)

        # Apply FFT on each patch
        freq_representation = torch.fft.rfft(x, dim=-1)  # (B, C, S, patch_size//2+1)

        # Extract magnitude and phase
        magnitude = torch.abs(freq_representation)
        phase = torch.angle(freq_representation)

        # Concatenate magnitude and phase
        freq_features = torch.cat([magnitude, phase], dim=-1)  # (B, C, S, 2*num_bins)

        # Apply MLP to map to embedding dimension
        embedded = self.frequency_to_embed(freq_features)  # (B, C, S, embed_dim)

        # Rearrange: (B, C, S, embed_dim) -> (B, C*S, embed_dim)
        embedded = rearrange(embedded, "B C S D -> B (C S) D")

        return embedded


# =============================================================================
# Component 3: Temporal Encoder with RoPE
# =============================================================================


class RotarySelfAttentionBlock(nn.Module):
    """
    Self-attention block with Rotary Position Embeddings (RoPE).

    RoPE encodes positional information by rotating query and key vectors,
    enabling better generalization to different sequence lengths.

    Parameters
    ----------
    dim : int
        Dimension of the input embeddings.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        Whether to use bias in QKV projection.
    qk_scale : float, optional
        Scale factor for attention scores. If None, uses 1/sqrt(head_dim).
    attn_drop : float
        Dropout probability for attention weights.
    proj_drop : float
        Dropout probability for output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Rotary embedding (learned_freq=False as per paper)
        self.rotary_emb = RotaryEmbedding(dim=head_dim, learned_freq=False)

        self.scale = qk_scale or head_dim**-0.5

        # QKV projection
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.attn_drop_fn = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, N, C).
        """
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings to Q and K
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Calculate attention scores
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply dropout
        attn_weights = self.attn_drop_fn(attn_weights)

        # Apply attention to values
        attn = attn_weights @ v  # (B, H, N, D)
        attn = rearrange(attn, "B H N D -> B N (H D)")

        # Output projection with dropout
        return self.proj_drop(self.proj(attn))


class FeedForwardBlock(nn.Module):
    """
    Feed-forward network with GELU activation.

    Parameters
    ----------
    dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension (typically 4× input dimension).
    dropout : float
        Dropout probability.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class RotaryTransformerBlock(nn.Module):
    """
    Transformer encoder block with RoPE.

    Pre-norm architecture with residual connections and stochastic depth.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dimension expansion ratio.
    qkv_bias : bool
        Whether to use bias in QKV projection.
    qk_scale : float, optional
        Scale for attention scores.
    drop : float
        Dropout probability.
    attn_drop : float
        Attention dropout probability.
    drop_path : float
        Stochastic depth probability.
    norm_layer : nn.Module
        Normalization layer class.
    """

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
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Stochastic depth
        if HAS_TIMM and DropPath is not None:
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        else:
            self.drop_path1 = nn.Identity()
            self.drop_path2 = nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = FeedForwardBlock(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, N, C).
        """
        # Pre-norm + attention + residual
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # Pre-norm + FFN + residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


# =============================================================================
# Component 4: Decoder Heads
# =============================================================================


class ClassificationHeadWithQueries(nn.Module):
    """
    Classification head using a learned aggregation query.

    Uses cross-attention with a single learned query to aggregate information
    from all temporal patches, followed by an MLP classifier.

    Parameters
    ----------
    input_dim : int
        Input patch size (not used, kept for compatibility).
    embed_dim : int
        Embedding dimension per query.
    num_queries : int
        Number of queries in the encoder output.
    num_heads : int
        Number of attention heads.
    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_queries: int = 8,
        num_heads: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(embed_dim * num_queries)  # Q×E
        self.reconstruction_shape = self.input_dim

        # Cross-attention for aggregation
        self.decoder_attn = nn.MultiheadAttention(
            self.embed_dim,
            num_heads,
            batch_first=True,
            dropout=0.15,
        )

        # MLP classifier
        if HAS_TIMM and TimmMlp is not None:
            self.decoder_ffn = TimmMlp(
                in_features=self.embed_dim,
                hidden_features=int(self.embed_dim * 4),
                out_features=num_classes,
                act_layer=nn.GELU,
                drop=0.15,
            )
        else:
            self.decoder_ffn = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim * 4)),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(int(self.embed_dim * 4), num_classes),
                nn.Dropout(0.15),
            )

        # Learned aggregation query
        self.learned_agg = nn.Parameter(
            torch.randn(1, 1, self.embed_dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Encoder output, shape (B, num_patches, Q×E).

        Returns
        -------
        torch.Tensor
            Class logits, shape (B, num_classes).
        """
        B, num_patches, embed_dim = x.shape

        # Expand learned aggregation query for batch
        decoder_queries = self.learned_agg.repeat(x.shape[0], 1, 1)  # (B, 1, Q×E)

        # Cross-attention: single query attends to all patches
        x, _ = self.decoder_attn(query=decoder_queries, key=x, value=x)  # (B, 1, Q×E)

        # Take the single aggregated token
        x = x[:, 0, :]  # (B, Q×E)

        # MLP classifier
        x = self.decoder_ffn(x)  # (B, num_classes)

        return x


class PatchReconstructionHeadWithQueries(nn.Module):
    """
    Reconstruction head for pre-training (optional).

    Uses C learned decoder queries (one per channel) and cross-attention
    to reconstruct masked patches.

    Parameters
    ----------
    input_dim : int
        Patch size (number of timesteps to reconstruct).
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_queries : int
        Number of queries in the encoder.
    """

    def __init__(
        self,
        input_dim: int = 40,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_queries: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.reconstruction_shape = self.input_dim
        self.num_queries = num_queries

        # Transformer decoder (1 layer)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            dim_feedforward=int(embed_dim * 4),
            norm_first=True,
        )
        self.decoder_pred = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.norm = nn.LayerNorm(embed_dim)

        # MLP to project to patch size
        if HAS_TIMM and TimmMlp is not None:
            self.decoder_linear = TimmMlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * 4),
                out_features=input_dim,
                act_layer=nn.GELU,
                drop=0.0,
            )
        else:
            self.decoder_linear = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * 4)),
                nn.GELU(),
                nn.Linear(int(embed_dim * 4), input_dim),
            )

    def forward(self, enc: torch.Tensor, decoder_queries: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        enc : torch.Tensor
            Encoder output, shape (B, num_patches, Q×E).
        decoder_queries : torch.Tensor
            Channel decoder queries, shape (B×num_patches, C, E).

        Returns
        -------
        torch.Tensor
            Reconstructed signal, shape (B, C, T).
        """
        B, num_patches, embed_dim = enc.shape

        # Rearrange encoder output: (B, S, Q×E) -> (B×S, Q, E)
        enc = rearrange(enc, "B S (Q E) -> (B S) Q E", Q=self.num_queries)

        # Cross-attention: decoder queries attend to encoder output
        out = self.decoder_pred(decoder_queries, enc)  # (B×S, C, E)

        # Normalize and project to patch size
        out = self.norm(out)
        out = self.decoder_linear(out)  # (B×S, C, patch_size)

        # Rearrange to signal: (B×S, C, P) -> (B, C, S×P)
        out = rearrange(out, "(B S) C P -> B C (S P)", B=B)

        return out


# =============================================================================
# Component 2: Channel-Unification Module
# =============================================================================


class CrossAttentionBlock(nn.Module):
    """
    Channel-unification module using cross-attention with learned queries.

    This is the core innovation of LUNA: maps variable number of channels (C)
    to a fixed number of queries (Q), achieving O(Q×C) complexity instead of
    O(C²) or O((C×S)²).

    Parameters
    ----------
    num_queries : int
        Number of learned queries (Q=4 for Base, Q=6 for Large, Q=8 for Huge).
    input_embed_dim : int
        Dimension of input embeddings (E).
    output_embed_dim : int
        Dimension of output embeddings (should equal input_embed_dim).
    num_heads : int
        Number of attention heads for cross-attention.
    dropout_p : float
        Dropout probability.
    ff_dim : int
        Hidden dimension of the feedforward network (typically 4×embed_dim).
    pre_norm : bool
        Whether to apply LayerNorm before attention (True for LUNA).

    Notes
    -----
    Architecture:
    1. Q learned query embeddings
    2. Cross-attention: queries attend to input channel embeddings
    3. Feedforward network on queries
    4. Query self-attention (3 layers) to refine representations
    """

    def __init__(
        self,
        num_queries: int,
        input_embed_dim: int,
        output_embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        ff_dim: int = 2048,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.pre_norm = pre_norm

        # Learned query embeddings: (1, Q, E)
        self.query_embed = nn.Parameter(
            torch.randn(1, num_queries, input_embed_dim), requires_grad=True
        )

        # Cross-attention: queries (Q) attend to input (C×S)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_embed_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )

        # Temperature parameter (not used, set to 1.0 for compatibility)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Normalization layers
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        self.queries_norm = nn.LayerNorm(input_embed_dim)

        # Feedforward network on queries
        if HAS_TIMM and TimmMlp is not None:
            self.ffn = TimmMlp(
                in_features=input_embed_dim,
                hidden_features=ff_dim,
                out_features=output_embed_dim,
                act_layer=nn.GELU,
                drop=dropout_p,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(input_embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(ff_dim, output_embed_dim),
                nn.Dropout(dropout_p),
            )

        # Query self-attention (3 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_embed_dim,
            nhead=num_heads,
            activation="gelu",
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.query_self_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
        )

    def initialize_weights(self):
        """Initialize weights using orthogonal initialization for queries."""
        nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear and normalization layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input channel×patch embeddings, shape (B*S, C, E).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Query embeddings: shape (B*S, Q, E)
            - Attention scores: shape (B*S, Q, C)
        """
        batch_size, num_channels, _ = x.size()

        # Expand queries for batch: (1, Q, E) -> (B*S, Q, E)
        queries = self.query_embed.repeat(batch_size, 1, 1)

        # Apply normalization
        queries_norm = self.queries_norm(queries)
        keys_norm = self.keys_norm(x)
        values_norm = self.values_norm(x)

        # Cross-attention: queries attend to input
        attention_out, attention_scores = self.cross_attention(
            query=queries_norm,
            key=keys_norm,
            value=values_norm,
        )  # attention_out: (B*S, Q, E), attention_scores: (B*S, Q, C)

        # Feedforward network with residual connection
        attention_out = self.ffn(attention_out) + attention_out

        # Query self-attention (3 layers)
        attention_out = self.query_self_attn(attention_out)

        return attention_out, attention_scores


# =============================================================================
# LUNA Model (Partial - Component 1 & 2 Complete)
# =============================================================================


class LUNA(EEGModuleMixin, nn.Module):
    """
    LUNA: Latent Unified Network Architecture for EEG Analysis.

    Topology-agnostic foundation model with linear complexity in channel count.
    From Döner et al. (2024) [Doner2024]_.

    .. versionadded:: 0.10

    LUNA addresses two fundamental challenges in EEG analysis:

    1. **Topology Heterogeneity**: Different datasets use varying electrode counts
       and layouts (20, 22, 29, 62 channels). LUNA works seamlessly across all.

    2. **Computational Complexity**: Standard transformers scale as O(C²) or O((C×S)²).
       LUNA achieves O(Q×C) through a channel-unification module with learned queries.

    The architecture consists of 4 main components:

    1. **Patch Feature Extraction**: Temporal CNN + FFT + Channel positional encoding
    2. **Channel-Unification**: Q learned queries map C channels to fixed latent space
    3. **Temporal Encoder**: Transformer blocks with RoPE on patch sequence
    4. **Decoder Head**: Classification (primary) or Reconstruction (pre-training)

    Three model sizes are available:
    - Base: 7M parameters (Q=4, E=64, depth=8)
    - Large: 43M parameters (Q=6, E=96, depth=10)
    - Huge: 311M parameters (Q=8, E=128, depth=24)

    Parameters
    ----------
    n_outputs : int, optional
        Number of output classes. If None, uses reconstruction mode.
    n_chans : int, optional
        Number of EEG channels.
    n_times : int, optional
        Number of time samples in the input window.
    sfreq : float, optional
        Sampling frequency in Hz.
    chs_info : list of dict, optional
        Channel information from mne.Info for extracting electrode locations.
    input_window_seconds : float, optional
        Length of the input window in seconds.
    patch_size : int, default=40
        Size of each temporal patch (timestamps).
    num_queries : int, default=4
        Number of learned queries (Q). Use 4 for Base, 6 for Large, 8 for Huge.
    embed_dim : int, default=64
        Embedding dimension (E). Use 64 for Base, 96 for Large, 128 for Huge.
    depth : int, default=8
        Number of transformer layers. Use 8 for Base, 10 for Large, 24 for Huge.
    num_heads : int, default=2
        Number of attention heads (per query).
    mlp_ratio : float, default=4.0
        MLP hidden dimension expansion ratio.
    drop_path : float, default=0.1
        Stochastic depth probability.

    Raises
    ------
    ValueError
        If required parameters cannot be inferred.
    ImportError
        If rotary-embedding-torch is not installed.

    References
    ----------
    .. [Doner2024] Döner, B., Ingolfsson, T. M., et al. (2024).
       LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis.
       arXiv preprint arXiv:2510.22257.

    Examples
    --------
    Create LUNA-Base for 4-class classification:

    >>> from braindecode.models import LUNA
    >>> model = LUNA(
    ...     n_outputs=4,
    ...     n_chans=22,
    ...     n_times=1000,
    ...     sfreq=250,
    ...     patch_size=40,
    ...     num_queries=4,
    ...     embed_dim=64,
    ...     depth=8,
    ... )

    The model works with variable channel counts:

    >>> import torch
    >>> x_22ch = torch.randn(2, 22, 1000)  # 22 channels
    >>> x_62ch = torch.randn(2, 62, 1000)  # 62 channels
    >>> out_22 = model(x_22ch)  # Works!
    >>> out_62 = model(x_62ch)  # Also works!

    Load pre-trained weights from HuggingFace:

    >>> # model = LUNA.from_pretrained("thorir/LUNA-base")

    Notes
    -----
    **Topology Invariance**: Unlike other EEG models, LUNA can process inputs
    with different channel counts without retraining. This enables transfer
    learning across heterogeneous datasets.

    **Efficiency**: LUNA uses 300× fewer FLOPs than BIOT on high-density EEG
    while maintaining state-of-the-art performance.

    **Pre-training**: The model can be pre-trained using reconstruction mode
    (set n_outputs=None) with masked patch prediction. For most users,
    classification mode with pre-trained weights is recommended.
    """

    def __init__(
        self,
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        n_times: Optional[int] = None,
        sfreq: Optional[float] = None,
        chs_info: Optional[list] = None,
        input_window_seconds: Optional[float] = None,
        # LUNA-specific parameters
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
    ):
        # Initialize EEGModuleMixin for parameter inference
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            chs_info=chs_info,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
        )

        # Check dependencies
        if not HAS_ROPE:
            raise ImportError(
                "rotary-embedding-torch is required for LUNA. "
                "Install with: pip install rotary-embedding-torch"
            )

        if not HAS_TIMM:
            warnings.warn(
                "timm not found. Using fallback implementations for Mlp and DropPath. "
                "Install timm for better performance: pip install timm"
            )

        # Validate parameters
        if self.n_times is not None and self.n_times % patch_size != 0:
            warnings.warn(
                f"n_times ({self.n_times}) is not divisible by patch_size ({patch_size}). "
                f"Input will be padded to {((self.n_times // patch_size) + 1) * patch_size} samples."
            )

        # Store LUNA-specific parameters
        self.patch_size = patch_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.patch_embed_size = embed_dim  # For compatibility with original
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path

        # Determine mode: classification or reconstruction
        self.mode = (
            "classification"
            if self.n_outputs is not None and self.n_outputs > 0
            else "reconstruction"
        )

        # Component 1: Patch Feature Extraction
        self.patch_embed = PatchEmbedNetwork(embed_dim=embed_dim, patch_size=patch_size)
        self.freq_embed = FrequencyFeatureEmbedder(
            patch_size=patch_size, embed_dim=embed_dim
        )

        # Channel location embedder MLP
        # Input: embed_dim (from NeRF encoding), Output: embed_dim
        if HAS_TIMM and TimmMlp is not None:
            self.channel_location_embedder = TimmMlp(
                in_features=embed_dim,
                hidden_features=embed_dim * 2,
                out_features=embed_dim,
                act_layer=nn.GELU,
                drop=0.0,
            )
        else:
            self.channel_location_embedder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )

        # Mask token (for pre-training)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Component 2: Channel-Unification Module
        self.cross_attn = CrossAttentionBlock(
            num_queries=num_queries,
            input_embed_dim=embed_dim,
            output_embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=0.1,
            ff_dim=int(mlp_ratio * embed_dim),
            pre_norm=True,
        )

        # Component 3: Temporal Encoder (RoPE transformer blocks)
        hidden_dim = int(embed_dim * num_queries)  # Q×E
        num_temporal_heads = int(num_heads * num_queries)  # Scales with queries

        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_temporal_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Component 4: Decoder Heads
        if self.mode == "reconstruction":
            # Reconstruction mode (pre-training)
            self.decoder_head = PatchReconstructionHeadWithQueries(
                input_dim=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_queries=num_queries,
            )
            # TODO: Add channel embeddings for reconstruction
            # self.channel_emb = ChannelEmbeddings(embed_dim)
        else:
            # Classification mode (fine-tuning)
            self.classifier = ClassificationHeadWithQueries(
                input_dim=patch_size,
                num_queries=num_queries,
                embed_dim=embed_dim,
                num_classes=self.n_outputs,
                num_heads=num_temporal_heads,
            )
            # Freeze mask token in classification mode
            self.mask_token.requires_grad = False

        # Remove placeholder
        # self.placeholder_output = nn.Linear(num_queries * embed_dim, n_outputs if n_outputs else 1)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize cross-attention
        self.cross_attn.initialize_weights()

        # Initialize mask token
        trunc_normal_(self.mask_token, std=0.02)

        # Apply standard initialization to other modules
        self.apply(self._init_weights)

        # Apply weight rescaling for better gradient flow
        self.fix_init_weight()

    def _init_weights(self, m):
        """Initialize weights for linear and normalization layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        """
        Rescale transformer weights for better gradient flow.

        Divides attention projection and MLP output weights by sqrt(2 * layer_id)
        to prevent gradient explosion in deep networks.
        """

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _get_channel_locations(self, n_chans: int) -> torch.Tensor:
        """
        Get channel locations from chs_info or use default positions.

        Parameters
        ----------
        n_chans : int
            Number of channels.

        Returns
        -------
        torch.Tensor
            Channel locations, shape (1, n_chans, 3).
        """
        # TODO: Extract from self.chs_info if available
        # For now, use uniform grid as placeholder
        # This should be improved to use MNE standard montages
        locations = torch.randn(1, n_chans, 3) * 0.1
        return locations

    def prepare_tokens(
        self,
        x_signal: torch.Tensor,
        channel_locations: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare patch tokens with channel location embeddings.

        Parameters
        ----------
        x_signal : torch.Tensor
            Input EEG signal, shape (B, C, T).
        channel_locations : torch.Tensor, optional
            3D electrode coordinates, shape (B, C, 3) or (1, C, 3).
            If None, will use default positions.
        mask : torch.Tensor, optional
            Masking tensor for pre-training, shape (B, C, T).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - x_tokenized: Patch embeddings with channel locations, shape (B*S, C, E)
            - channel_locations_emb: Channel location embeddings, shape (B*S, C, E)
        """
        B, C, T = x_signal.shape
        num_patches_per_channel = T // self.patch_size

        # Extract patch embeddings (temporal CNN)
        x_patched = self.patch_embed(x_signal)  # (B, C*S, D)

        # Extract frequency embeddings
        freq_embed = self.freq_embed(x_signal)  # (B, C*S, embed_dim)

        # Combine temporal and frequency embeddings
        x_patched = x_patched + freq_embed

        # Apply masking if provided (for pre-training)
        x_masked = x_patched.clone()
        if mask is not None:
            # Reshape mask to match patches
            mask = rearrange(mask, "B C (S P) -> B (C S) P", P=self.patch_size)
            # A patch is masked if any of its timesteps are masked
            mask = (mask.sum(dim=-1) > 0).unsqueeze(-1).float()  # (B, C*S, 1)

            # Expand mask token and apply
            mask_tokens = self.mask_token.repeat(
                x_masked.shape[0], x_masked.shape[1], 1
            )
            x_masked = torch.where(mask.bool(), mask_tokens, x_masked)

        # Process channel locations
        if channel_locations is None:
            channel_locations = self._get_channel_locations(C)
            channel_locations = channel_locations.to(x_signal.device)

        # Ensure channel_locations has batch dimension
        if channel_locations.dim() == 2:
            channel_locations = channel_locations.unsqueeze(0)

        # Normalize channel locations to [0, 1]
        channel_min = torch.min(channel_locations, dim=1, keepdim=True)[0]
        channel_max = torch.max(channel_locations, dim=1, keepdim=True)[0]
        channel_locations = (channel_locations - channel_min) / (
            channel_max - channel_min + 1e-8
        )

        # Add small jitter during training with mask (data augmentation)
        if mask is not None and self.training:
            channel_locations = (
                channel_locations + torch.randn_like(channel_locations) * 0.02
            )

        # Apply NeRF positional encoding
        channel_locations = nerf_positional_encoding(
            channel_locations, self.patch_embed_size
        )

        # Map through MLP
        channel_locations_emb = self.channel_location_embedder(
            channel_locations
        )  # (B, C, E)

        # Rearrange patch embeddings: (B, C*S, E) -> (B*S, C, E)
        x_tokenized = rearrange(x_masked, "B (C S) E -> (B S) C E", C=C)

        # Repeat channel location embeddings for all patches: (B, C, E) -> (B*S, C, E)
        # Each of B batches has S patches, so we need to repeat B times and interleave
        channel_locations_emb = channel_locations_emb.unsqueeze(1)  # (B, 1, C, E)
        channel_locations_emb = channel_locations_emb.expand(
            B, num_patches_per_channel, C, -1
        )  # (B, S, C, E)
        channel_locations_emb = rearrange(
            channel_locations_emb, "B S C E -> (B S) C E"
        )  # (B*S, C, E)

        # Add channel location embeddings to patch embeddings
        x_tokenized = x_tokenized + channel_locations_emb

        return x_tokenized, channel_locations_emb

    def forward(
        self, X: torch.Tensor, channel_locations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through complete LUNA architecture.

        Parameters
        ----------
        X : torch.Tensor
            Input EEG signal, shape (batch, n_chans, n_times).
        channel_locations : torch.Tensor, optional
            3D electrode coordinates, shape (batch, n_chans, 3) or (1, n_chans, 3).
            If None, will use default positions.

        Returns
        -------
        torch.Tensor
            Output predictions:
            - Classification mode: shape (batch, n_outputs)
            - Reconstruction mode: shape (batch, n_chans, n_times)
        """
        x_original = X
        B, C, T = X.shape

        # Pad if necessary
        if T % self.patch_size != 0:
            pad_len = ((T // self.patch_size) + 1) * self.patch_size - T
            X = F.pad(X, (0, pad_len))
            T = X.shape[2]

        num_patches = T // self.patch_size

        # Component 1: Prepare tokens with patch embeddings and channel locations
        x_tokenized, channel_locations_emb = self.prepare_tokens(
            X, channel_locations=channel_locations, mask=None
        )  # (B*S, C, E)

        # Component 2: Channel-Unification
        # Cross-attention maps C channels to Q queries
        x, attention_scores = self.cross_attn(x_tokenized)  # (B*S, Q, E)

        # Rearrange for temporal encoder: (B*S, Q, E) -> (B, S, Q*E)
        x = rearrange(x, "(B S) Q E -> B S (Q E)", B=B, S=num_patches)

        # Component 3: Temporal Encoder (RoPE transformer blocks)
        for blk in self.blocks:
            x = blk(x)  # (B, S, Q*E)

        # Final normalization
        x_latent = self.norm(x)  # (B, S, Q*E)

        # Component 4: Decoder Head
        if self.mode == "classification":
            # Classification: aggregate patches and classify
            x_classified = self.classifier(x_latent)  # (B, n_outputs)
            return x_classified
        else:
            # Reconstruction: decode to original signal (for pre-training)
            # TODO: Implement channel embeddings for reconstruction
            # channel_emb = self.channel_emb(channel_names)
            # decoder_queries = channel_locations_emb + channel_emb
            decoder_queries = channel_locations_emb  # Placeholder
            x_reconstructed = self.decoder_head(x_latent, decoder_queries)

            # Return reconstruction and original (for computing loss)
            return x_reconstructed  # , x_original, attention_scores

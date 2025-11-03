"""LUNA (Latent Unified Network Architecture) model.

This implementation is adapted from ETH Zurich's BioFoundation repository.

Reference:
    Döner, B., Ingolfsson, T. M., Benini, L., & Magimai-Doss, M. (2024).
    LUNA: A Model-Based Universal Analysis Framework for Large-Scale
    EEG Data. arXiv preprint arXiv:2510.22257.

The model architecture is preserved exactly as in the original implementation
to enable loading of pre-trained weights from HuggingFace Hub.

Original Authors: Berkay Döner, Thorir Mar Ingolfsson
Braindecode Adaptation: Bruno Aristimunha
"""

import math
from typing import Optional

import mne
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath, Mlp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from .base import EEGModuleMixin


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


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


SEED_PRETRAINING_CHANNEL_LIST = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]
TUEG_CHANNEL_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "A1-T3",
    "T4-A2",
]
SIENA_CHANNEL_LIST = [
    "FP1",
    "FP2",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FC1",
    "FC5",
    "CP1",
    "CP5",
    "F9",
    "FZ",
    "CZ",
    "PZ",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
    "FC2",
    "FC6",
    "CP2",
    "CP6",
    "F10",
]

all_channels = set()
for ds in [
    SEED_PRETRAINING_CHANNEL_LIST,
    TUEG_CHANNEL_LIST,
    SIENA_CHANNEL_LIST,
]:
    for ch in ds:
        all_channels.add(ch)
CHANNEL_NAMES_TO_IDX = {ch: i for i, ch in enumerate(sorted(all_channels))}
CHANNEL_IDX_TO_NAMES = {i: ch for ch, i in CHANNEL_NAMES_TO_IDX.items()}


def get_channel_indices(channel_names):
    indices = []
    for name in channel_names:
        indices.append(CHANNEL_NAMES_TO_IDX[name])
    return indices


def get_channel_names(channel_indices):
    names = []
    for idx in channel_indices:
        names.append(CHANNEL_IDX_TO_NAMES[idx])
    return names


def get_channel_locations(channel_names):
    if "-" in channel_names[0]:
        names = list(set([part for ch in channel_names for part in ch.split("-")]))
    else:
        names = channel_names
    ch_types = ["eeg"] * len(names)  # Channel types
    info = mne.create_info(ch_names=names, sfreq=256, ch_types=ch_types)
    info = info.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={"cb1": "POO7", "cb2": "POO8"},
    )
    locs = []
    for name in channel_names:
        if name in TUEG_CHANNEL_LIST:
            electrode1, electrode2 = name.split("-")
            loc1 = info.get_montage().get_positions()["ch_pos"][electrode1]
            loc2 = info.get_montage().get_positions()["ch_pos"][electrode2]
            locs.append(((loc1 + loc2) / 2))
        else:
            locs.append(info.get_montage().get_positions()["ch_pos"][name])
    return locs


class ChannelEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(ChannelEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(len(CHANNEL_NAMES_TO_IDX), embed_dim)

    def forward(self, indices):
        return self.embeddings(indices)

    def initialize_weights(self):
        torch.init.normal_(self.embeddings.weight, std=2.0)


class FrequencyFeatureEmbedder(nn.Module):
    """
    This class takes data that is of the form (B, C, T) and patches it
    along the time dimension (T) into patches of size P (patch_size).
    The output is of the form (B, C, S, P) where S = T // P.
    """

    def __init__(self, patch_size, embed_dim):
        super(FrequencyFeatureEmbedder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        in_features = 2 * (patch_size // 2 + 1)
        self.frequency_to_embed = Mlp(
            in_features=in_features,
            hidden_features=int(4 * in_features),
            out_features=embed_dim,
        )

    def forward(self, x):
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


class RotarySelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
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

    def forward(self, x):
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


class GEGLU(nn.Module):
    def __init__(self):
        super(GEGLU, self).__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class RotaryTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
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
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForwardBlock(
            dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=drop
        )

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchReconstructionHeadWithQueries(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_queries: int = 4,
    ):
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
        self.decoder_linear = Mlp(
            embed_dim, int(embed_dim * 4), input_dim, act_layer=nn.GELU, drop=0.0
        )  # nn.Linear(embed_dim, input_dim, bias=True)

    def forward(self, enc, decoder_queries):
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


class ClassificationHeadWithQueries(nn.Module):
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
        self.embed_dim = int(embed_dim * num_queries)
        self.reconstruction_shape = self.input_dim
        self.decoder_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads, batch_first=True, dropout=0.15
        )
        self.decoder_ffn = Mlp(
            in_features=self.embed_dim,
            hidden_features=int(self.embed_dim * 4),
            out_features=num_classes,
            act_layer=nn.GELU,
            drop=0.15,
        )

        self.learned_agg = nn.Parameter(
            torch.randn(1, 1, self.embed_dim), requires_grad=True
        )

    def forward(self, x):
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


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        num_queries,
        input_embed_dim,
        output_embed_dim,
        num_heads,
        dropout_p=0.1,
        ff_dim=2048,
        pre_norm=True,
    ):
        super(CrossAttentionBlock, self).__init__()
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
        # Note: Original uses Mlp(..., norm_layer=nn.LayerNorm) but timm 0.4.12 doesn't support it
        # For weight loading, we omit the normalization here to match parameter count
        self.ffn = Mlp(
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

    def initialize_weights(self):
        torch.nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x is the input with shape (batch_size*num_patches, num_channels, embed_dim)
        batch_size, num_channels, _ = x.size()
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


class PatchEmbedNetwork(nn.Module):
    def __init__(self, embed_dim=64, patch_size=40):
        super(PatchEmbedNetwork, self).__init__()
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

    def forward(self, x):
        """
        x: (B, C, T)
        output: (B, C*S, D) where S = T//patch_size, D = embed_dim
        """
        x = rearrange(x, "B C (S P) -> B (C S) P", P=self.patch_size)
        x = x.unsqueeze(1)
        x = self.proj_in(x)
        x = rearrange(x, "B E CS D -> B CS (D E)")
        return x


class LUNA(EEGModuleMixin, nn.Module):
    """LUNA (Latent Unified Network Architecture) model.

    LUNA is a topology-invariant EEG model that processes signals from varying
    numbers of channels using a channel-unification mechanism with learned queries.

    The architecture consists of:
    1. Patch Feature Extraction (temporal CNN + FFT-based features)
    2. Channel-Unification Module (cross-attention with learned queries)
    3. Patch-wise Temporal Encoder (RoPE-based transformer)
    4. Decoder Heads (classification or reconstruction)

    Parameters
    ----------
    n_outputs : int, optional
        Number of output classes. If 0, model operates in reconstruction mode.
        Default: 0 (reconstruction mode).
    n_chans : int, optional
        Number of EEG channels. Not used for parameter initialization but
        kept for braindecode compatibility. Default: None.
    n_times : int, optional
        Number of time samples. Not used for parameter initialization but
        kept for braindecode compatibility. Default: None.
    sfreq : float, optional
        Sampling frequency in Hz. Not used for parameter initialization but
        kept for braindecode compatibility. Default: None.
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
    .. [LUNA] Döner, B., Ingolfsson, T. M., Benini, L., & Magimai-Doss, M. (2024).
       LUNA: A Model-Based Universal Analysis Framework for Large-Scale EEG Data.
       arXiv preprint arXiv:2510.22257.
    """

    def __init__(
        self,
        n_outputs: Optional[int] = 0,
        n_chans: Optional[int] = None,
        n_times: Optional[int] = None,
        sfreq: Optional[float] = None,
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        drop_path: float = 0.0,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
        )

        # Map braindecode parameters to LUNA parameters
        num_classes = n_outputs if n_outputs else 0

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.patch_embed_size = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth

        self.patch_embed = PatchEmbedNetwork(
            embed_dim=self.embed_dim, patch_size=patch_size
        )
        self.freq_embed = FrequencyFeatureEmbedder(
            embed_dim=self.embed_dim, patch_size=patch_size
        )
        # Note: Original uses Mlp(..., norm_layer=nn.LayerNorm) but timm 0.4.12 doesn't support it
        # For weight loading, we omit the normalization here to match parameter count
        self.channel_location_embedder = Mlp(
            in_features=int(self.patch_embed_size),
            out_features=int(self.patch_embed_size),
            hidden_features=int(self.patch_embed_size * 2),
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cross_attn = CrossAttentionBlock(
            num_queries=num_queries,
            input_embed_dim=self.embed_dim,
            output_embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=int(mlp_ratio * self.embed_dim),
            pre_norm=True,
        )
        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=int(self.embed_dim * self.num_queries),
                    num_heads=int(self.num_heads * self.num_queries),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(int(self.embed_dim * self.num_queries))

        if num_classes == 0:  # reconstruction (pre-training)
            self.decoder_head = PatchReconstructionHeadWithQueries(
                input_dim=patch_size,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_queries=num_queries,
            )
            self.channel_emb = ChannelEmbeddings(self.embed_dim)
        else:  # classification
            self.classifier = ClassificationHeadWithQueries(
                input_dim=patch_size,
                num_queries=num_queries,
                embed_dim=self.embed_dim,
                num_classes=num_classes,
                num_heads=self.num_heads,
            )
            self.mask_token.requires_grad = (
                False  # no use of mask token for classification
            )

        self.initialize_weights()

    def initialize_weights(self):
        self.cross_attn.initialize_weights()
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def prepare_tokens(self, x_signal, channel_locations, mask=None):
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

    def forward(self, X, mask=None, channel_locations=None, channel_names=None):
        """Forward pass."""
        x_signal = X
        B, C, T = x_signal.shape

        if channel_locations is None:
            channel_locations = torch.randn(B, C, 3, device=x_signal.device)

        x, channel_locations_emb = self.prepare_tokens(
            x_signal, channel_locations, mask=mask
        )
        x, attention_scores = self.cross_attn(x)
        x = rearrange(x, "(B t) Q D -> B t (Q D)", B=B)
        num_patches = x.shape[1]

        for blk in self.blocks:
            x = blk(x)
        x_latent = self.norm(x)

        if self.num_classes > 0:
            return self.classifier(x_latent)
        else:
            channel_emb = self.channel_emb(channel_names)
            channel_emb = channel_emb.repeat(num_patches, 1, 1)
            decoder_queries = channel_locations_emb + channel_emb
            return self.decoder_head(x_latent, decoder_queries)

    @classmethod
    def from_pretrained(
        cls, variant="base", weights_path=None, n_outputs=None, **kwargs
    ):
        """Load a pretrained LUNA model.

        Parameters
        ----------
        variant : str, optional
            Model variant: 'base', 'large', or 'huge'. Default is 'base'.
        weights_path : str, optional
            Path to the weights file. If None, will look in default location:
            'thesis/luna-weights/LUNA_{variant}.safetensors'
        n_outputs : int, optional
            Number of output classes for classification. If None, loads for
            reconstruction (pretraining mode).
        **kwargs : dict
            Additional arguments passed to LUNA.__init__()

        Returns
        -------
        model : LUNA
            LUNA model with pretrained weights loaded.

        Examples
        --------
        >>> # Load pretrained base model for classification
        >>> model = LUNA.from_pretrained('base', n_outputs=4)
        >>>
        >>> # Load pretrained large model from custom path
        >>> model = LUNA.from_pretrained('large',
        ...                               weights_path='path/to/LUNA_large.safetensors',
        ...                               n_outputs=2)
        """
        from pathlib import Path

        from safetensors.torch import load_file

        # Model configurations
        configs = {
            "base": dict(embed_dim=64, num_queries=4, depth=8, num_heads=2),
            "large": dict(embed_dim=96, num_queries=6, depth=10, num_heads=2),
            "huge": dict(embed_dim=128, num_queries=8, depth=24, num_heads=2),
        }

        if variant.lower() not in configs:
            raise ValueError(
                f"variant must be one of {list(configs.keys())}, got {variant}"
            )

        config = configs[variant.lower()]

        # Update config with user-provided kwargs
        # Remove num_classes from config if present (we use n_outputs)
        for key in list(config.keys()):
            if key in kwargs:
                config[key] = kwargs.pop(key)

        # Determine weights path
        if weights_path is None:
            weights_path = (
                Path("thesis/luna-weights") / f"LUNA_{variant.lower()}.safetensors"
            )
        else:
            weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found at {weights_path}. "
                f"Please download weights from HuggingFace: ETH-MSRL/LUNA"
            )

        # Create model
        if n_outputs is not None:
            model = cls(n_outputs=n_outputs, patch_size=40, **config, **kwargs)
        else:
            # Pretraining mode - would need decoder_head
            raise NotImplementedError(
                "Reconstruction mode not yet implemented. "
                "Please specify n_outputs for classification."
            )

        # Load pretrained weights
        print(
            f"Loading pretrained LUNA-{variant.upper()} weights from {weights_path}..."
        )
        pretrained = load_file(str(weights_path))

        # Map channel_location_embedder keys (pretrained has Sequential wrapper)
        key_mapping = {
            "channel_location_embedder.0.fc1.weight": "channel_location_embedder.fc1.weight",
            "channel_location_embedder.0.fc1.bias": "channel_location_embedder.fc1.bias",
            "channel_location_embedder.0.fc2.weight": "channel_location_embedder.fc2.weight",
            "channel_location_embedder.0.fc2.bias": "channel_location_embedder.fc2.bias",
            "channel_location_embedder.0.norm.weight": "channel_location_embedder.norm.weight",
            "channel_location_embedder.0.norm.bias": "channel_location_embedder.norm.bias",
        }

        mapped_pretrained = {}
        for k, v in pretrained.items():
            mapped_pretrained[key_mapping.get(k, k)] = v

        # Load weights (strict=False because classifier head is randomly initialized)
        result = model.load_state_dict(mapped_pretrained, strict=False)

        print(f"✅ Loaded {len(pretrained) - len(result.unexpected_keys)} weights")
        if result.missing_keys:
            print(
                f"⚠️  {len(result.missing_keys)} keys not found in pretrained (using random init)"
            )

        return model
        return model

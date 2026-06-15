# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import DropPath


class _PatchEmbedding(nn.Module):
    """Conv + FFT + channel one-hot + depthwise time encoding (sum of 4 branches).

    Faithful to the released EEG-DINO PatchEmbedding, made device-agnostic and
    parametrized by ``num_channels`` and the ``proj_in`` conv widths.
    """

    def __init__(
        self,
        feature_size,
        num_channels,
        patch_size=200,
        conv_channels=(25, 25, 25),
        groups=5,
        drop_prob=0.1,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        c0, c1, c2 = conv_channels
        if c2 * 8 != feature_size:
            raise ValueError(
                f"conv_channels[-1] * 8 ({c2 * 8}) must equal feature_size "
                f"({feature_size}); the temporal conv reshape requires it."
            )
        self.time_encoding = nn.Sequential(
            nn.Conv2d(
                feature_size,
                feature_size,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                groups=feature_size,
            ),
        )
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, c0, (1, 49), (1, 25), (0, 24)),
            nn.GroupNorm(groups, c0),
            nn.GELU(),
            nn.Conv2d(c0, c1, (1, 3), (1, 1), (0, 1)),
            nn.GroupNorm(groups, c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, (1, 3), (1, 1), (0, 1)),
            nn.GroupNorm(groups, c2),
            nn.GELU(),
        )
        n_freq = patch_size // 2 + 1
        self.spectral_proj = nn.Sequential(
            nn.Linear(n_freq, feature_size), nn.Dropout(drop_prob)
        )
        self.channel_embedding = nn.Linear(num_channels, feature_size)

    def forward(self, x):
        batch_size, n_chans, n_patches, patch_size = x.shape

        x_time = x.contiguous().view(batch_size, 1, n_chans * n_patches, patch_size)
        patch_emb = self.proj_in(x_time)
        patch_emb = (
            patch_emb.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, n_chans, n_patches, self.feature_size)
        )

        x_freq = x.contiguous().view(batch_size * n_chans * n_patches, patch_size)
        spectral = torch.abs(torch.fft.rfft(x_freq, dim=-1, norm="forward"))
        spectral = spectral.contiguous().view(batch_size, n_chans, n_patches, -1)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        chan_idx = torch.arange(n_chans, device=x.device)
        one_hot = F.one_hot(chan_idx, num_classes=self.num_channels).float()
        chan_emb = self.channel_embedding(one_hot)  # (n_chans, D)
        chan_emb = chan_emb.unsqueeze(0).unsqueeze(2)  # (1, n_chans, 1, D)
        patch_emb = patch_emb + chan_emb

        time_emb = self.time_encoding(patch_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        patch_emb = patch_emb + time_emb
        return patch_emb


class _Mlp(nn.Module):
    """Feed-forward block (fc1 -> act -> fc2 -> drop); names match released keys."""

    def __init__(self, in_features, hidden_features, activation=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    """BEiT-style attention: fused qkv (bias=False) + separate q/v bias."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, n_tokens, dim = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch_size, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer block (LayerScale disabled, matching released config)."""

    def __init__(
        self, d_model, num_heads, dim_feedforward, activation=nn.GELU, drop_prob=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _Attention(
            d_model, num_heads=num_heads, attn_drop=drop_prob, proj_drop=drop_prob
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _Mlp(d_model, dim_feedforward, activation=activation, drop=drop_prob)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


EEGDINO_CONFIGS = {
    "small": dict(
        feature_size=200,
        num_layers=12,
        num_heads=8,
        dim_feedforward=512,
        patch_conv_channels=(25, 25, 25),
        patch_conv_groups=5,
    ),
    "medium": dict(
        feature_size=512,
        num_layers=16,
        num_heads=16,
        dim_feedforward=1024,
        patch_conv_channels=(64, 128, 64),
        patch_conv_groups=8,
    ),
    "large": dict(
        feature_size=1024,
        num_layers=24,
        num_heads=24,
        dim_feedforward=2048,
        patch_conv_channels=(128, 256, 128),
        patch_conv_groups=16,
    ),
}


class EEGDINO(EEGModuleMixin, nn.Module):
    r"""EEG-DINO model from Wang et al. (2025) [eegdino]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. rubric:: Architectural Overview

    EEG-DINO is a ViT-style encoder pre-trained with DINO-v2 hierarchical
    self-distillation. A time-frequency patch embedding turns 1-second EEG
    segments into tokens (convolutional temporal projection + FFT magnitude
    branch + learnable one-hot channel embedding + depthwise temporal encoding),
    which a stack of pre-norm transformer layers contextualizes. A learnable
    global token is inserted after the first layer. For classification, the
    patch tokens are pooled across channels then across time and passed to an
    MLP head (``final_layer``). Only the encoder is integrated here; the
    self-distillation pre-training is out of scope.

    .. versionadded:: 1.7

    Parameters
    ----------
    feature_size : int, optional
        Transformer embedding dimension ``D``. Default 200 (Small). Tied to the
        patch convolutions: ``patch_conv_channels[-1] * 8`` must equal it.
    num_layers : int, optional
        Number of transformer encoder layers. Default 12.
    num_heads : int, optional
        Number of attention heads. Default 8.
    dim_feedforward : int, optional
        Hidden size of the feed-forward block. Default 512.
    num_global_tokens : int, optional
        Number of learnable global tokens. Default 1.
    global_token_layer : int, optional
        1-based index of the encoder layer after which the global tokens are
        inserted. Default 1.
    patch_size : int, optional
        Samples per patch. Default 200 (1 s at 200 Hz). Locked to 200 for the
        released weights (the FFT branch uses ``patch_size // 2 + 1`` bins).
    num_channels : int, optional
        Size of the one-hot channel embedding (max supported channels).
        Default 19 (the released configuration).
    patch_conv_channels : tuple of int, optional
        Output widths of the three temporal convolutions in the patch
        embedding. Default ``(25, 25, 25)``.
    patch_conv_groups : int, optional
        GroupNorm groups in the patch convolutions. Default 5.
    head_hidden_divisors : tuple of int, optional
        The MLP head hidden sizes are ``D // d`` for each ``d``. Default ``(2, 4)``.
    head_drop_probs : tuple of float, optional
        Dropout after each MLP head hidden layer. Default ``(0.5, 0.3)``.
    activation : nn.Module, optional
        Activation class used throughout. Default ``nn.GELU``.
    drop_prob : float, optional
        Dropout / stochastic-depth probability in the encoder. Default 0.1.
    return_features : bool, optional
        If True, ``forward`` returns ``{"features", "cls_token"}``. Default False.
    return_encoder_output : bool, optional
        If True, ``final_layer`` is ``nn.Identity`` and ``forward`` returns the
        pooled encoder representation (for linear probing). Default False.

    References
    ----------
    .. [eegdino] Wang, X., Liu, X., Liu, X., Si, Q., Xu, Z., Li, Y., & Zhen, X.
       (2025). EEG-DINO: Learning EEG Foundation Models via Hierarchical
       Self-Distillation. MICCAI 2025.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        feature_size: int = 200,
        num_layers: int = 12,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        num_global_tokens: int = 1,
        global_token_layer: int = 1,
        patch_size: int = 200,
        num_channels: int = 19,
        patch_conv_channels: tuple[int, int, int] = (25, 25, 25),
        patch_conv_groups: int = 5,
        head_hidden_divisors: tuple[int, int] = (2, 4),
        head_drop_probs: tuple[float, float] = (0.5, 0.3),
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.1,
        return_features: bool = False,
        return_encoder_output: bool = False,
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

        if self._sfreq is not None and self.sfreq != 200:
            warn(
                f"EEG-DINO was trained at 200 Hz but sfreq={self.sfreq}. "
                "Inputs are not resampled; results may be unreliable.",
                UserWarning,
            )
        if self.n_chans > num_channels:
            raise ValueError(
                f"n_chans={self.n_chans} exceeds num_channels={num_channels}. "
                "Increase num_channels (incompatible with released weights) or "
                "reduce the channel set."
            )

        self.feature_size = feature_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_global_tokens = num_global_tokens
        self.global_token_layer = global_token_layer
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_conv_channels = patch_conv_channels
        self.patch_conv_groups = patch_conv_groups
        self.head_hidden_divisors = head_hidden_divisors
        self.head_drop_probs = head_drop_probs
        self.activation = activation
        self.drop_prob = drop_prob
        self.return_features = return_features
        self.return_encoder_output = return_encoder_output

        self.patch_embedding = _PatchEmbedding(
            feature_size=feature_size,
            num_channels=num_channels,
            patch_size=patch_size,
            conv_channels=patch_conv_channels,
            groups=patch_conv_groups,
            drop_prob=drop_prob,
        )
        self.encoder_layers = nn.ModuleList(
            _TransformerEncoderLayer(
                d_model=feature_size,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                activation=activation,
                drop_prob=drop_prob,
            )
            for _ in range(num_layers)
        )
        self.global_tokens = nn.Parameter(
            torch.zeros(1, num_global_tokens, feature_size)
        )
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        self.head_token_mlp = nn.Sequential(
            nn.Linear(feature_size, feature_size), activation()
        )
        self.head_time_mlp = nn.Sequential(
            nn.Linear(feature_size, feature_size), activation()
        )
        self._build_final_layer()

    def _build_final_layer(self):
        if self.return_encoder_output:
            self.final_layer = nn.Identity()
            return
        dim = self.feature_size
        hidden1 = dim // self.head_hidden_divisors[0]
        hidden2 = dim // self.head_hidden_divisors[1]
        self.final_layer = nn.Sequential(
            nn.Linear(dim, hidden1),
            self.activation(),
            nn.Dropout(self.head_drop_probs[0]),
            nn.Linear(hidden1, hidden2),
            self.activation(),
            nn.Dropout(self.head_drop_probs[1]),
            nn.Linear(hidden2, self.n_outputs),
        )

    def forward(self, x):  # replaced in Task 4
        return self.final_layer(
            torch.zeros(x.shape[0], self.feature_size, device=x.device)
        )

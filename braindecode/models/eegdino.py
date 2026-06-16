# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Code adapted from https://huggingface.co/eegdino/EEG-DINO
#
# License: BSD (3-clause)
from __future__ import annotations

from typing import Sequence
from warnings import warn

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import DropPath


class EEGDINO(EEGModuleMixin, nn.Module):
    r"""EEG-DINO from Wang et al. (2025) [eegdino]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://braindecode.org/dev/_static/model/eegdino.png
        :align: center
        :alt: EEG-DINO Architecture

    EEG-DINO is a ViT-style EEG foundation model pre-trained with DINO-v2
    hierarchical self-distillation. Only the *encoder* (plus a classification
    head) is integrated here; the self-distillation pre-training is out of scope.
    The forward path is, end to end:

    ``(batch, n_chans, n_times)`` → patchify → time-frequency embedding →
    decoupled positional embedding → transformer encoder (+ global token) →
    pooling head → ``(batch, n_outputs)``.

    .. rubric:: Step 1 -- Patchify

    The signal is divided by 100 (the amplitude scaling used during
    pre-training) and split along time into non-overlapping patches of
    ``patch_size`` samples (200 samples = 1 second at 200 Hz), giving one
    *token* per (channel, patch). Inputs whose length is not a multiple of
    ``patch_size`` are zero-padded with a warning.

    .. rubric:: Step 2 -- Time-Frequency Embedding (TFE)

    Each patch is embedded by summing two branches, exactly as in
    :class:`~braindecode.models.CBraMod` (the EEG-DINO paper reuses CBraMod's
    TFE): a **time-domain** branch of stacked grouped convolutions
    (``proj_in``), and a **frequency-domain** branch projecting the magnitude
    of the patch's real FFT (``spectral_proj``). The embedding dimension
    ``emb_dim`` is therefore *derived* from the convolution configuration, not
    set independently.

    .. rubric:: Step 3 -- Decoupled Positional Embedding (DPE)

    Where CBraMod uses a single convolutional positional encoding (ACPE),
    EEG-DINO *decouples* space and time and adds both to every token: a
    learnable **one-hot channel embedding** (``channel_embedding``, so input
    channel ``i`` maps to slot ``i``) and a **depthwise temporal convolution**
    over the patch axis (``time_encoding``).

    .. rubric:: Step 4 -- Transformer Encoder & Global Token

    Tokens are flattened to a single sequence and processed by ``n_layer``
    pre-norm transformer blocks (BEiT-style attention with separate query/value
    biases). A learnable ``global_tokens`` summary token is prepended after the
    ``global_token_layer``-th block and attends jointly with the patch tokens.

    .. rubric:: Step 5 -- Classification Head (``final_layer``)

    The patch tokens (global token excluded) are mean-pooled into a single
    ``emb_dim`` vector and mapped to ``n_outputs`` by a linear ``final_layer``
    (:class:`~braindecode.EEGClassifier` applies the softmax). With
    ``return_encoder_output=True`` the pooled representation is returned instead
    (linear probing).

    .. important::
       **Pre-trained Weights Available**

       Small and Medium encoders converted from the released checkpoints are
       hosted on the Hugging Face Hub, one repository per size (the head is
       re-initialized, so fine-tune or linear-probe before use)::

           from braindecode.models import EEGDINO

           model = EEGDINO.from_pretrained(
               "braindecode/eegdino-small-pretrained",  # or -medium-pretrained
               n_outputs=6,
               n_chans=19,
               sfreq=200,
           )

       The Small/Medium/Large architectures are also available in
       :data:`EEGDINO_CONFIGS`. Requires ``braindecode[hub]``.

    .. versionadded:: 1.7

    Parameters
    ----------
    patch_size : int, default=200
        Temporal patch size in samples (200 = 1 second at 200 Hz). Fixed at 200
        for the released weights (the FFT branch uses ``patch_size // 2 + 1`` bins).
    n_layer : int, default=12
        Number of transformer encoder layers.
    nhead : int, default=8
        Number of attention heads.
    dim_feedforward : int, default=512
        Hidden size of the transformer feed-forward block.
    channels_kernel_stride_padding_norm : sequence of tuple, optional
        Configuration of the time-domain convolutions in the patch embedding,
        as ``(out_channels, kernel, stride, padding, (groups, group_channels))``
        per layer. The embedding dimension is derived from this (see
        :class:`~braindecode.models.CBraMod`). Default is the EEG-DINO-Small /
        CBraMod configuration.
    num_channels : int, default=19
        Size of the one-hot channel embedding, i.e. the maximum number of input
        channels. Default 19 (the released configuration); ``n_chans`` must not
        exceed it.
    n_global_tokens : int, default=1
        Number of learnable global summary tokens.
    global_token_layer : int, default=1
        1-based index of the encoder layer after which the global tokens are
        inserted.
    activation : type[nn.Module], default=nn.GELU
        Activation used throughout.
    drop_prob : float, default=0.1
        Dropout / stochastic-depth probability in the encoder.
    return_features : bool, default=False
        If True, ``forward`` returns ``{"features", "cls_token"}``.
    return_encoder_output : bool, default=False
        If True, ``final_layer`` is :class:`~torch.nn.Identity` and ``forward``
        returns the pooled encoder representation (linear probing).

    References
    ----------
    .. [eegdino] Wang, X., Liu, X., Liu, X., Si, Q., Xu, Z., Li, Y., & Zhen, X.
       (2025). EEG-DINO: Learning EEG Foundation Models via Hierarchical
       Self-Distillation. In Medical Image Computing and Computer Assisted
       Intervention (MICCAI 2025).
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        patch_size: int = 200,
        n_layer: int = 12,
        nhead: int = 8,
        dim_feedforward: int = 512,
        channels_kernel_stride_padding_norm: Sequence[
            tuple[int, int, int, int, tuple[int, int]]
        ] = (
            (25, 49, 25, 24, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
        ),
        num_channels: int = 19,
        n_global_tokens: int = 1,
        global_token_layer: int = 1,
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
                f"EEG-DINO was trained at 200 Hz but sfreq={self.sfreq}. Inputs "
                "are not resampled internally; results may be unreliable.",
                UserWarning,
            )
        if self.n_chans > num_channels:
            raise ValueError(
                f"n_chans={self.n_chans} exceeds num_channels={num_channels}; "
                "raise num_channels (breaks released-weight compatibility) or "
                "use fewer channels."
            )

        self.patch_size = patch_size
        self.n_global_tokens = n_global_tokens
        self.global_token_layer = global_token_layer
        self.activation = activation
        self.return_features = return_features
        self.return_encoder_output = return_encoder_output

        self.patch_embedding = _PatchEmbedding(
            patch_size=patch_size,
            channels_kernel_stride_padding_norm=channels_kernel_stride_padding_norm,
            num_channels=num_channels,
            drop_prob=drop_prob,
        )
        self.emb_dim = self.patch_embedding.emb_dim

        self.encoder_layers = nn.ModuleList(
            _TransformerEncoderLayer(
                self.emb_dim, nhead, dim_feedforward, activation, drop_prob
            )
            for _ in range(n_layer)
        )
        self.global_tokens = nn.Parameter(torch.zeros(1, n_global_tokens, self.emb_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        self.final_layer = self._make_head(self.n_outputs)

    def _make_head(self, n_outputs):
        if self.return_encoder_output:
            return nn.Identity()
        return nn.Linear(self.emb_dim, n_outputs)

    def reset_head(self, n_outputs):
        """Replace ``final_layer`` to output ``n_outputs`` classes."""
        self._n_outputs = n_outputs
        self.final_layer = self._make_head(n_outputs)

    def _patchify(self, x):
        """``(batch, n_chans, n_times)`` -> scaled ``(batch, n_chans, n_patches, patch_size)``."""
        n_times = x.shape[-1]
        if n_times % self.patch_size:
            pad = self.patch_size - n_times % self.patch_size
            x = F.pad(x, (0, pad))
            warn(
                f"n_times={n_times} is not a multiple of patch_size="
                f"{self.patch_size}; zero-padded by {pad} samples.",
                UserWarning,
            )
        x = x / 100.0  # amplitude scaling used during EEG-DINO pre-training
        return rearrange(x, "b c (n p) -> b c n p", p=self.patch_size)

    def forward(self, x, return_features: bool | None = None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, n_chans, n_times)`` or pre-patchified
            ``(batch, n_chans, n_patches, patch_size)``.
        return_features : bool, optional
            Overrides ``self.return_features`` for this call.

        Returns
        -------
        torch.Tensor or dict
            Logits ``(batch, n_outputs)``; or, with features,
            ``{"features": (batch, n_chans * n_patches, emb_dim),
            "cls_token": (batch, emb_dim)}``.
        """
        if return_features is None:
            return_features = self.return_features
        if x.ndim == 3:
            x = self._patchify(x)

        patch_emb = self.patch_embedding(x)  # (batch, n_chans, n_patches, emb_dim)
        tokens = rearrange(patch_emb, "b c n d -> b (c n) d")

        global_tokens = self.global_tokens.expand(tokens.shape[0], -1, -1)
        for i, layer in enumerate(self.encoder_layers):
            tokens = layer(tokens)
            if i + 1 == self.global_token_layer:
                tokens = torch.cat([global_tokens, tokens], dim=1)

        global_out = tokens[:, : self.n_global_tokens]
        patch_out = tokens[:, self.n_global_tokens :]
        if return_features:
            return {"features": patch_out, "cls_token": global_out.mean(dim=1)}

        pooled = patch_out.mean(dim=1)  # average over channel-patch tokens
        if self.return_encoder_output:
            return pooled
        return self.final_layer(pooled)


#: Architecture presets. Small and Medium have released weights; Large does not.
EEGDINO_CONFIGS = {
    "small": dict(
        n_layer=12,
        nhead=8,
        dim_feedforward=512,
        channels_kernel_stride_padding_norm=(
            (25, 49, 25, 24, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
        ),
    ),
    "medium": dict(
        n_layer=16,
        nhead=16,
        dim_feedforward=1024,
        channels_kernel_stride_padding_norm=(
            (64, 49, 25, 24, (8, 64)),
            (128, 3, 1, 1, (8, 128)),
            (64, 3, 1, 1, (8, 64)),
        ),
    ),
    "large": dict(
        n_layer=24,
        nhead=24,
        dim_feedforward=2048,
        channels_kernel_stride_padding_norm=(
            (128, 49, 25, 24, (16, 128)),
            (256, 3, 1, 1, (16, 256)),
            (128, 3, 1, 1, (16, 128)),
        ),
    ),
}


class _PatchEmbedding(nn.Module):
    """Time-frequency patch embedding with a decoupled positional embedding.

    The time-domain (``proj_in``) and frequency-domain (``spectral_proj``)
    branches are CBraMod's TFE (see :class:`~braindecode.models.CBraMod`); the
    one-hot ``channel_embedding`` and depthwise ``time_encoding`` are EEG-DINO's
    decoupled positional embedding.
    """

    def __init__(
        self,
        patch_size,
        channels_kernel_stride_padding_norm,
        num_channels,
        drop_prob=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels_kernel_stride_padding_norm = channels_kernel_stride_padding_norm
        self.num_channels = num_channels

        in_channels, conv_layers = 1, []
        for (
            out_channels,
            kernel,
            stride,
            padding,
            norm,
        ) in channels_kernel_stride_padding_norm:
            conv_layers += [
                nn.Conv2d(
                    in_channels, out_channels, (1, kernel), (1, stride), (0, padding)
                ),
                nn.GroupNorm(*norm),
                nn.GELU(),
            ]
            in_channels = out_channels
        self.proj_in = nn.Sequential(*conv_layers)

        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, self.emb_dim), nn.Dropout(drop_prob)
        )
        self.channel_embedding = nn.Linear(num_channels, self.emb_dim)
        # Sequential wrapper keeps the released-checkpoint key ``time_encoding.0.*``.
        self.time_encoding = nn.Sequential(
            nn.Conv2d(
                self.emb_dim, self.emb_dim, (1, 5), (1, 1), (0, 2), groups=self.emb_dim
            )
        )

    @property
    def emb_dim(self):
        """Embedding dimension implied by the convolution configuration."""
        reduced = self.patch_size
        for _, kernel, stride, padding, _ in self.channels_kernel_stride_padding_norm:
            reduced = (reduced + 2 * padding - kernel) // stride + 1
        return self.channels_kernel_stride_padding_norm[-1][0] * reduced

    def forward(self, x):
        # x: (batch, n_chans, n_patches, patch_size)
        n_chans = x.shape[1]

        # Time-frequency embedding (CBraMod TFE): conv branch + FFT-magnitude branch.
        time_tokens = self.proj_in(rearrange(x, "b c n p -> b 1 (c n) p"))
        time_tokens = rearrange(time_tokens, "b d (c n) q -> b c n (d q)", c=n_chans)
        spectrum = torch.fft.rfft(x, dim=-1, norm="forward").abs()
        patch_emb = time_tokens + self.spectral_proj(spectrum)

        # Decoupled positional embedding: one-hot channel + depthwise temporal conv.
        channel_ids = torch.arange(n_chans, device=x.device)
        channel_emb = self.channel_embedding(
            F.one_hot(channel_ids, self.num_channels).float()
        )
        patch_emb = patch_emb + rearrange(channel_emb, "c d -> 1 c 1 d")
        temporal_emb = self.time_encoding(rearrange(patch_emb, "b c n d -> b d c n"))
        return patch_emb + rearrange(temporal_emb, "b d c n -> b c n d")


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer block (no LayerScale, matching the released config)."""

    def __init__(
        self, emb_dim, nhead, dim_feedforward, activation=nn.GELU, drop_prob=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = _Attention(emb_dim, nhead, attn_drop=drop_prob, proj_drop=drop_prob)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = _MLP(emb_dim, dim_feedforward, activation=activation, drop=drop_prob)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _Attention(nn.Module):
    """BEiT-style attention: fused ``qkv`` without bias plus separate q/v biases.

    ``head_dim`` is decoupled from ``emb_dim`` so the embedding dimension need
    not be divisible by ``nhead`` (required by the Large preset, 1024/24).
    """

    def __init__(self, emb_dim, nhead=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.nhead = nhead
        head_dim = emb_dim // nhead
        all_head_dim = head_dim * nhead
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(emb_dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        query, key, value = rearrange(
            qkv, "b n (three h d) -> three b h n d", three=3, h=self.nhead
        )
        attn = (query * self.scale @ key.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = rearrange(attn @ value, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(out))


class _MLP(nn.Module):
    """Feed-forward block; ``fc1``/``fc2`` names match the released checkpoint."""

    def __init__(self, emb_dim, hidden_dim, activation=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

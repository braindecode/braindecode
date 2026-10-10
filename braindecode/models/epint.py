# Authors: Runkai Zhang <271013216@qq.com>
#          Julien Gadonneix (braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/RunKZhang/EpiNT (MIT).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

from braindecode.functional import rotate_pairs
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock


class EpiNT(EEGModuleMixin, nn.Module, license="mit"):
    r"""EpiNT from Zhang et al. (2025) [EpiNT2025]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://raw.githubusercontent.com/RunKZhang/EpiNT/master/assets/ModelStructure.png
       :align: center
       :alt: EpiNT architecture

    .. rubric:: Architecture Overview

    EpiNT (Epilepsy Neurophysiological Transformer) is pretrained on 2,700 hours
    of scalp EEG and intracranial EEG (ECoG, sEEG) from 1,199 patients. It is
    **channel independent**: every channel is encoded alone, so one checkpoint
    serves any montage and modality.

    1. ``instance_norm`` z-scores each channel over time (affine, one scale and
       one offset shared by all channels).
    2. ``patch_embedding`` projects non-overlapping patches of ``patch_size``
       samples to ``embed_dim``, and a learned ``cls_token`` is prepended.
    3. ``encoder`` is a stack of post-norm Transformer layers with rotary
       position embeddings (RoPE) in the self-attention.
    4. The class token of each channel is averaged over channels and
       ``final_layer`` (dropout and one linear layer) classifies it.

    The pretraining mask token and frequency-domain mapping quantizer are not
    used downstream, so they are not part of this module.

    .. rubric:: Fixed sequence length

    Each encoder ``LayerNorm`` normalises over the token **and** feature axes
    (``nn.LayerNorm([n_patches + 1, embed_dim])``), as upstream. The number of
    patches, i.e. ``n_times // patch_size``, is therefore fixed at
    construction, and the input must have exactly ``n_times`` samples. The
    released checkpoint uses ``n_times=3072``: 12 s at 256 Hz for scalp EEG and
    3 s at 1024 Hz for iEEG, the two rates of the pretraining corpus.

    .. important::
       **Pretrained weights.** The authors release the pretrained backbone
       (``weights/representations.bin``,
       `License <https://github.com/RunKZhang/EpiNT/blob/master/LICENSE>`_).
       The default hyper-parameters are those of that checkpoint, and its 102
       tensors load through :attr:`mapping` with only the pretraining mask
       token unused::

           state = torch.load("representations.bin", map_location="cpu")
           model = EpiNT(n_chans=1, n_outputs=2, n_times=3072)
           model.load_state_dict(state, strict=False)

       With ``n_chans=1``, the logits and the class token match the upstream
       ``EpiNT`` in classification mode within 1e-5.

    .. versionadded:: 1.9

    Parameters
    ----------
    patch_size : int, optional
        Samples per patch. Default 256.
    embed_dim : int, optional
        Transformer width. Default 512.
    n_layers : int, optional
        Number of Transformer layers. Default 6.
    n_heads : int, optional
        Number of attention heads; ``embed_dim // n_heads`` must be even for
        RoPE. Default 8.
    ffn_dim : int, optional
        Inner dimension of the feed-forward blocks. Default 2048.
    drop_prob : float, optional
        Dropout in the Transformer layers. Default 0.1.
    head_drop_prob : float, optional
        Dropout before ``final_layer``. Default 0.1.
    rope_theta : float, optional
        RoPE base frequency. Default 10000.
    activation : type[nn.Module], optional
        Feed-forward activation. Default ``nn.ReLU``, as pretrained.

    References
    ----------
    .. [EpiNT2025] Zhang, R., Yu, H., Gan, J. Q. and Wang, H., 2025.
       Cross-modal epileptic signal harmonization: Frequency domain mapping
       quantization for pre-training a unified neurophysiological transformer.
       arXiv:2506.17068.
       Code: https://github.com/RunKZhang/EpiNT
    """

    def __init__(
        self,
        patch_size: int = 256,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        drop_prob: float = 0.1,
        head_drop_prob: float = 0.1,
        rope_theta: float = 10000.0,
        activation: type[nn.Module] = nn.ReLU,
        # braindecode signal parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
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

        if embed_dim % n_heads or (embed_dim // n_heads) % 2:
            raise ValueError(
                f"embed_dim={embed_dim} must split into n_heads={n_heads} heads "
                "of even width for RoPE."
            )
        self.patch_size = patch_size
        self.n_patches = self.n_times // patch_size
        if self.n_patches < 1:
            raise ValueError(
                f"n_times={self.n_times} is shorter than one patch "
                f"(patch_size={patch_size})."
            )
        n_tokens = self.n_patches + 1

        self.merge_channels = Rearrange("batch chans times -> (batch chans) 1 times")
        self.instance_norm = nn.InstanceNorm1d(1, affine=True)
        self.patchify = Rearrange(
            "batch 1 (patches size) -> batch patches size", size=patch_size
        )
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.cls_token, std=0.1)
        # Interleaved-pair RoPE angles, as ``precompute_freqs_cis`` upstream.
        head_dim = embed_dim // n_heads
        inv_freq = 1.0 / rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(n_tokens).float(), inv_freq)
        angles = angles.repeat_interleave(2, dim=-1)  # (tokens, head_dim)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.encoder = nn.ModuleList(
            [
                _EpiNTLayer(
                    n_tokens, embed_dim, n_heads, ffn_dim, drop_prob, activation
                )
                for _ in range(n_layers)
            ]
        )
        self.split_channels = Rearrange(
            "(batch chans) dim -> batch chans dim", chans=self.n_chans
        )
        self.pool = Reduce("batch chans dim -> batch dim", "mean")
        self.final_layer = nn.Sequential(
            nn.Dropout(head_drop_prob), nn.Linear(embed_dim, self.n_outputs)
        )
        nn.init.kaiming_uniform_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

        self.mapping = {
            "norm.weight": "instance_norm.weight",
            "norm.bias": "instance_norm.bias",
            "embed.proj.weight": "patch_embedding.weight",
            "embed.proj.bias": "patch_embedding.bias",
            "embed.cls_embed": "cls_token",
            "head.weight": "final_layer.1.weight",
            "head.bias": "final_layer.1.bias",
        }
        # Upstream calls the encoder ``transformer_encoder`` and the FFN linears
        # ``linear1``/``linear2`` (here ``FeedForwardBlock`` children 0 and 3).
        for key in self.encoder.state_dict():
            upstream = key.replace("ffn.0.", "ffn.linear1.")
            upstream = upstream.replace("ffn.3.", "ffn.linear2.")
            self.mapping[f"transformer_encoder.{upstream}"] = f"encoder.{key}"

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer[1]
        head = nn.Linear(old.in_features, n_outputs).to(old.weight)
        nn.init.kaiming_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        self.final_layer[1] = head

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Classify a batch of signals.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If ``True``, return ``{"features": pooled, "cls_token": cls}``:
            ``cls`` holds the class token of every channel, shape
            ``(batch, n_chans, embed_dim)``, and ``pooled`` is its mean over
            channels. A scripted model ignores this flag.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape ``(batch, n_outputs)``, or the feature dict.
        """
        if x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(
                f"EpiNT was built for (n_chans, n_times)=({self.n_chans}, "
                f"{self.n_times}) and got {tuple(x.shape[1:])}; its LayerNorms "
                "fix the number of patches."
            )
        x = x[..., : self.n_patches * self.patch_size]
        z = self.instance_norm(self.merge_channels(x))  # (batch*chans, 1, times)
        z = self.patch_embedding(self.patchify(z))  # (batch*chans, patches, dim)
        cls = self.cls_token.expand(z.shape[0], 1, -1).to(z)
        z = torch.cat([cls, z], dim=1)  # (batch*chans, tokens, dim)
        cos, sin = self.rope_cos.to(z), self.rope_sin.to(z)
        for layer in self.encoder:
            z = layer(z, cos, sin)
        cls_token = self.split_channels(z[:, 0])  # (batch, chans, dim)
        pooled = self.pool(cls_token)
        logits = self.final_layer(pooled)
        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {"features": pooled, "cls_token": cls_token}
        return logits


class _RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE on queries and keys."""

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_concat = nn.Linear(embed_dim, embed_dim)
        self.split_heads = Rearrange(
            "batch tokens (heads dim) -> batch heads tokens dim", heads=n_heads
        )
        self.merge_heads = Rearrange(
            "batch heads tokens dim -> batch tokens (heads dim)"
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        q = q * cos + rotate_pairs(q) * sin
        k = k * cos + rotate_pairs(k) * sin
        out = F.scaled_dot_product_attention(q, k, v)  # scale 1 / sqrt(head_dim)
        return self.w_concat(self.merge_heads(out))


class _EpiNTLayer(nn.Module):
    """Post-norm Transformer layer; the LayerNorms span tokens and features."""

    def __init__(
        self,
        n_tokens: int,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.attention = _RoPEAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm([n_tokens, embed_dim])
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = FeedForwardBlock(
            emb_size=embed_dim,
            expansion=1,
            drop_p=drop_prob,
            activation=activation,
            hidden_features=ffn_dim,
            output_drop_p=drop_prob,
        )
        self.norm2 = nn.LayerNorm([n_tokens, embed_dim])
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.attention(x, cos, sin)))
        return self.norm2(x + self.dropout2(self.ffn(x)))

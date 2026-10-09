# Authors: Hui Zheng (original implementation)
#          Adam Mounir <am91ris@gmail.com> (braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/liulab-repository/Du-IN (MIT).
"""Du-IN (Zheng et al., NeurIPS 2024).

* paper: https://arxiv.org/abs/2405.11459
* code: https://github.com/liulab-repository/Du-IN
* weights: https://huggingface.co/datasets/liulab-repository/Du-IN
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from braindecode.functional import sinusoidal_positional_encoding
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import warn_if_sfreq_differs


class DuIN(EEGModuleMixin, nn.Module, license="mit"):
    r"""Du-IN from Zheng et al. (2024) [DuIN2024]_.

    :bdg-danger:`Foundation Model` :bdg-success:`Convolution`
    :bdg-info:`Attention/Transformer`

    .. rubric:: Architecture Overview

    Du-IN decodes speech from intracranial (sEEG) signals with **region-level**
    tokens: instead of one token per channel, the few channels recorded in a
    language-related region (ventral sensorimotor cortex, superior temporal
    gyrus) are fused into a single token per 100 ms patch [DuIN2024]_.

    1. Project the channels linearly into a small hidden "neural" space.
    2. Cut the signal into non-overlapping patches and encode each patch with a
       stack of strided 1D convolutions into one patch embedding.
    3. Add fixed sinusoidal temporal embeddings and run a Transformer encoder
       over the patch sequence.
    4. Flatten the encoded sequence and classify it with a one-hidden-layer MLP.

    .. rubric:: Macro Components

    ``DuIN.spatial_projection``
        **Operations.** A linear layer from the ``n_chans`` input channels to
        ``n_neural`` hidden channels, applied at every time sample.

        **Role.** Fuses the channels of the region into a common neural space.
        Its weights give the per-channel contribution scores used by the
        authors to select about 10 channels per subject.

    ``DuIN.spatial_encoder``
        **Operations.** Each patch of ``patch_size`` samples goes through
        ``Conv1d`` + ``BatchNorm1d`` blocks (``n_filters``, ``kernel_sizes``,
        ``strides``); the output, time-major, is flattened into one
        ``embed_dim``-dimensional patch embedding.

        **Role.** Turns a patch of raw signal into a region-level token.

    ``DuIN.encoder``
        **Operations.** Post-norm Transformer layers whose attention applies
        layer normalisation to the queries and keys of each head (as LaBraM),
        followed by a ReLU feed-forward block.

        **Role.** Models the dynamics across the patch sequence.

    ``DuIN.head_hidden`` and ``DuIN.final_layer``
        **Operations.** The encoded ``(n_patches, embed_dim)`` sequence is
        flattened, mapped to ``head_hidden_dim`` units with a ReLU, then to
        ``n_outputs`` by a linear layer.

        **Role.** The upstream "label prediction head" of the Du-IN CLS model.

    .. rubric:: Pretraining

    Du-IN is pretrained per subject in two stages: a VQ-VAE learns a discrete
    codebook of patch embeddings, then a masked model (Du-IN MAE) predicts the
    codebook indices of 50% masked patches. Fine-tuning ("Du-IN (mae)", 62.70%
    top-1 accuracy over the 61 words, averaged over 12 subjects) loads the
    encoder of the MAE and trains it with the head above. Only the
    classification model is ported here: the VQ-VAE, the codebook and the MAE
    token head are not.

    .. rubric:: Differences from the reference

    - The upstream head ends with a sigmoid before a cross-entropy loss. The
      port returns the logits, as every braindecode classifier.
    - The upstream subject layer maps a one-hot subject id to a
      channel-to-neural matrix. The released checkpoints are single-subject,
      so it is a plain linear layer (``spatial_projection``) here; the
      multi-subject variant ("Du-IN (poms)") is not ported.
    - The paper describes the patch convolutions as depthwise; the released
      code and checkpoints use full ``Conv1d`` layers, which the port follows.
    - The temporal embedding table is rebuilt from ``n_times`` and not stored
      in the state dict (it has no parameters).

    .. important::
       **Pre-trained weights.** The authors release one Du-IN MAE checkpoint
       per subject of their 61-word dataset (Hugging Face dataset
       ``liulab-repository/Du-IN``, ``pretrains/duin/<subject>/mae/model/
       checkpoint-399.pth``, CC BY 4.0). Each was trained on that subject's 10
       selected channels at 1000 Hz, so it only fits that subject's montage.
       ``load_state_dict`` accepts the upstream keys directly; the head is not
       pretrained::

           model = DuIN(n_chans=10, n_outputs=61, n_times=3000, sfreq=1000)
           state = torch.load(
               "checkpoint-399.pth", map_location="cpu", weights_only=True
           )
           model.load_state_dict(state, strict=False)

       The encoder then matches the upstream ``duin_cls`` model to float
       precision on identical inputs.

    .. versionadded:: 1.8.2

    Parameters
    ----------
    patch_size : int, optional
        Number of time samples per patch. Default 100 (100 ms at 1000 Hz), as
        the released model. ``n_times`` must be a multiple of it.
    n_neural : int, optional
        Number of hidden channels of ``spatial_projection``. Default 16.
    n_filters : tuple of int, optional
        Output channels of each patch convolution. Default ``(128, 128, 16)``.
    kernel_sizes : tuple of int, optional
        Kernel size of each patch convolution. Default ``(19, 3, 3)``.
    strides : tuple of int, optional
        Stride of each patch convolution. Default ``(10, 1, 1)``. Their product
        must divide ``patch_size``.
    n_layers : int, optional
        Number of Transformer layers. Default 8.
    n_heads : int, optional
        Number of attention heads. Default 8.
    head_dim : int, optional
        Dimension of each attention head. Default 64 (so the attention width,
        512, is larger than ``embed_dim``, as upstream).
    ffn_dim : int, optional
        Inner dimension of the feed-forward blocks. Default 320.
    head_hidden_dim : int, optional
        Hidden units of the classification head. Default 128. ``0`` gives a
        single linear layer on the flattened sequence.
    activation : type[nn.Module], optional
        Activation of the feed-forward blocks and of the head. Default
        :class:`~torch.nn.ReLU`, as upstream.
    attn_drop_prob : float, optional
        Dropout on the attention weights. Default 0.2.
    drop_prob : float, optional
        Dropout after the first feed-forward layer. Default 0.2.

    References
    ----------
    .. [DuIN2024] Zheng, H., Wang, H.-T., Jiang, W.-B., Chen, Z.-T., He, L.,
       Lin, P.-Y., Wei, P.-H., Zhao, G.-G., & Liu, Y.-Z. (2024). Du-IN:
       Discrete units-guided mask modeling for decoding speech from
       intracranial neural signals. Advances in Neural Information Processing
       Systems 37 (NeurIPS 2024). arXiv:2405.11459. Code:
       https://github.com/liulab-repository/Du-IN (MIT).
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
        patch_size: int = 100,
        n_neural: int = 16,
        n_filters: tuple[int, ...] = (128, 128, 16),
        kernel_sizes: tuple[int, ...] = (19, 3, 3),
        strides: tuple[int, ...] = (10, 1, 1),
        n_layers: int = 8,
        n_heads: int = 8,
        head_dim: int = 64,
        ffn_dim: int = 320,
        head_hidden_dim: int = 128,
        activation: type[nn.Module] = nn.ReLU,
        attn_drop_prob: float = 0.2,
        drop_prob: float = 0.2,
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ):
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

        n_filters, kernel_sizes, strides = (
            tuple(n_filters),
            tuple(kernel_sizes),
            tuple(strides),
        )
        if not len(n_filters) == len(kernel_sizes) == len(strides) >= 1:
            raise ValueError(
                "n_filters, kernel_sizes and strides must have the same, "
                f"non-zero length; got {len(n_filters)}, {len(kernel_sizes)} "
                f"and {len(strides)}."
            )
        total_stride = math.prod(strides)
        if patch_size % total_stride:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by the product of "
                f"strides ({total_stride})."
            )
        if self.n_times % patch_size or self.n_times < patch_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be a positive multiple of "
                f"patch_size ({patch_size}): it is not divisible."
            )
        warn_if_sfreq_differs("DuIN", self._sfreq, 1000)

        self.patch_size = patch_size
        self.n_neural = n_neural
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.head_hidden_dim = head_hidden_dim
        self.attn_drop_prob = attn_drop_prob
        self.drop_prob = drop_prob
        self.n_patches = self.n_times // patch_size
        # Time-major flattening of the last convolution: (frames, filters).
        self.embed_dim = n_filters[-1] * (patch_size // total_stride)
        if self.embed_dim % 2:
            raise ValueError(
                f"The patch embedding size ({self.embed_dim}) must be even for "
                "the sinusoidal temporal embedding."
            )

        self.spatial_projection = nn.Linear(self.n_chans, n_neural)
        self.to_patches = Rearrange(
            "batch neural (patches time) -> (batch patches) neural time",
            time=patch_size,
        )
        blocks = []
        in_channels = n_neural
        for out_channels, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
            blocks.append(
                _DuINConvBlock(in_channels, out_channels, kernel_size, stride)
            )
            in_channels = out_channels
        self.spatial_encoder = nn.Sequential(*blocks)
        self.to_tokens = Rearrange(
            "(batch patches) filters frames -> batch patches (frames filters)",
            patches=self.n_patches,
        )
        self.register_buffer(
            "time_embedding",
            sinusoidal_positional_encoding(self.n_patches, self.embed_dim),
            persistent=False,
        )
        self.encoder = nn.ModuleList(
            [
                _DuINTransformerLayer(
                    embed_dim=self.embed_dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ffn_dim=ffn_dim,
                    activation=activation,
                    attn_drop_prob=attn_drop_prob,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )
        self.flatten = nn.Flatten(start_dim=1)
        n_features = self.n_patches * self.embed_dim
        if head_hidden_dim:
            self.head_hidden = nn.Sequential(
                nn.Linear(n_features, head_hidden_dim), activation()
            )
            n_features = head_hidden_dim
        else:
            self.head_hidden = nn.Identity()
        self.final_layer = nn.Linear(n_features, self.n_outputs)

        self._init_weights()

    def _init_weights(self) -> None:
        """Upstream initialisation (truncated normal, depth-scaled Transformer)."""
        nn.init.trunc_normal_(self.spatial_projection.weight, std=0.02)
        nn.init.zeros_(self.spatial_projection.bias)
        for depth, layer in enumerate(self.encoder, start=1):
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
                    module.weight.data.div_(math.sqrt(2.0 * depth))
        for module in [*self.head_hidden.modules(), self.final_layer]:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Also accept an upstream Du-IN checkpoint (``duin_mae``/``duin_cls``).

        Upstream keys are renamed to this implementation. The pretraining
        parts (``mask_emb``, the VQ codebook, the MAE token head), the fixed
        temporal table and the upstream classification head are dropped, so
        load such a checkpoint with ``strict=False``.
        """
        if "subj_block.subj_layer.W.weight" in state_dict:
            state_dict = _convert_upstream_state_dict(state_dict, self.n_neural)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        head = nn.Linear(self.final_layer.in_features, n_outputs)
        self.final_layer = head.to(self.final_layer.weight)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Decode a batch of signals.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If ``True``, return the flattened encoded sequence the head
            consumes, ``(batch, n_patches * embed_dim)``, as
            ``{"features": features, "cls_token": None}`` (braindecode
            foundation-model convention; Du-IN has no class token). A
            TorchScript-compiled model returns the logits instead.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            when ``return_features`` is set.
        """
        if x.shape[1] != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {x.shape[1]}.")
        if x.shape[-1] != self.n_times:
            raise ValueError(
                f"DuIN was configured for {self.n_times} time samples, "
                f"but received {x.shape[-1]}."
            )
        # 1. channels -> neural space: (batch, n_neural, n_times)
        h = self.spatial_projection(x.transpose(1, 2)).transpose(1, 2)
        # 2. one token per patch: (batch, n_patches, embed_dim)
        tokens = self.to_tokens(self.spatial_encoder(self.to_patches(h)))
        # 3. temporal embedding and Transformer encoder
        z = tokens + self.time_embedding.to(tokens.dtype)
        for layer in self.encoder:
            z = layer(z)
        # 4. flatten and classify
        features = self.flatten(z)
        logits = self.final_layer(self.head_hidden(features))
        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {"features": features, "cls_token": None}  # nosec B105
        return logits


class _DuINConvBlock(nn.Module):
    """``Conv1d`` + ``BatchNorm1d`` over the samples of a patch.

    Strided convolutions pad by ``(kernel_size - 1) // 2`` and the others use
    ``"same"`` padding, as upstream ``PatchTokenizer``.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()
        padding: int | str = "same" if stride == 1 else (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _DuINAttention(nn.Module):
    """Multi-head self-attention with layer-normalised queries and keys."""

    def __init__(
        self, embed_dim: int, n_heads: int, head_dim: int, attn_drop_prob: float
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_drop_prob = attn_drop_prob
        inner_dim = n_heads * head_dim
        self.q = nn.Linear(embed_dim, inner_dim)
        self.k = nn.Linear(embed_dim, inner_dim)
        self.v = nn.Linear(embed_dim, inner_dim)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.proj = nn.Linear(inner_dim, embed_dim)
        self.split_heads = Rearrange(
            "batch seq (heads dim) -> batch heads seq dim", heads=n_heads
        )
        self.merge_heads = Rearrange("batch heads seq dim -> batch seq (heads dim)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(self.split_heads(self.q(x)))
        k = self.k_norm(self.split_heads(self.k(x)))
        v = self.split_heads(self.v(x))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_prob if self.training else 0.0
        )
        return self.proj(self.merge_heads(out))


class _DuINTransformerLayer(nn.Module):
    """Post-norm Transformer layer (attention, then feed-forward)."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        head_dim: int,
        ffn_dim: int,
        activation: type[nn.Module],
        attn_drop_prob: float,
        drop_prob: float,
    ):
        super().__init__()
        self.attn = _DuINAttention(embed_dim, n_heads, head_dim, attn_drop_prob)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_attn(x + self.attn(x))
        return self.norm_ffn(x + self.ffn(x))


# Upstream module prefixes -> this implementation, for the encoder layers.
_UPSTREAM_LAYER_KEYS = {
    "mha.W_q.W.": "attn.q.",
    "mha.W_k.W.": "attn.k.",
    "mha.W_v.W.": "attn.v.",
    "mha.norm_q.": "attn.q_norm.",
    "mha.norm_k.": "attn.k_norm.",
    "mha.proj.0.": "attn.proj.",
    "norm_mha.": "norm_attn.",
    "ffn.fc1.0.": "ffn.0.",
    "ffn.fc2.0.": "ffn.3.",
    "norm_ffn.": "norm_ffn.",
}


def _convert_upstream_state_dict(state_dict, n_neural: int) -> OrderedDict:
    """Rename an upstream Du-IN state dict to :class:`DuIN` keys."""
    converted = OrderedDict()
    for key, value in state_dict.items():
        if key == "subj_block.subj_layer.W.weight":
            if value.shape[1] != 1:
                raise ValueError(
                    "Only single-subject Du-IN checkpoints can be loaded; this "
                    f"one has {value.shape[1]} subjects."
                )
            # (n_chans * n_neural, 1) -> nn.Linear weight (n_neural, n_chans)
            converted["spatial_projection.weight"] = (
                value[:, 0].reshape(-1, n_neural).T.contiguous()
            )
        elif key == "subj_block.subj_layer.B.weight":
            converted["spatial_projection.bias"] = value[:, 0]
        elif key.startswith("subj_block."):
            raise ValueError(f"Unsupported upstream subject-block parameter {key!r}.")
        elif key.startswith("tokenizer.conv_blocks."):
            # tokenizer.conv_blocks.<i>.<0: conv | 1: batch norm>.1.<param>
            block, part, _, param = key[len("tokenizer.conv_blocks.") :].split(".", 3)
            name = "conv" if part == "0" else "bn"
            converted[f"spatial_encoder.{block}.{name}.{param}"] = value
        elif key.startswith("encoder.1.xfmr_blocks."):
            layer, rest = key[len("encoder.1.xfmr_blocks.") :].split(".", 1)
            if rest == "mha.attention.scale":  # fixed 1 / sqrt(head_dim)
                continue
            for src, dst in _UPSTREAM_LAYER_KEYS.items():
                if rest.startswith(src):
                    converted[f"encoder.{layer}.{dst}{rest[len(src) :]}"] = value
                    break
            else:
                raise ValueError(f"Unknown upstream encoder parameter {key!r}.")
        # mask_emb, emb_time (fixed table), vq_block, contra_block and the
        # upstream heads (cls_block) have no counterpart here.
    return converted

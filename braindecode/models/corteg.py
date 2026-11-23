# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: MIT
# Adapted from https://github.com/LiuyinYang1101/CORTEG (MIT).
# Copyright (c) 2026 Liuyin Yang
"""CORTEG: an ST-EEGFormer adapted to intracranial recordings."""

from __future__ import annotations

import math
from functools import partial
from typing import Dict, Optional, Union

import mne
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from braindecode.functional import (
    check_ch_mask,
    init_weights_,
    masked_mean,
    sinusoidal_positional_encoding,
)
from braindecode.functional.patches import pad_or_crop_to_patches
from braindecode.models.base import EEGModuleMixin
from braindecode.models.steegformer import STEEGFORMER_CHANNEL_ORDER
from braindecode.models.util import chs_info_positions
from braindecode.modules import (
    FeedForwardBlock,
    MultiHeadAttention,
    PatchTokenizer,
    PreNormTransformerBlock,
)


class CORTEG(EEGModuleMixin, nn.Module, license="mit"):
    r"""CORTEG from Yang et al. (2026) [Yang2026b]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    .. figure:: https://raw.githubusercontent.com/LiuyinYang1101/CORTEG/main/docs/assets/figure1_overview_scheme.png
        :align: center
        :width: 1000px

        CORTEG overview, reproduced from the official repository
        (``LiuyinYang1101/CORTEG``).

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    A scalp-EEG foundation model, :class:`~braindecode.models.STEEGFormer`,
    adapted to intracranial (ECoG) recordings. The transformer stays the
    pretrained one; around it, a spatial adapter places each electrode by its
    3D position, and a second token stream carries the high-gamma activity the
    scalp model was never trained on. The paper trains only the spatial
    adapter, the high-gamma patch embedding, LoRA on the last
    ``n_merge_blocks`` blocks with their LayerNorms, and the head (about 297 K
    parameters for the small backbone); this module builds the network and
    freezes the channel codebook, and leaves LoRA and the freezing of the other
    parameters to the training code. With ``B`` the batch, ``C = n_chans``,
    ``D = embed_dim``, ``P`` the patches per electrode (``n_times /
    patch_size``, rounded as ``on_non_divisible`` says) and ``N = P * C``::

        x (B, C, T), x_hi (B, C, T_hi), pos (B, C, 3)
        -> patch embeddings                 lo (B, N, D), hi (B, N, D)
        -> + temporal and channel embedding of each token
        -> CLS token, low-frequency tokens  (B, 1 + N, D)
        -> encoder blocks, with the fusion  (B, 1 + N, D)
        -> norm                             (B, 1 + N, D)
        -> mean over all tokens             (B, D)
        -> final_layer                      (B, n_outputs)

    .. rubric:: Macro Components

    - **Dual-stream tokenizer** (``CORTEG.patch_embed``,
      ``CORTEG.patch_embed_hi``). *Operations:* the low-frequency stream ``x``
      is cut into patches of ``patch_size`` samples, the high-gamma stream
      ``x_hi`` into patches of ``hi_patch_size``, each linearly embedded and
      the high-gamma tokens average-pooled to the low-frequency count. The
      high-gamma embedding starts from the pretrained low-frequency one,
      linearly interpolated to its patch size. *Role:* keep the pretrained
      low-frequency tokenization and add the high-gamma dynamics on the same
      token grid, so that both streams get the same temporal and channel
      embeddings (Section 3.1 of the paper).
    - **KNNSoftFourier spatial adapter** (``CORTEG.channel_adapter``).
      *Operations:* a soft branch maps each electrode position to softmax
      weights over the 142 positioned slots of the pretrained channel
      codebook (``CORTEG.channel_codebook``, frozen) and returns their
      weighted sum; the last bias starts from a Gaussian ``knn_k``-nearest-
      neighbour kernel over the slots' scalp positions, so the soft branch
      starts as a nearest-electrode interpolation. A Fourier residual
      (``n_fourier_freqs`` log-spaced frequencies per axis, up to
      ``fourier_max_freq``, zero-initialized output) adds a position-specific
      correction. *Role:* replace the scalp channel
      embedding, which knows no intracranial electrode.
    - **Transformer encoder** (``CORTEG.encoder``). *Operations:* the
      ``depth`` pretrained pre-norm ViT blocks of ST-EEGFormer, over a CLS
      token and the (time-patch, channel) tokens of the low-frequency stream.
      *Role:* the pretrained backbone, which extracts EEG-like low-frequency
      representations; with ``"average"`` its first ``depth - n_merge_blocks``
      blocks read the low-frequency stream alone (Section 3.1).
    - **Late fusion** (``merge_strategy``). *Operations:* ``"average"``
      replaces the low-frequency tokens by their mean with the high-gamma
      tokens before the last ``n_merge_blocks`` blocks; ``"layerwise_gate"``
      adds the high-gamma tokens after every block, scaled per block and per
      sample by a small gate network that starts at zero. *Role:* bring the
      high-gamma activity into the backbone, after its early blocks have read
      the low-frequency stream (``"average"``, the paper's late fusion,
      Section 3.1) or at every block, which avoids searching for the merge
      point (``"layerwise_gate"``, Section 5.4).
    - **Read-out** (``CORTEG.norm``, ``CORTEG.final_layer``). *Operations:*
      layer norm, mean over all tokens (CLS included), linear head. *Role:*
      one vector per window, projected to ``n_outputs`` values (Section 3.1).

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** patches of ``patch_size`` samples (16 at 128 Hz, 125 ms),
      each tagged with the fixed sinusoidal encoding of its position; the CLS
      token takes position 0.
    - **Spatial:** the 3D position of each electrode, in metres (the paper's
      MNI coordinates), passed as ``pos`` or read from ``chs_info``, becomes a
      channel embedding through the spatial adapter.
    - **Spectral:** the second stream, ``x_hi``, carries the high-gamma
      envelope that the scalp model never saw.

    .. rubric:: Additional Mechanisms

    - **Input streams:** the paper decomposes the raw signal into two streams,
      both z-scored per channel with training-set statistics. ``x`` is the
      low-frequency stream, resampled to 128 Hz; ``x_hi`` is the 70-200 Hz
      Hilbert envelope at 200 Hz, so that its 25-sample patches line up with
      those of ``x``. With MNE, ``raw.copy().filter(70, 200)
      .apply_hilbert(envelope=True).resample(200)`` gives ``x_hi``. Without
      ``x_hi`` the model reads ``x`` alone, as the paper's low-frequency-only
      ablation.
    - **Channel padding:** ``ch_mask`` is an addition to the reference, which
      batches one patient at a time: padded channels take part in no
      attention and no pooling.

    .. rubric:: Pretrained Weights

    The ST-EEGFormer small backbone, with the high-gamma embedding
    warm-started from its low-frequency one, loads from the re-hosted
    checkpoint; the adapter, gate and head keep their initialization::

        model = CORTEG.from_pretrained(
            "braindecode/corteg-small-pretrained", n_outputs=5, n_times=128,
            chs_info=chs_info,
        )

    The authors' Stanford model (finger flexion, nine patients pooled) is
    re-hosted with its LoRA factors folded in::

        model = CORTEG.from_pretrained(
            "braindecode/corteg-stanford-pooled", chs_info=chs_info
        )

    The Stanford model reads 1 s windows of the two streams above (``x`` of 128
    and ``x_hi`` of 200 samples) after a notch at 60, 120 and 180 Hz and a
    common average reference (``data/stanford_preprocessing/`` of the authors'
    repository). The outputs match the reference implementation
    (``LiuyinYang1101/CORTEG`` at ``caafb78``) for the same weights and inputs,
    up to float32 rounding, for both merge strategies (:gh:`1551`).

    A bare ``STEEGFormer`` checkpoint (``braindecode/STEEGFormer-small``) is not
    a CORTEG checkpoint. It has no high-gamma embedding and names the channel
    table ``channel_pos.embedding.weight``, so ``strict=True`` raises, and
    ``strict=False`` leaves ``channel_codebook`` at zero and derives
    ``patch_embed_hi`` from the random initialization. Load
    ``braindecode/corteg-small-pretrained`` instead.

    Parameters
    ----------
    patch_size : int, default=16
        Patch size of the low-frequency stream (16 for the released
        backbones).
    hi_patch_size : int, default=25
        Patch size of the high-gamma stream.
    n_temporal_positions : int, default=2048
        Positions of the fixed temporal encoding, CLS token included (2048, as in
        STEEGFormer); a window has at most ``n_temporal_positions - 1`` patches.
    embed_dim : int, default=512
        Token embedding dimension (512 for the small backbone).
    depth : int, default=8
        Number of Transformer encoder blocks.
    num_heads : int, default=8
        Number of attention heads.
    mlp_ratio : int, default=4
        Hidden-to-embedding ratio of the MLP blocks.
    norm_eps : float, default=1e-6
        Epsilon of the LayerNorms of the blocks and of the final norm (1e-6, as in
        STEEGFormer).
    drop_prob : float, default=0.0
        Dropout rate.
    drop_path : float, default=0.1
        Stochastic-depth rate (max of a linear schedule over ``depth``).
    activation : type[nn.Module], default=nn.GELU
        Activation layer class of the feed-forward blocks and of the MLPs of
        the spatial adapter and the gate.
    merge_strategy : {"average", "layerwise_gate"}, default="average"
        Fusion of the high-gamma stream: the paper's default mean before the
        last ``n_merge_blocks`` blocks (the released Stanford adapter), or the
        layer-wise gate (its BrainTreebank results).
    n_merge_blocks : int, default=4
        Number of blocks after the ``"average"`` fusion.
    knn_k : int, default=8
        Number of nearest scalp slots of the soft branch's initial kernel.
    min_bandwidth : float, default=1e-6
        Smallest bandwidth of that kernel, in metres (1e-6).
    weight_eps : float, default=1e-8
        Floor that keeps the normalization and the logarithm of the kernel
        weights finite (1e-8).
    adapter_hidden : int, default=128
        Hidden width of the spatial adapter's MLPs.
    n_fourier_freqs : int, default=32
        Number of Fourier frequencies per axis of the residual branch.
    fourier_max_freq : float, default=50.0
        Highest of those frequencies, log-spaced from 1, in radians per metre.
    gate_bottleneck : int, default=16
        Hidden width of the layer-wise gate.
    init_std : float, default=0.02
        Standard deviation of the normal initialization of the CLS token (0.02).
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        How a stream whose length is not a multiple of its patch size is
        handled (right zero-pad, crop, or raise); the reference crops.

    References
    ----------
    .. [Yang2026b] Yang, L., Sun, Q., Van Dyck, B., Calvo Merino, E. &
       Van Hulle, M. M. (2026). CORTEG: Foundation models enable
       cross-modality representation transfer from scalp to intracranial
       brain recordings. Advances in Neural Information Processing Systems
       (NeurIPS 2026). https://arxiv.org/abs/2605.10337
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: default_pos is sized by n_chans
    _time_inputs = ("x", "x_hi")

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
        # model-specific parameters (defaults: small backbone)
        patch_size: int = 16,
        hi_patch_size: int = 25,
        n_temporal_positions: int = 2048,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        norm_eps: float = 1e-6,
        drop_prob: float = 0.0,
        drop_path: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        merge_strategy: str = "average",
        n_merge_blocks: int = 4,
        knn_k: int = 8,
        min_bandwidth: float = 1e-6,
        weight_eps: float = 1e-8,
        adapter_hidden: int = 128,
        n_fourier_freqs: int = 32,
        fourier_max_freq: float = 50.0,
        gate_bottleneck: int = 16,
        init_std: float = 0.02,
        on_non_divisible: str = "pad",
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

        if merge_strategy not in ("average", "layerwise_gate"):
            raise ValueError(
                "merge_strategy must be 'average' or 'layerwise_gate', got "
                f"{merge_strategy!r}."
            )
        if not 1 <= n_merge_blocks <= depth:
            raise ValueError(
                f"n_merge_blocks must be in [1, depth={depth}], got {n_merge_blocks}."
            )
        # A whole-number float, such as the 4.0 of STEEGFormer's config, gives the
        # same blocks as the integer; a fraction has no integer expansion
        if isinstance(mlp_ratio, bool) or mlp_ratio != int(mlp_ratio):
            raise ValueError(f"mlp_ratio must be a whole number, got {mlp_ratio!r}.")
        vocab_positions = _vocab_positions()  # (n_slots, 3)
        n_slots = len(vocab_positions)
        self.embed_dim = embed_dim
        self.merge_strategy = merge_strategy
        self.merge_at = depth - n_merge_blocks
        self.knn_k = knn_k
        # Read again by reset_parameters, through _init_parameters.
        self.min_bandwidth = min_bandwidth
        self.weight_eps = weight_eps
        self.init_std = init_std

        default_pos = chs_info_positions(self._chs_info, "CORTEG")
        # Montage-specific, rebuilt from chs_info: not saved with the weights.
        self.register_buffer("default_pos", default_pos, persistent=False)
        self.register_buffer("vocab_positions", vocab_positions, persistent=False)

        # The ST-EEGFormer backbone; its weights keep STEEGFormer's names, except
        # the channel codebook.
        self.patch_embed = PatchTokenizer(
            patch_size=patch_size,
            n_times=self.n_times,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
            output_order="patch_channel",
        )
        # One temporal position per patch, plus position 0 for the CLS token.
        n_patches = self.patch_embed.n_patches(self.n_times)
        if n_temporal_positions < n_patches + 1:
            raise ValueError(
                f"n_temporal_positions ({n_temporal_positions}) must be at least "
                f"n_patches + 1 = {n_patches + 1} for n_times={self.n_times}."
            )
        # Fixed sinusoidal encoding of the patch positions: (n_temporal_positions, D),
        # position 0 for the CLS token and 1.. for the time patches.
        self.register_buffer(
            "temporal_encoding",
            sinusoidal_positional_encoding(n_temporal_positions, embed_dim),
            persistent=False,
        )
        # The pretrained channel embedding of the vocabulary's slots, (n_slots, D):
        # the soft branch's codebook, frozen as in the paper and filled by a checkpoint.
        self.channel_codebook = nn.Parameter(
            torch.empty(n_slots, embed_dim), requires_grad=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.encoder = nn.Sequential(
            *[
                PreNormTransformerBlock(
                    embed_dim,
                    MultiHeadAttention(embed_dim, num_heads, drop_prob),
                    FeedForwardBlock(
                        embed_dim,
                        expansion=int(mlp_ratio),
                        drop_p=drop_prob,
                        activation=activation,
                    ),
                    drop_prob,
                    eps=norm_eps,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # x_hi of this length gives one high-gamma token per low-frequency token.
        self.patch_embed_hi = PatchTokenizer(
            patch_size=hi_patch_size,
            n_times=self.patch_embed.n_patches(self.n_times) * hi_patch_size,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
            output_order="patch_channel",
        )
        self.channel_adapter = _KNNSoftFourierAdapter(
            n_slots,
            embed_dim,
            adapter_hidden,
            n_fourier_freqs,
            fourier_max_freq,
            activation,
        )
        self.layerwise_gate: Optional[_LayerwiseGate] = (
            _LayerwiseGate(embed_dim, depth, gate_bottleneck, activation)
            if merge_strategy == "layerwise_gate"
            else None
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.patch_size = patch_size
        self.hi_patch_size = hi_patch_size
        self.on_non_divisible = on_non_divisible
        self._n_times_step = patch_size
        self._min_n_times = 1 if on_non_divisible == "pad" else patch_size

        self._init_parameters()

    def _snap_grid_tensor(self, name: str, x: torch.Tensor) -> torch.Tensor:
        patch = self.hi_patch_size if name == "x_hi" else self.patch_size
        return pad_or_crop_to_patches(x, patch, self.on_non_divisible)

    def _accepts_any_n_chans(self, args: tuple, kwargs: dict) -> bool:
        if super()._accepts_any_n_chans(args, kwargs):
            return True
        pos = self._call_arg(args, kwargs, "pos")
        return pos is not None

    def _init_parameters(self) -> None:
        """Model-specific initialization (see ``reset_parameters``)."""
        for module in (self.patch_embed, self.encoder, self.norm, self.final_layer):
            module.apply(_init_weights)
        nn.init.normal_(self.cls_token, std=self.init_std)
        # Zero, as if there were no channel embedding, until a checkpoint fills it.
        nn.init.zeros_(self.channel_codebook)
        self._init_high_gamma_from_low()
        prior = None
        if self.default_pos is not None:
            prior = _knn_log_prior(
                self.default_pos,
                self.vocab_positions,
                self.knn_k,
                self.min_bandwidth,
                self.weight_eps,
            )
        self.channel_adapter.init_parameters(prior)
        if self.layerwise_gate is not None:
            self.layerwise_gate.init_parameters()

    @torch.no_grad()
    def _init_high_gamma_from_low(self):
        """Set the high-gamma patch embedding from the low-frequency one.

        The kernel ``patch_embed.proj.weight``, ``(embed_dim, patch_size)``, is
        resized along its patch axis to ``(embed_dim, hi_patch_size)``; the bias,
        ``(embed_dim,)``, is copied.
        """
        lo, hi = self.patch_embed.proj, self.patch_embed_hi.proj
        # Linear resize of the kernel viewed as one (embed_dim, patch_size) signal:
        # (1, embed_dim, patch_size) -> (1, embed_dim, hi_patch_size).
        resized = F.interpolate(lo.weight[None], size=hi.in_features, mode="linear")
        hi.weight.copy_(resized[0])
        hi.bias.copy_(lo.bias)

    def reset_head(self, n_outputs: int) -> None:
        """Replace the linear head for a new ``n_outputs``."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        _init_weights(self.final_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_hi: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode a batch of intracranial recordings.

        Parameters
        ----------
        x : torch.Tensor
            Low-frequency stream, ``(batch, n_chans, n_times)``.
        x_hi : torch.Tensor, optional
            High-gamma stream of the same channels, ``(batch, n_chans,
            n_times_hi)``. Without it the model reads ``x`` alone.
        pos : torch.Tensor, optional
            Electrode positions in metres, ``(batch, n_chans, 3)`` or
            ``(n_chans, 3)``; by default those of ``chs_info``.
        ch_mask : torch.Tensor, optional
            ``(batch, n_chans)`` mask of real channels, see
            :func:`~braindecode.functional.check_ch_mask`. Padded channels
            take part in no attention and no pooling.
        return_features : bool
            If ``True``, return ``{"features", "cls_token", "tokens"}``: the
            layer-normalized patch tokens ``(batch, n_patches * n_chans,
            embed_dim)``, the CLS token, and the patch tokens as ``(batch,
            n_chans, n_patches, embed_dim)``; padded channels' tokens are zero.

        Returns
        -------
        torch.Tensor | dict
            Outputs ``(batch, n_outputs)``, or the feature dict.
        """
        # Shapes: B batch, C channels, P patches, D embed_dim, N = P * C tokens.
        batch_size, n_chans, n_times = x.shape
        n_patches = self.patch_embed.n_patches(n_times)
        mask = check_ch_mask(ch_mask, x)  # (B, C), None without padding
        if pos is None:
            pos = self.default_pos
        hi: Optional[torch.Tensor] = None
        gates: Optional[torch.Tensor] = None
        token_mask: Optional[torch.Tensor] = None
        key_mask: Optional[torch.Tensor] = None
        attn_mask: Optional[torch.Tensor] = None

        if pos is None:
            raise ValueError(
                "CORTEG places each electrode by its 3D position: pass pos, or "
                "chs_info with the electrode positions."
            )
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)  # (1, C, 3), shared by the batch
        if pos.shape[1] != n_chans or pos.shape[2] != 3:
            raise ValueError(
                f"pos must have shape (batch, {n_chans}, 3) or ({n_chans}, 3), "
                f"got {list(pos.shape)}."
            )
        if mask is not None:
            # Padding may hold NaN, which the embeddings would spread.
            x = x.masked_fill(~mask[:, :, None], 0.0)  # (B, C, T)
            pos = pos.masked_fill(~mask[:, :, None], 0.0)  # (B, C, 3)
            if x_hi is not None:
                x_hi = x_hi.masked_fill(~mask[:, :, None], 0.0)  # (B, C, T_hi)
            # (B, N), patch-major like the tokens
            token_mask = mask.repeat(1, n_patches)
            key_mask = F.pad(token_mask, (1, 0), value=True)  # (B, 1 + N), CLS first
            attn_mask = key_mask[:, None, None, :]  # (B, 1, 1, 1 + N)

        # Each electrode's embedding, from its position and the pretrained codebook.
        channel_emb = self.channel_adapter(pos, self.channel_codebook)  # (B or 1, C, D)
        # Positions 1..P of the temporal encoding, (1, P, 1, D), plus the channel
        # embedding (B or 1, 1, C, D) -> (B or 1, P, C, D)
        temporal_emb = self.temporal_encoding[1 : n_patches + 1][None, :, None]
        embedding = temporal_emb + channel_emb[:, None]
        tokens = self.patch_embed(x)  # (B, P, C, D)
        lo = (tokens + embedding).flatten(1, 2)  # (B, N, D)
        if x_hi is not None:
            hi_tokens = self.patch_embed_hi(x_hi)  # (B, P_hi, C, D)
            if hi_tokens.shape[1] != n_patches:
                pooled = F.adaptive_avg_pool1d(
                    hi_tokens.permute(0, 2, 3, 1).flatten(0, 1), n_patches
                )  # (B * C, D, P)
                hi_tokens = pooled.reshape(
                    batch_size, n_chans, self.embed_dim, n_patches
                ).permute(0, 3, 1, 2)  # (B, P, C, D)
            hi = (hi_tokens + embedding).flatten(1, 2)  # (B, N, D)

        # CLS token at position 0 of the temporal encoding: (1, 1, D)
        cls = self.cls_token + self.temporal_encoding[0]
        z = torch.cat([cls.expand(batch_size, -1, -1), lo], dim=1)  # (B, 1 + N, D)
        z = self.pos_drop(z)
        if hi is not None and self.layerwise_gate is not None:
            gates = self.layerwise_gate(
                masked_mean(lo, token_mask, [1]), masked_mean(hi, token_mask, [1])
            )  # (B, depth)
        for i, block in enumerate(self.encoder):
            if hi is not None and gates is None and i == self.merge_at:
                # "average" fusion: the CLS token is kept, the others averaged.
                z = torch.cat([z[:, :1], 0.5 * (z[:, 1:] + hi)], dim=1)  # (B, 1 + N, D)
            # (B, 1 + N, D); padded channels are keys no token attends to
            z = block(z) if attn_mask is None else block.forward_masked(z, attn_mask)
            if hi is not None and gates is not None:
                # "layerwise_gate" fusion: hi added after each block, not to CLS;
                # (B, 1 + N, D)
                z = z + gates[:, i, None, None] * F.pad(hi, (0, 0, 1, 0))
        z = self.norm(z)  # (B, 1 + N, D)

        if return_features:
            patches = z[:, 1:]  # (B, N, D)
            if token_mask is not None:
                patches = patches.masked_fill(~token_mask[:, :, None], 0.0)
            per_channel = patches.reshape(
                batch_size, n_patches, n_chans, self.embed_dim
            ).transpose(1, 2)  # (B, C, P, D)
            return {"features": patches, "cls_token": z[:, 0], "tokens": per_channel}
        return self.final_layer(masked_mean(z, key_mask, [1]))  # (B, n_outputs)


# Helpers, below the model class.

# Weight initialization of the ST-EEGFormer reference: Xavier-uniform linear
# layers with zero bias, unit LayerNorms.
_init_weights = partial(init_weights_, weights={nn.Linear: nn.init.xavier_uniform_})


def _vocab_positions() -> torch.Tensor:
    """``(142, 3)`` positions (m) of ST-EEGFormer's positioned vocabulary slots.

    The slot names come from ``STEEGFORMER_CHANNEL_ORDER``, the channel
    vocabulary of the released ST-EEGFormer checkpoints; their positions from
    MNE's Colin27 10-05 montage, which gives the reference's table to 6
    decimals.
    """
    montage = mne.channels.make_standard_montage("colin27_1005")
    ch_pos = montage.get_positions()["ch_pos"]
    positions = np.stack([ch_pos[name] for name in STEEGFORMER_CHANNEL_ORDER])
    return torch.as_tensor(positions, dtype=torch.float32)


def _knn_log_prior(
    positions: torch.Tensor,
    vocab_positions: torch.Tensor,
    k: int,
    min_bandwidth: float,
    weight_eps: float,
) -> torch.Tensor:
    """Log Gaussian k-NN weight of each vocabulary slot, averaged over ``positions``.

    Steps:

    1. Take the ``k`` nearest slots of each electrode.
    2. Weight them with a Gaussian kernel whose bandwidth is the median of
       their distances, at least ``min_bandwidth``, normalized per electrode.
    3. Scatter the weights over all slots, take the log of the weights plus
       ``weight_eps`` and average over the electrodes.

    ``positions`` is ``(n_chans, 3)`` and ``vocab_positions`` ``(n_slots, 3)``,
    both in metres; the result is ``(n_slots,)``.
    """
    # (n_chans, n_slots) -> (n_chans, k) distances and slot indices
    distances, slots = torch.cdist(positions, vocab_positions).topk(
        k, dim=-1, largest=False
    )
    sigma = distances.median().clamp(min=min_bandwidth)  # scalar
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)  # (n_chans, k)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=weight_eps)
    # (n_chans, n_slots), zero outside the k nearest slots
    dense = weights.new_zeros((len(positions), len(vocab_positions))).scatter_(
        1, slots, weights
    )
    return torch.log(dense + weight_eps).mean(dim=0)  # (n_slots,)


class _KNNSoftFourierAdapter(nn.Module):
    """Channel embedding of a 3D electrode position (the paper's KNNSoftFourier).

    Steps:

    1. ``soft``: the position selects a softmax-weighted sum of the pretrained
       codebook slots (:class:`_SoftCodebookBranch`).
    2. ``residual``: a Fourier-feature MLP adds a position-specific correction
       (:class:`_FourierBranch`), zero at initialization.

    ``forward(pos, codebook)`` maps ``pos`` ``(batch or 1, n_chans, 3)`` and
    ``codebook`` ``(n_slots, embed_dim)`` to ``(batch or 1, n_chans, embed_dim)``.
    """

    def __init__(
        self,
        n_slots: int,
        embed_dim: int,
        hidden: int,
        n_freqs: int,
        max_freq: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.soft = _SoftCodebookBranch(n_slots, hidden, activation)
        self.residual = _FourierBranch(embed_dim, hidden, n_freqs, max_freq, activation)

    def init_parameters(self, prior: Optional[torch.Tensor]) -> None:
        """Start the soft branch at ``prior`` (uniform if ``None``) and the
        residual at zero."""
        with torch.no_grad():
            bias = self.soft.net[-1].bias  # (n_slots,)
            bias.copy_(torch.zeros_like(bias) if prior is None else prior)
            self.soft.scale.fill_(1.0)
        nn.init.zeros_(self.residual.net[-1].weight)

    def forward(self, pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # (B or 1, C, 3) -> (B or 1, C, D), the two branches summed
        return self.soft(pos, codebook) + self.residual(pos)


class _SoftCodebookBranch(nn.Module):
    """Softmax attention of a position over the codebook slots.

    Steps:

    1. An MLP maps the 3D position to one logit per slot.
    2. The softmax of the logits weights the slots of the codebook; the weighted
       sum is multiplied by a learned scalar ``scale``.

    ``forward(pos, codebook)`` maps ``pos`` ``(batch or 1, n_chans, 3)`` and
    ``codebook`` ``(n_slots, embed_dim)`` to ``(batch or 1, n_chans, embed_dim)``.
    """

    def __init__(self, n_slots: int, hidden: int, activation: type[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),  # 3: x, y, z of the electrode
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, n_slots),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # (B or 1, C, 3) -> (B or 1, C, n_slots) weights over the slots
        weights = self.net(pos).softmax(dim=-1)
        # (B or 1, C, n_slots) @ (n_slots, D) -> (B or 1, C, D)
        return self.scale * torch.matmul(weights, codebook)


class _FourierBranch(nn.Module):
    """MLP on sinusoidal features of a position, at log-spaced frequencies.

    Steps:

    1. Multiply each of the 3 coordinates by the ``n_freqs`` frequencies, log-spaced
       from 1 to ``max_freq`` radians per metre.
    2. Concatenate the sine and the cosine of the angles.
    3. Apply an MLP to the ``6 * n_freqs`` features.

    ``forward(pos)`` maps ``pos`` ``(batch or 1, n_chans, 3)`` to
    ``(batch or 1, n_chans, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden: int,
        n_freqs: int,
        max_freq: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.freqs: torch.Tensor
        self.register_buffer(
            "freqs", torch.logspace(0, math.log10(max_freq), n_freqs), persistent=False
        )
        # sin and cos of each frequency on each of the 3 axes
        self.net = nn.Sequential(
            nn.Linear(6 * n_freqs, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, embed_dim, bias=False),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # (B or 1, C, 3, 1) * (n_freqs,) -> (B or 1, C, 3 * n_freqs) angles
        angles = (pos[..., None] * self.freqs).flatten(-2)
        # (B or 1, C, 6 * n_freqs) -> (B or 1, C, D)
        return self.net(torch.cat([angles.sin(), angles.cos()], dim=-1))


class _LayerwiseGate(nn.Module):
    """One tanh gate per block and sample, from the mean low and high tokens.

    Steps:

    1. Concatenate the mean low-frequency and high-gamma tokens of the sample.
    2. Apply a LayerNorm and a bottleneck MLP with one output per block.
    3. Squash the outputs with a tanh into gates in ``(-1, 1)``.

    ``forward(lo, hi)`` maps two ``(batch, embed_dim)`` means to the gates
    ``(batch, depth)``.
    """

    def __init__(
        self, embed_dim: int, depth: int, bottleneck: int, activation: type[nn.Module]
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, bottleneck),
            activation(),
            nn.Linear(bottleneck, depth),
        )

    def init_parameters(self) -> None:
        """Start every gate closed, at the low-frequency-only model."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        # (B, D) and (B, D) -> (B, 2 * D) -> (B, depth) gates in (-1, 1)
        return torch.tanh(self.net(torch.cat([lo, hi], dim=-1)))

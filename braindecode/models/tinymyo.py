# Authors: Matteo Fasulo, Giusy Spacone, Thorir Mar Ingolfsson, Yawei Li,
#          Luca Benini, Andrea Cossettini (original implementation)
#          ptlnextdoor <67443596+ptlnextdoor@users.noreply.github.com>
#          (braindecode adaptation)
#
# License: Apache-2.0
# Adapted from ``models/TinyMyo.py`` of https://github.com/pulp-bio/BioFoundation
# at commit d4484feccf62eeec666380cb55b5d5b1d517cd71 (Apache-2.0,
# Copyright (C) 2025-2026 ETH Zurich, Switzerland).
"""TinyMyo, a small foundation model for surface EMG."""

from __future__ import annotations

import math
from functools import partial
from typing import Literal, Optional

import torch
from torch import nn

from braindecode.functional import check_n_chans, init_weights_
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import PatchTokenizer, PreNormBlock, TimmAttention, TimmMlp


class TinyMyo(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""TinyMyo from Fasulo et al. (2026) :footcite:`fasulo2026tinymyo`.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. figure:: ../_static/model/tinymyo_arch.png
       :align: center
       :alt: TinyMyo architecture
       :width: 1000px

       Overview of TinyMyo: per-channel EMG patches with a shared projection and
       channel embeddings, the transformer encoder layer, the self-supervised
       reconstruction head and the downstream heads. Reproduced from Fig. 1 of
       :footcite:`fasulo2026tinymyo` (© IEEE).

    .. versionadded:: 2.1

    TinyMyo is a 3.6M-parameter Transformer encoder for surface EMG (sEMG).
    The authors run it on the GAP9 microcontroller. They pretrained it with
    masked patch reconstruction on NinaPro DB6, NinaPro DB7 and emg2pose
    (2000 Hz, up to 16 channels, 0.5 s windows), and fine-tune it for gesture
    classification, hand-kinematics regression and silent-speech decoding.

    .. rubric:: Architecture Overview

    The model cuts each channel into patches of ``patch_size`` samples, and
    each patch becomes one token. One Transformer encoder reads the tokens of
    all channels, so attention mixes time and channels. With :math:`B` the
    batch size, :math:`C` = ``n_chans``, :math:`P` patches per channel and
    :math:`D` = ``embed_dim``::

        x (B, C, n_times)
        1. patch_embed: Linear on each patch  -> (B, C, P, D)
        2. + channel_embed (one per channel slot)
        3. channel-major tokens               -> (B, C * P, D)
        4. blocks: RoPE self-attention + MLP  -> (B, C * P, D)
        5. norm                               -> (B, C, P, D)   return_features
        6. mean over P, flatten C and D       -> (B, C * D)
        7. final_layer                        -> (B, n_outputs)

    .. rubric:: Macro Components

    ``TinyMyo.patch_embed``
        **Operations.** One :class:`~torch.nn.Linear` shared by all channels
        maps each patch of ``patch_size`` samples to a ``D``-wide token (a
        strided :class:`~torch.nn.Conv2d` in the reference, the same
        operation). **Role.** Channel-independent patching.

    ``TinyMyo.channel_embed``
        **Operations.** A learned vector per channel index, added to every
        token of that channel. **Role.** Channel slots.

    ``TinyMyo.blocks``
        **Operations.** Pre-norm blocks with rotary position embeddings (RoPE)
        on queries and keys. **Role.** Transformer encoder.

    ``TinyMyo.final_layer``
        **Operations.** Averages the tokens of each channel over time,
        concatenates the channels and applies one :class:`~torch.nn.Linear`
        (``EMGClassificationHead`` in the reference). **Role.** Classification
        head.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time.** The rotary positions restart at zero for each channel, so
      the first patch of every channel has the same position.
    - **Channels.** Channel slots follow the electrode order used in
      fine-tuning. The checkpoints have 16 slots, and a model with fewer
      channels uses the first ``n_chans``.
    - **Frequency.** The patch projection takes time-domain EMG samples.

    .. rubric:: Additional Mechanisms

    During training, each Transformer block applies stochastic depth to its
    attention and MLP residual branches.

    .. rubric:: Pretrained Weights

    The authors' weights (`PulpBio/TinyMyo
    <https://huggingface.co/PulpBio/TinyMyo>`_, CC BY-ND 4.0), converted to
    these parameter names, are on the Hugging Face Hub:

    ==================================  ===========  =========  =====================
    Repository                          ``n_chans``  Outputs    Task
    ==================================  ===========  =========  =====================
    ``braindecode/tinymyo-pretrained``  up to 16     (encoder)  masked reconstruction
    ``braindecode/tinymyo-db5``         16           53         NinaPro DB5 gestures
    ``braindecode/tinymyo-epn612``      8            6          EPN-612 gestures
    ``braindecode/tinymyo-uci-emg``     8            6          UCI EMG gestures
    ==================================  ===========  =========  =====================

    .. important::
       Fine-tune the pretrained encoder with a new head, or load a released
       classifier as it is:

       .. code-block:: python

           from braindecode.models import TinyMyo

           # Pretrained encoder, new 4-class head
           model = TinyMyo.from_pretrained(
               "braindecode/tinymyo-pretrained", n_chans=8, n_outputs=4, n_times=1000
           )

           # Released EPN-612 classifier
           model = TinyMyo.from_pretrained("braindecode/tinymyo-epn612")

    The authors pretrain on 20-450 Hz band-passed signals with a 50 Hz notch,
    min-max normalized per channel. In fine-tuning they band-pass at 20-90
    Hz and z-score each channel.

    With the released weights (float32, CPU), the outputs match the reference
    (``models/TinyMyo.py`` of BioFoundation at ``d4484fe``) within 1e-5, and
    so do the gradients of the released classifiers (:gh:`1649`); the
    re-hosted files reproduce the originals (``PulpBio/TinyMyo`` at
    ``bc14906``).

    .. rubric:: Differences from the Reference Implementation

    - This class ports the classification head only. The reference also has
      the pretraining decoder (with ``mask`` and ``pad_mask_ch`` inputs) and
      a convolutional regression head for NinaPro DB8, whose checkpoint is not
      re-hosted.
    - The reference requires ``n_times`` to equal its ``img_size``. This
      class computes the RoPE tables for the input length, so any length
      works, and ``on_non_divisible`` handles a partial last patch. The
      authors trained the weights on 1000-sample windows.

    Parameters
    ----------
    patch_size : int, default=20
        Samples per patch.
    embed_dim : int, default=192
        Width of the tokens.
    depth : int, default=8
        Number of Transformer blocks (``n_layer`` in the reference). The
        smaller TinyssimoMyo checkpoint has 4.
    num_heads : int, default=3
        Attention heads per block. They must divide ``embed_dim``, into heads
        of even width for RoPE.
    mlp_ratio : float, default=4.0
        Hidden width of the MLP over ``embed_dim``.
    qkv_bias : bool, default=True
        Bias of the query, key and value projections.
    drop_prob : float, default=0.1
        Dropout after the attention projection and in the MLP
        (``proj_drop`` in the reference).
    attn_drop : float, default=0.1
        Dropout on the attention weights.
    drop_path : float, default=0.1
        Stochastic depth rate of each block.
    activation : type[nn.Module], default=nn.GELU
        Activation of the MLP.
    norm_layer : type[nn.Module], default=nn.LayerNorm
        Normalization of the blocks and of the encoder output.
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        How :class:`~braindecode.modules.PatchTokenizer` treats a time axis
        that is not a multiple of ``patch_size``.
    n_channel_slots : int, default=16
        Number of channel slots (rows of ``channel_embed``), which is the
        largest ``n_chans`` the model accepts. The released checkpoints have 16.
    init_std : float, default=0.02
        Std of the truncated normal that initializes ``channel_embed``,
        truncated at ``+-init_std``.

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: final_layer.weight

    def __init__(
        self,
        # Braindecode parameters
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        chs_info=None,
        n_times: Optional[int] = None,
        input_window_seconds: Optional[float] = None,
        sfreq: Optional[float] = None,
        *,
        # Model-specific parameters
        patch_size: int = 20,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_prob: float = 0.1,
        attn_drop: float = 0.1,
        drop_path: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        on_non_divisible: Literal["pad", "crop", "error"] = "pad",
        n_channel_slots: int = 16,
        init_std: float = 0.02,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if not 1 <= self.n_chans <= n_channel_slots:
            raise ValueError(
                f"TinyMyo needs 1 to {n_channel_slots} channel slots, got n_chans={self.n_chans}."
            )
        # TimmAttention checks num_heads >= 1 and an even head width, but
        # builds a narrower attention when the heads do not split embed_dim.
        if num_heads >= 1 and embed_dim % num_heads:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.patch_embed = PatchTokenizer(
            patch_size=patch_size,
            n_times=self.n_times,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
        )
        # Read by _init_parameters, which reset_parameters calls again.
        self.init_std = init_std
        self.channel_embed = nn.Parameter(torch.zeros(1, n_channel_slots, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                PreNormBlock(
                    embed_dim,
                    TimmAttention(
                        embed_dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop,
                        proj_drop=drop_prob,
                        rotary_runs=self.n_chans,
                        rotary_run_len=self.patch_embed.n_patches(self.n_times),
                    ),
                    TimmMlp(
                        embed_dim,
                        hidden_features=int(embed_dim * mlp_ratio),
                        act_layer=activation,
                        drop=drop_prob,
                    ),
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.final_layer = nn.Linear(self.n_chans * embed_dim, self.n_outputs)
        self.on_non_divisible = on_non_divisible
        self._n_times_step = patch_size
        self._min_n_times = 1 if on_non_divisible == "pad" else patch_size

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialization of the reference."""
        nn.init.trunc_normal_(
            self.channel_embed,
            std=self.init_std,
            a=-self.init_std,
            b=self.init_std,
        )
        # The reference reinitializes the patch convolution weight and keeps
        # the bias drawn by Conv2d, whose fan-in equals that of this Linear.
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        for module in (self.blocks, self.norm, self.final_layer):
            module.apply(_init_weights)
        # Deeper blocks start with smaller residual branches
        with torch.no_grad():
            for layer_id, block in enumerate(self.blocks, start=1):
                block.attn.proj.weight.div_(math.sqrt(2.0 * layer_id))
                block.mlp.fc2.weight.div_(math.sqrt(2.0 * layer_id))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_chans, n_times)``.
        return_features : bool, default=False
            Return ``{"features": (batch, n_chans, n_patches, embed_dim),
            "cls_token": None}``, the normed encoder tokens, instead of the
            logits.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape ``(batch, n_outputs)``, or the features.
        """
        if torch.jit.is_scripting():
            # The eager check lives in EEGModuleMixin.__call__ (scripting skips it).
            check_n_chans(
                x,
                self.n_chans,
                "TinyMyo",
                "build the model with the n_chans of this input.",
            )
        # A token per patch of each channel: (batch, C, n_times) -> (batch, C, P, D)
        x = self.patch_embed(x)
        n_patches = x.shape[2]
        # The embedding of its channel slot added to every token: (batch, C, P, D)
        x = x + self.channel_embed[:, : self.n_chans]
        # One sequence, the patches of channel 0 first: (batch, C * P, D)
        x = x.flatten(1, 2)
        # Attention over time and channels; the RoPE positions restart at each
        # channel: (batch, C * P, D)
        for block in self.blocks:
            x = block(x)
        # Normed tokens, channels back on their own axis: (batch, C, P, D)
        x = self.norm(x).unflatten(1, (self.n_chans, n_patches))
        if return_features:
            return {"features": x, "cls_token": None}
        # Mean over the patches of each channel, channels side by side:
        # (batch, C * D)
        x = x.mean(dim=2).flatten(1)
        # Linear classifier: (batch, n_outputs)
        return self.final_layer(x)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        _init_weights(self.final_layer)


# Xavier-uniform Linear weights and zero biases, the initialization of the
# reference (models/TinyMyo.py of pulp-bio/BioFoundation)
_init_weights = partial(init_weights_, weights={nn.Linear: nn.init.xavier_uniform_})

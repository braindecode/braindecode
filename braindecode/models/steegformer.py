# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)
"""STEEGFormer — WIP draft (see #1040).

Port of Yang et al. (2025), https://github.com/LiuyinYang1101/STEEGFormer
"""
from __future__ import annotations

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin


class STEEGFormer(EEGModuleMixin, nn.Module):
    r"""STEEGFormer from Yang et al. (2025) [steegformer2025]_.

    :bdg-info:`Attention/Transformer` :bdg-warning:`Foundation / Self-supervised`

    .. rubric:: Architectural Overview

    ViT-based foundation model, pre-trained with a Masked Autoencoder (MAE)
    objective on raw EEG. Segments (<= 6 s @ 128 Hz) are split into temporal
    patches per channel, embedded as tokens, enriched with temporal/channel
    positional embeddings and a CLS token, processed by a Transformer encoder,
    and read out by an average-pooling classification head.

    .. note::
        Work in progress (draft). Only the public signature is defined for now;
        the architecture is not implemented yet. See #1040.

    Parameters
    ----------
    %(EEGModuleMixin)s
    patch_size : int
        Temporal patch size (unfold), default 16.
    embed_dim : int
        Token embedding dimension (512 / 768 / 1024 across variants).
    depth : int
        Number of Transformer encoder blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Hidden-to-embedding ratio of the MLP blocks.
    drop_rate : float
        Dropout rate.
    global_pool : str
        Token aggregation before the head (``"avg"`` or ``"cls"``).

    References
    ----------
    .. [steegformer2025] Yang, L. et al. (2025). STEEGFormer.
       OpenReview, https://openreview.net/pdf?id=5Xwm8e6vbh
    """   

    def __init__(
        self,
        # --- signal-related (handled by EEGModuleMixin) ---
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None, 
        # --- model hyperparameters ---
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,  
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        global_pool: str = "avg",
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

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.global_pool = global_pool

        # TODO(#1040): patch embedding, positional embeddings + CLS token,
        # Transformer encoder, and the avg-pool classification head.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_chans, n_times)
        raise NotImplementedError("STEEGFormer is a WIP draft — see #1040.")

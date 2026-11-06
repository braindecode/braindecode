# Authors: Yonghao Song <eeyhsong@gmail.com> (original implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/eeyhsong/NICE-EEG (MIT).

from typing import Dict, Optional, Union

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import PostNormResidual


class NICE(EEGModuleMixin, nn.Module, license="mit"):
    r"""NICE EEG/MEG encoder from Song et al. (2024) [song2024nice]_.

    :bdg-success:`Convolution`

    .. versionadded:: 2.1

    NICE (Natural Image Contrast EEG) learns an EEG/MEG embedding aligned with
    the image embedding of a frozen CLIP ViT-L/14 by contrastive learning, and
    decodes seen images zero-shot by nearest-neighbour search in that space.
    This class is the brain branch: the TSConv encoder (``Enc_eeg``) and its
    projector (``Proj_eeg``).

    - **TSConv**: a temporal convolution with ``n_filters`` kernels of size
      ``(1, filter_time_length)``, average pooling of size
      ``(1, pool_time_length)`` and stride ``(1, pool_time_stride)``, batch
      norm + activation, a spatial convolution over all ``n_chans``, batch
      norm + activation, dropout and a 1x1 convolution. Unlike
      :class:`EEGConformer`'s patch embedding, the pooling comes before the
      spatial convolution.
    - **Projector**: the flattened ``(time, filters)`` map goes through a
      linear layer to ``embed_dim``, then
      ``LayerNorm(x + Dropout(Linear(GELU(x))))``. Its output is the aligned
      embedding, returned as ``"features"`` by ``return_features=True``; the
      contrastive loss L2-normalises it.
    - **Head**: ``final_layer`` is a linear classifier on the embedding (not in
      the original, which only does retrieval).

    The paper uses THINGS-EEG2 (63 channels, 250 Hz, 0-1 s, so
    ``n_chans=63, n_times=250``) and, in its appendix A.6, THINGS-MEG (271
    channels, 200 Hz, 0-1 s, so ``n_chans=271, n_times=200``), with the same
    TSConv hyperparameters. The plug-in spatial modules of the paper (NICE-SA,
    NICE-GA) are not included.

    The authors' checkpoint (`eeyhsong/NICE <https://huggingface.co/eeyhsong/NICE>`_,
    CC-BY-NC-ND-4.0, 10 subjects of THINGS-EEG2) keeps its original keys,
    which :attr:`mapping` translates: ``model.load_state_dict({**enc, **proj},
    strict=False)`` with ``enc`` and ``proj`` the state dicts of
    ``pretrain_Enc_eeg_cls.pth`` and ``pretrain_Proj_eeg_cls.pth``. At Hub
    revision ``7f98b08`` the encoder file is truncated and :func:`torch.load`
    rejects it; the projector file loads.

    `License <https://github.com/eeyhsong/NICE-EEG/blob/main/LICENSE>`_

    Parameters
    ----------
    n_filters : int, default=40
        Number of temporal and spatial filters (``k`` in the paper).
    filter_time_length : int, default=25
        Temporal kernel length (``m1``).
    pool_time_length : int, default=51
        Average-pooling length (``m2``).
    pool_time_stride : int, default=5
        Average-pooling stride (``s``).
    embed_dim : int, default=768
        Size of the aligned embedding, the CLIP ViT-L/14 image feature size.
    activation : type[nn.Module], default=nn.ELU
        Activation of the convolutional encoder.
    drop_prob : float, default=0.5
        Dropout of the convolutional encoder.
    drop_prob_proj : float, default=0.5
        Dropout in the projector's residual branch.

    References
    ----------
    .. [song2024nice] Song, Y., Liu, B., Li, X., Shi, N., Wang, Y., & Gao, X.
        (2024). Decoding Natural Images from EEG for Object Recognition.
        International Conference on Learning Representations (ICLR).
        https://arxiv.org/abs/2308.13234
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        n_filters: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 51,
        pool_time_stride: int = 5,
        embed_dim: int = 768,
        activation: type[nn.Module] = nn.ELU,
        drop_prob: float = 0.5,
        drop_prob_proj: float = 0.5,
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

        self.embed_dim = embed_dim
        # Paper: k * ((T - m1 - m2 + 1) / s + 1) features enter the projector.
        n_steps = (
            self.n_times - filter_time_length - pool_time_length + 1
        ) // pool_time_stride + 1

        self.add_kernel_dim = Rearrange("batch chans time -> batch 1 chans time")
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, filter_time_length)),
            nn.AvgPool2d((1, pool_time_length), (1, pool_time_stride)),
            nn.BatchNorm2d(n_filters),
            activation(),
            nn.Conv2d(n_filters, n_filters, (self.n_chans, 1)),
            nn.BatchNorm2d(n_filters),
            activation(),
            nn.Dropout(drop_prob),
        )
        self.projection = nn.Conv2d(n_filters, n_filters, 1)
        # The original flattens time-major: (h w) e, then view(batch, -1).
        self.flatten = Rearrange("batch filters 1 time -> batch (time filters)")
        self.projector = nn.Sequential(
            nn.Linear(n_filters * n_steps, embed_dim),
            PostNormResidual(
                nn.Sequential(nn.GELU(), nn.Linear(embed_dim, embed_dim)),
                embed_dim,
                drop_prob_proj,
            ),
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

        # Keys of the released pretrain_Enc_eeg_cls.pth / pretrain_Proj_eeg_cls.pth.
        self.mapping = {
            **{f"0.shallownet.{k}": f"tsconv.{k}" for k in self.tsconv.state_dict()},
            "0.projection.0.weight": "projection.weight",
            "0.projection.0.bias": "projection.bias",
            "0.weight": "projector.0.weight",
            "0.bias": "projector.0.bias",
            "1.fn.1.weight": "projector.1.module.1.weight",
            "1.fn.1.bias": "projector.1.module.1.bias",
            "2.weight": "projector.1.layernorm.weight",
            "2.bias": "projector.1.layernorm.bias",
        }

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head; the encoder is kept."""
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)

    def forward(
        self, x: Tensor, return_features: bool = False
    ) -> Union[Tensor, Dict[str, Optional[Tensor]]]:
        """Encode ``x`` of shape (batch, n_chans, n_times).

        Returns the logits (batch, n_outputs), or with ``return_features=True``
        a dict whose ``"features"`` is the aligned embedding (batch, embed_dim).
        """
        x = self.add_kernel_dim(x)
        x = self.tsconv(x)
        x = self.projection(x)
        x = self.flatten(x)
        features = self.projector(x)
        logits = self.final_layer(features)
        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {"features": features, "cls_token": None}
        return logits

"""
CTNet: a convolutional transformer network for EEG-based motor imagery
classification from Wei Zhao et al. (2024).
"""

# Authors: Wei Zhao <zhaowei701@163.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
# License: MIT

from __future__ import annotations

import math

import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegconformer import _FeedForwardBlock, _MultiHeadAttention


class CTNet(EEGModuleMixin, nn.Module):
    """CTNet from Zhao W et al (2024) [ctnet]_.

        .. figure:: https://raw.githubusercontent.com/snailpt/CTNet/main/architecture.png
           :align: center
           :alt: CTNet Architecture

        Overview of the Convolutional Transformer Network for EEG-Based MI Classification




    Parameters
    ----------


    Notes
    -----
    Something that the author wants to highlight. It is only the braindecode
    adaptation from the source code [ctnetcode]_.


    References
    ----------
    .. [ctnet] Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024).
        CTNet: a convolutional transformer network for EEG-based motor imagery
        classification. Scientific Reports, 14(1), 20237.
    .. [ctnetcode] Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024).
        CTNet: a convolutional transformer network for EEG-based motor imagery
        classification.
        https://github.com/snailpt/CTNet
    """

    def __init__(
        self,
        # Base arguments
        n_outputs=None,
        n_chans=None,
        sfreq=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        # Model specific arguments
        activation: nn.Module = nn.GELU,
        heads=4,
        emb_size=40,
        depth=6,
        eeg1_f1=20,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        drop_prob_cnn=0.3,
        drop_prob_posi=0.1,
        drop_prob_final=0.5,
        flatten_eeg1=600,
        # Other ways to initialize the model
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del (
            n_outputs,
            n_chans,
            chs_info,
            n_times,
            sfreq,
        )

        self.number_class, self.number_channel = 2, 22
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1

        # Layers

        self.flatten = nn.Flatten()

        self.cnn = _PatchEmbeddingCNN(
            n_filters_time=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            depth_multiplier=eeg1_D,
            pool_size_1=eeg1_pooling_size1,
            pool_size_2=eeg1_pooling_size2,
            drop_prob=drop_prob_cnn,
            n_chans=self.n_chans,
        )

        self.position = _PositionalEncoding(emb_size, drop_prob=drop_prob_posi)
        self.trans = _TransformerEncoder(heads, depth, emb_size)

        self.flatten_drop_layer = nn.Sequential(
            nn.Flatten(), nn.Dropout(drop_prob_final)
        )

        self.final_layer = nn.Linear(self.flatten_eeg1, self.number_class)

    def forward(self, x: Tensor) -> Tensor:
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        flatten_feature = self.flatten(features)
        out = self.final_layer(flatten_feature)
        return out


class _PatchEmbeddingCNN(nn.Module):
    def __init__(
        self,
        n_filters_time: int = 16,
        kernel_size: int = 64,
        depth_multiplier: int = 2,
        pool_size_1: int = 8,
        pool_size_2: int = 8,
        drop_prob: float = 0.3,
        n_chans: int = 22,
    ):
        super().__init__()
        f2 = depth_multiplier * n_filters_time
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(
                1, n_filters_time, (1, kernel_size), (1, 1), padding="same", bias=False
            ),  # [batch, 22, 1000]
            nn.BatchNorm2d(n_filters_time),
            # channel depth-wise conv
            nn.Conv2d(
                n_filters_time,
                f2,
                (n_chans, 1),
                (1, 1),
                groups=n_filters_time,
                padding="valid",
                bias=False,
            ),  #
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pool_size_1)),
            # pooling acts as slicing to obtain 'patch'
            # along the time dimension as in ViT
            nn.Dropout(drop_prob),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2 to adjust the length
            # of feature into transformer encoder
            nn.AvgPool2d((1, pool_size_2)),
            nn.Dropout(drop_prob),
        )

        self.projection = nn.Sequential(
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


# Residual Add is different
class _ResidualAdd(nn.Module):
    def __init__(self, module, emb_size, drop_p):
        super().__init__()
        self.module = module
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.module(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    _MultiHeadAttention(emb_size, num_heads, drop_p),
                ),
                emb_size,
                drop_p,
            ),
            _ResidualAdd(
                nn.Sequential(
                    _FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                ),
                emb_size,
                drop_p,
            ),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(
            *[_TransformerEncoderBlock(emb_size, heads) for _ in range(depth)]
        )


class _PositionalEncoding(nn.Module):
    def __init__(self, embedding, length=100, drop_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):  # x-> [batch, embedding, length]
        x = x + self.encoding[:, : x.shape[1], :].cuda()
        return self.dropout(x)

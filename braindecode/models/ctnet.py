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
from torch import nn, Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegconformer import _MultiHeadAttention, _TransformerEncoder


class CTNet(EEGModuleMixin, nn.Module):
    """CTNet from Zhao W et al (2024) [ctnet]_.

        .. figure:: https://raw.githubusercontent.com/snailpt/CTNet/main/architecture.png
           :align: center
           :alt: CTNet Architecture

        Overview of the Convolutional Transformer Network for EEG-Based MI Classification


    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors. It is only the adaptation from the source code [ctnetcode]_.



    Parameters
    ----------


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
        drop_prob: float = 0.5,
        activation: nn.Module = nn.ReLU,
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

    def forward(self, x: Tensor) -> Tensor:
        pass


class PatchEmbeddingCNN(nn.Module):
    def __init__(
        self,
        f1=16,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        number_channel=22,
        emb_size=40,
    ):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(
                1, f1, (1, kernel_size), (1, 1), padding="same", bias=False
            ),  # [batch, 22, 1000]
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(
                f1,
                f2,
                (number_channel, 1),
                (1, 1),
                groups=f1,
                padding="valid",
                bias=False,
            ),  #
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )

        self.projection = nn.Sequential(
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


# Residual Add is different
class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)

        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    _MultiHeadAttention(emb_size, num_heads, drop_p),
                ),
                emb_size,
                drop_p,
            ),
            ResidualAdd(
                nn.Sequential(
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                ),
                emb_size,
                drop_p,
            ),
        )


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(
        self,
        heads=4,
        depth=6,
        emb_size=40,
        number_channel=22,
        f1=20,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        **kwargs,
    ):
        super().__init__(
            PatchEmbeddingCNN(
                f1=f1,
                kernel_size=kernel_size,
                D=D,
                pooling_size1=pooling_size1,
                pooling_size2=pooling_size2,
                dropout_rate=dropout_rate,
                number_channel=number_channel,
                emb_size=emb_size,
            ),
        )


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):  # x-> [batch, embedding, length]
        x = x + self.encoding[:, : x.shape[1], :].cuda()
        return self.dropout(x)


class EEGTransformer(nn.Module):
    def __init__(
        self,
        heads=4,
        emb_size=40,
        depth=6,
        database_type="A",
        eeg1_f1=20,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.3,
        eeg1_number_channel=22,
        flatten_eeg1=600,
        **kwargs,
    ):
        super().__init__()
        self.number_class, self.number_channel = 2, 22
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        # print('self.number_channel', self.number_channel)
        self.cnn = BranchEEGNetTransformer(
            heads,
            depth,
            emb_size,
            number_channel=self.number_channel,
            f1=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            D=eeg1_D,
            pooling_size1=eeg1_pooling_size1,
            pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
        )
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = _TransformerEncoder(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.final_layer = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.flatten_eeg1, self.number_class)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)

        features = cnn + trans
        out = self.classification(self.flatten(features))
        return features, out

"""
CTNet: a convolutional transformer network for EEG-based motor imagery
classification from Wei Zhao et al. (2024).
"""

# Authors: Wei Zhao <zhaowei701@163.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
# License: MIT

from __future__ import annotations

import math
from typing import Any

import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegconformer import _FeedForwardBlock, _MultiHeadAttention


class CTNet(EEGModuleMixin, nn.Module):
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
        drop_prob_cnn: float = 0.3,
        drop_prob_posi: float = 0.1,
        drop_prob_final: float = 0.5,
        # other parameters
        heads: int = 4,
        emb_size: int = 40,
        depth: int = 6,
        eeg1_f1: int = 20,
        eeg1_kernel_size: int = 64,
        eeg1_D: int = 2,
        eeg1_pooling_size1: int = 8,
        eeg1_pooling_size2: int = 8,
        flatten_eeg1: int = 600,
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

        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.activation = activation

        # Layers
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        self.flatten = nn.Flatten()

        self.cnn = _PatchEmbeddingCNN(
            n_filters_time=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            depth_multiplier=eeg1_D,
            pool_size_1=eeg1_pooling_size1,
            pool_size_2=eeg1_pooling_size2,
            drop_prob=drop_prob_cnn,
            n_chans=self.n_chans,
            activation=self.activation,
        )

        self.position = _PositionalEncoding(emb_size, drop_prob=drop_prob_posi)
        self.trans = _TransformerEncoder(
            heads, depth, emb_size, activation=self.activation
        )

        self.flatten_drop_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_prob_final),
        )

        self.final_layers = nn.Linear(self.flatten_eeg1, self.n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CTNet model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, n_channels, n_times).

        Returns
        -------
        Tensor
            Output with shape (batch_size, n_outputs).
        """
        x = self.ensuredim(x)
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        flatten_feature = self.flatten(features)
        out = self.final_layers(flatten_feature)
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
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        n_filters_out = depth_multiplier * n_filters_time
        self.cnn_module = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_time,
                kernel_size=(1, kernel_size),
                stride=(1, 1),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(n_filters_time),
            # Channel depth-wise convolution
            nn.Conv2d(
                in_channels=n_filters_time,
                out_channels=n_filters_out,
                kernel_size=(n_chans, 1),
                stride=(1, 1),
                groups=n_filters_time,
                padding="valid",
                bias=False,
            ),
            nn.BatchNorm2d(n_filters_out),
            activation(),
            # First average pooling
            nn.AvgPool2d(kernel_size=(1, pool_size_1)),
            nn.Dropout(drop_prob),
            # Spatial convolution
            nn.Conv2d(
                in_channels=n_filters_out,
                out_channels=n_filters_out,
                kernel_size=(1, 16),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(n_filters_out),
            activation(),
            # Second average pooling
            nn.AvgPool2d(kernel_size=(1, pool_size_2)),
            nn.Dropout(drop_prob),
        )

        self.projection = nn.Sequential(
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Patch Embedding CNN.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, 1, n_channels, n_times).

        Returns
        -------
        Tensor
            Embedded patches of shape (batch_size, num_patches, embedding_dim).
        """
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class _ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module, emb_size: int, drop_p: float):
        super().__init__()
        self.module = module
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass with residual connection.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Tensor
            Output tensor after applying residual connection.
        """
        res = self.module(x, **kwargs)
        out = self.layernorm(self.drop(res) + x)
        return out


class _TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 4,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.attention = _ResidualAdd(
            nn.Sequential(
                _MultiHeadAttention(emb_size, num_heads, drop_p),
            ),
            emb_size,
            drop_p,
        )
        self.feed_forward = _ResidualAdd(
            nn.Sequential(
                _FeedForwardBlock(
                    emb_size,
                    expansion=forward_expansion,
                    drop_p=forward_drop_p,
                    activation=activation,
                ),
            ),
            emb_size,
            drop_p,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass of the transformer encoder block.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Tensor
            Output tensor after transformer encoder block.
        """
        x = self.attention(x, **kwargs)
        x = self.feed_forward(x, **kwargs)
        return x


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        heads: int,
        depth: int,
        emb_size: int,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    emb_size=emb_size,
                    num_heads=heads,
                    activation=activation,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after transformer encoder.
        """
        return self.layers(x)


class _PositionalEncoding(nn.Module):
    def __init__(self, embedding: int, length: int = 100, drop_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the positional encoding.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns
        -------
        Tensor
            Tensor with positional encoding added.
        """
        seq_length = x.size(1)
        encoding = self.encoding[:, :seq_length, :]
        x = x + encoding
        return self.dropout(x)

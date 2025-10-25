"""
CTNet: a convolutional transformer network for EEG-based motor imagery
classification from Wei Zhao et al. (2024).
"""

# Authors: Wei Zhao <zhaowei701@163.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
# License: MIT

from __future__ import annotations

import math
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    FeedForwardBlock,
    MultiHeadAttention,
)


class CTNet(EEGModuleMixin, nn.Module):
    """CTNet from Zhao, W et al (2024) [ctnet]_.

     A Convolutional Transformer Network for EEG-Based Motor Imagery Classification

     .. figure:: https://raw.githubusercontent.com/snailpt/CTNet/main/architecture.png
        :align: center
        :alt: CTNet Architecture

    CTNet is an end-to-end neural network architecture designed for classifying motor imagery (MI) tasks from EEG signals.
    The model combines convolutional neural networks (CNNs) with a Transformer encoder to capture both local and global temporal dependencies in the EEG data.

    The architecture consists of three main components:

    1. **Convolutional Module**:
        - Apply :class:`EEGNet` to perform some feature extraction, denoted here as
        _PatchEmbeddingEEGNet module.

    2. **Transformer Encoder Module**:
        - Utilizes multi-head self-attention mechanisms as EEGConformer but
        with residual blocks.

    3. **Classifier Module**:
        - Combines features from both the convolutional module
        and the Transformer encoder.
        - Flattens the combined features and applies dropout for regularization.
        - Uses a fully connected layer to produce the final classification output.

    Parameters
    ----------
    activation : nn.Module, default=nn.GELU
        Activation function to use in the network.
    heads : int, default=4
        Number of attention heads in the Transformer encoder.
    emb_size : int or None, default=None
        Embedding size (dimensionality) for the Transformer encoder.
    depth : int, default=6
        Number of encoder layers in the Transformer.
    n_filters_time : int, default=20
        Number of temporal filters in the first convolutional layer.
    kernel_size : int, default=64
        Kernel size for the temporal convolutional layer.
    depth_multiplier : int, default=2
        Multiplier for the number of depth-wise convolutional filters.
    pool_size_1 : int, default=8
        Pooling size for the first average pooling layer.
    pool_size_2 : int, default=8
        Pooling size for the second average pooling layer.
    drop_prob_cnn : float, default=0.3
        Dropout probability after convolutional layers.
    drop_prob_posi : float, default=0.1
        Dropout probability for the positional encoding in the Transformer.
    drop_prob_final : float, default=0.5
        Dropout probability before the final classification layer.

    Notes
    -----
    This implementation is adapted from the original CTNet source code
    [ctnetcode]_ to comply with Braindecode's model standards.

    References
    ----------
    .. [ctnet] Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024).
        CTNet: a convolutional transformer network for EEG-based motor imagery
        classification. Scientific Reports, 14(1), 20237.
    .. [ctnetcode] Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024).
        CTNet source code:
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
        activation_patch: nn.Module = nn.ELU,
        activation_transformer: nn.Module = nn.GELU,
        drop_prob_cnn: float = 0.3,
        drop_prob_posi: float = 0.1,
        drop_prob_final: float = 0.5,
        # other parameters
        heads: int = 4,
        emb_size: Optional[int] = 40,
        depth: int = 6,
        n_filters_time: Optional[int] = None,
        kernel_size: int = 64,
        depth_multiplier: Optional[int] = 2,
        pool_size_1: int = 8,
        pool_size_2: int = 8,
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

        self.activation_patch = activation_patch
        self.activation_transformer = activation_transformer
        self.drop_prob_cnn = drop_prob_cnn
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.kernel_size = kernel_size
        self.drop_prob_posi = drop_prob_posi
        self.drop_prob_final = drop_prob_final
        self.heads = heads
        self.depth = depth
        # n_times - pool_size_1 / p
        self.sequence_length = math.floor(
            (
                math.floor((self.n_times - self.pool_size_1) / self.pool_size_1 + 1)
                - self.pool_size_2
            )
            / self.pool_size_2
            + 1
        )

        self.depth_multiplier, self.n_filters_time, self.emb_size = self._resolve_dims(
            depth_multiplier, n_filters_time, emb_size
        )

        # Layers
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        self.flatten = nn.Flatten()

        self.cnn = _PatchEmbeddingEEGNet(
            n_filters_time=self.n_filters_time,
            kernel_size=self.kernel_size,
            depth_multiplier=self.depth_multiplier,
            pool_size_1=self.pool_size_1,
            pool_size_2=self.pool_size_2,
            drop_prob=self.drop_prob_cnn,
            n_chans=self.n_chans,
            activation=self.activation_patch,
        )

        self.position = _PositionalEncoding(
            emb_size=self.emb_size,
            drop_prob=self.drop_prob_posi,
            n_times=self.n_times,
            pool_size=self.pool_size_1,
        )

        self.trans = _TransformerEncoder(
            self.heads,
            self.depth,
            self.emb_size,
            activation=self.activation_transformer,
        )

        self.flatten_drop_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.drop_prob_final),
        )

        self.final_layer = nn.Linear(
            in_features=int(self.emb_size * self.sequence_length),
            out_features=self.n_outputs,
        )

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
        out = self.final_layer(flatten_feature)
        return out

    @staticmethod
    def _resolve_dims(
        depth_multiplier: Optional[int],
        n_filters_time: Optional[int],
        emb_size: Optional[int],
    ) -> tuple[int, int, int]:
        # Basic type/positivity checks for provided values
        for name, val in (
            ("depth_multiplier", depth_multiplier),
            ("n_filters_time", n_filters_time),
            ("emb_size", emb_size),
        ):
            if val is not None:
                if not isinstance(val, int):
                    raise TypeError(f"{name} must be int, got {type(val).__name__}")
                if val <= 0:
                    raise ValueError(f"{name} must be > 0, got {val}")

        missing = [
            k
            for k, v in {
                "depth_multiplier": depth_multiplier,
                "n_filters_time": n_filters_time,
                "emb_size": emb_size,
            }.items()
            if v is None
        ]

        if len(missing) >= 2:
            # Too many unknowns → ambiguous
            raise ValueError(
                "Specify exactly two of {depth_multiplier, n_filters_time, emb_size}; the third will be inferred."
            )

        if len(missing) == 1:
            # Infer the missing one
            if missing[0] == "emb_size":
                assert depth_multiplier is not None and n_filters_time is not None
                emb_size = depth_multiplier * n_filters_time
            elif missing[0] == "n_filters_time":
                assert emb_size is not None and depth_multiplier is not None
                if emb_size % depth_multiplier != 0:
                    raise ValueError(
                        f"emb_size={emb_size} must be divisible by depth_multiplier={depth_multiplier}"
                    )
                n_filters_time = emb_size // depth_multiplier
            else:  # missing depth_multiplier
                assert emb_size is not None and n_filters_time is not None
                if emb_size % n_filters_time != 0:
                    raise ValueError(
                        f"emb_size={emb_size} must be divisible by n_filters_time={n_filters_time}"
                    )
                depth_multiplier = emb_size // n_filters_time

        else:
            # All provided: enforce consistency
            assert (
                depth_multiplier is not None
                and n_filters_time is not None
                and emb_size is not None
            )
            prod = depth_multiplier * n_filters_time
            if prod != emb_size:
                raise ValueError(
                    "`depth_multiplier * n_filters_time` must equal `emb_size`, "
                    f"but got {depth_multiplier} * {n_filters_time} = {prod} != {emb_size}. "
                    "Fix by setting one of: "
                    f"emb_size={prod}, "
                    f"n_filters_time={emb_size // depth_multiplier if emb_size % depth_multiplier == 0 else 'not integer'}, "
                    f"depth_multiplier={emb_size // n_filters_time if emb_size % n_filters_time == 0 else 'not integer'}."
                )

        # Ensure plain ints for the return type
        assert (
            depth_multiplier is not None
            and n_filters_time is not None
            and emb_size is not None
        )
        return depth_multiplier, n_filters_time, emb_size


class _PatchEmbeddingEEGNet(nn.Module):
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
        self.eegnet_module = nn.Sequential(
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
        x = self.eegnet_module(x)
        x = self.projection(x)
        return x


class _ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module, emb_size: int, drop_p: float):
        super().__init__()
        self.module = module
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor) -> Tensor:
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
        res = self.module(x)
        out = self.layernorm(self.drop(res) + x)
        return out


class _TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim_feedforward: int,
        num_heads: int = 4,
        drop_prob: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.attention = _ResidualAdd(
            nn.Sequential(
                MultiHeadAttention(dim_feedforward, num_heads, drop_prob),
            ),
            dim_feedforward,
            drop_prob,
        )
        self.feed_forward = _ResidualAdd(
            nn.Sequential(
                FeedForwardBlock(
                    dim_feedforward,
                    expansion=forward_expansion,
                    drop_p=forward_drop_p,
                    activation=activation,
                ),
            ),
            dim_feedforward,
            drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
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
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        nheads: int,
        depth: int,
        dim_feedforward: int,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    dim_feedforward=dim_feedforward,
                    num_heads=nheads,
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
    def __init__(
        self,
        n_times: int,
        emb_size: int,
        length: int = 100,
        drop_prob: float = 0.1,
        pool_size: int = 8,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.n_times = n_times

        if int(n_times / (pool_size * pool_size)) > length:
            warn(
                "the temporal dimensional is too long for this default length. "
                "The length parameter will be automatically adjusted to "
                "avoid inference issues."
            )
            length = int(n_times / (pool_size * pool_size))

        self.dropout = nn.Dropout(drop_prob)
        self.encoding = nn.Parameter(torch.randn(1, length, emb_size))

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

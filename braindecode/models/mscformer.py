"""
MSCFormer: a multi-scale convolutional transformer network for EEG-based
motor imagery classification from Wei Zhao et al. (2025).
"""

# Authors: Wei Zhao <zhaowei701@163.com>
#          LiQing <325196192+qinxwew@users.noreply.github.com>
#          (braindecode adaptation)
# License: Apache-2.0

from __future__ import annotations

import math

import torch
from einops.layers.torch import Rearrange
from mne.utils import warn
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, MultiHeadAttention


class MSCFormer(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""MSCFormer from Zhao, W et al (2025) [mscformer]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    Multi-scale convolutional transformer network for motor imagery
    brain-computer interface.

    .. figure:: https://raw.githubusercontent.com/snailpt/MSCFormer/main/architecture.png
       :align: center
       :alt: MSCFormer Architecture

    MSCFormer is an end-to-end network for classifying motor imagery (MI)
    tasks from EEG signals. To handle the individual variability of EEG
    signals and the limited receptive field of CNNs, the model combines a
    multi-branch multi-scale convolutional module with a Transformer
    encoder for global feature integration.

    The architecture consists of three main components:

    1. **Multi-scale convolutional module**:

        - Three parallel branches with temporal convolution kernels of
          different widths ``(85, 65, 45)`` followed by a depth-wise
          spatial convolution across channels, batch normalization,
          activation, average pooling and dropout.
        - The branch outputs are concatenated along the feature dimension,
          giving an embedding size of ``3 * n_filters_time``.

    2. **Transformer encoder module**:

        - A learnable class token (BERT-style) is prepended to the
          embedded patches, followed by a learnable positional encoding.
        - A stack of Transformer encoder blocks with post-norm residual
          connections captures global dependencies across temporal
          patches.

    3. **Classifier module**:

        - The Transformer output at the class-token position is passed
          through dropout and a fully connected layer producing the
          classification logits.

    Parameters
    ----------
    kernel_sizes : tuple of int, default=(85, 65, 45)
        Kernel widths of the temporal convolutions in the three
        multi-scale branches.
    n_filters_time : int, default=16
        Number of temporal filters in each convolutional branch; the
        total embedding size is ``n_filters_time * len(kernel_sizes)``.
    pooling_size : int, default=44
        Average pooling size in the convolutional module. The original
        implementation uses 44 for BCI IV-2a and 52 for BCI IV-2b.
    cnn_drop_prob : float, default=0.5
        Dropout probability in the convolutional module. The original
        implementation uses 0.5 for subject-specific training and 0.25
        for cross-subject training.
    activation_cnn : nn.Module, default=nn.ELU
        Activation function in the convolutional branches.
    activation_ffn : nn.Module, default=nn.GELU
        Activation function in the Transformer feed-forward blocks.
    num_heads : int, default=8
        Number of attention heads in the Transformer encoder.
    num_layers : int, default=5
        Number of encoder blocks in the Transformer.
    forward_expansion : int, default=4
        Expansion factor of the feed-forward block hidden layer.
    att_drop_prob : float, default=0.5
        Dropout probability in the attention and feed-forward residual
        blocks.
    att_positional_drop_prob : float, default=0.1
        Dropout probability applied after the positional encoding.
    final_drop_prob : float, default=0.25
        Dropout probability before the final classification layer.
    attention_scale : float or None, default=None
        Multiplier applied to the attention logits before the softmax in
        :class:`braindecode.modules.MultiHeadAttention`. When ``None``
        (default), it reproduces the released source scale
        ``embed_dim ** -0.5`` (i.e. ``1 / sqrt(3 * n_filters_time)``),
        *not* the more common ``head_dim ** -0.5``: the two only coincide
        when ``num_heads == 1``. Numerically verified against the
        original implementation (max abs logit diff < 1e-6 with matched
        weights); passing ``head_dim ** -0.5`` explicitly instead gives a
        max abs logit diff of about 0.035 on a random smoke input.

    Notes
    -----
    This implementation is adapted from the original MSCFormer source
    code [mscformercode]_ to comply with Braindecode's model standards.
    The multi-head attention is the shared
    :class:`braindecode.modules.MultiHeadAttention`, configured through
    ``attention_scale`` to match the original ``embed_dim ** -0.5`` logit
    scaling by default (see the parameter description above).

    References
    ----------
    .. [mscformer] Zhao, W., Zhang, B., Zhou, H. et al. Multi-scale
        convolutional transformer network for motor imagery brain-computer
        interface. Scientific Reports, 15, 12935 (2025).
        https://doi.org/10.1038/s41598-025-96611-5
    .. [mscformercode] Zhao, W. et al. MSCFormer source code:
        https://github.com/snailpt/MSCFormer
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
        kernel_sizes: tuple[int, int, int] = (85, 65, 45),
        n_filters_time: int = 16,
        pooling_size: int = 44,
        cnn_drop_prob: float = 0.5,
        activation_cnn: type[nn.Module] = nn.ELU,
        activation_ffn: type[nn.Module] = nn.GELU,
        num_heads: int = 8,
        num_layers: int = 5,
        forward_expansion: int = 4,
        att_drop_prob: float = 0.5,
        att_positional_drop_prob: float = 0.1,
        final_drop_prob: float = 0.25,
        attention_scale: float | None = None,
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

        self.kernel_sizes = kernel_sizes
        self.n_filters_time = n_filters_time
        self.pooling_size = pooling_size
        self.cnn_drop_prob = cnn_drop_prob
        self.activation_cnn = activation_cnn
        self.activation_ffn = activation_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.att_drop_prob = att_drop_prob
        self.att_positional_drop_prob = att_positional_drop_prob
        self.final_drop_prob = final_drop_prob
        if attention_scale is not None and attention_scale <= 0:
            raise ValueError("attention_scale must be positive or None.")
        self.attention_scale = attention_scale

        self.embed_dim = n_filters_time * len(kernel_sizes)

        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")

        self.cnn = _MultiScalePatchEmbeddingCNN(
            kernel_sizes=kernel_sizes,
            n_filters_time=n_filters_time,
            pooling_size=pooling_size,
            drop_prob=cnn_drop_prob,
            n_chans=self.n_chans,
            activation=activation_cnn,
        )

        # Number of patches produced by the average pooling, plus one
        # class token.
        n_tokens = math.floor((self.n_times - pooling_size) / pooling_size + 1) + 1

        self.position = _PositionalEncoding(
            emb_size=self.embed_dim,
            drop_prob=att_positional_drop_prob,
            n_tokens=n_tokens,
        )

        attention_scale = (
            self.embed_dim**-0.5 if attention_scale is None else attention_scale
        )
        self.trans = _TransformerEncoder(
            num_heads=num_heads,
            depth=num_layers,
            emb_size=self.embed_dim,
            drop_prob=att_drop_prob,
            forward_expansion=forward_expansion,
            activation=activation_ffn,
            attention_scale=attention_scale,
        )

        self.final_layer = nn.Sequential(
            nn.Dropout(final_drop_prob),
            nn.Linear(self.embed_dim, self.n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MSCFormer model.

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
        x = self.cnn(x)

        # Prepend a zero class token (BERT-style); its learned content
        # comes from the positional encoding at index 0.
        cls_token = x.new_zeros((x.shape[0], 1, x.shape[2]))
        x = torch.cat((cls_token, x), dim=1)
        x = x * math.sqrt(self.embed_dim)
        x = self.position(x)
        x = self.trans(x)

        # Use the class-token position as the final feature.
        features = x[:, 0, :]
        out = self.final_layer(features)
        return out


class _MultiScalePatchEmbeddingCNN(nn.Module):
    """Multi-scale convolutional patch embedding.

    Parameters
    ----------
    kernel_sizes : tuple of int
        Kernel widths of the temporal convolutions, one per branch.
    n_filters_time : int
        Number of temporal filters in each branch.
    pooling_size : int
        Average pooling size.
    drop_prob : float
        Dropout probability.
    n_chans : int
        Number of EEG channels; the depth-wise spatial convolution
        covers all of them.
    """

    def __init__(
        self,
        kernel_sizes: tuple[int, int, int],
        n_filters_time: int,
        pooling_size: int,
        drop_prob: float,
        n_chans: int,
        activation: type[nn.Module] = nn.ELU,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    # Temporal convolution, one scale per branch.
                    nn.Conv2d(
                        1,
                        n_filters_time,
                        (1, kernel_size),
                        (1, 1),
                        padding="same",
                    ),
                    # Depth-wise spatial convolution across channels.
                    nn.Conv2d(
                        n_filters_time,
                        n_filters_time,
                        (n_chans, 1),
                        (1, 1),
                        groups=n_filters_time,
                    ),
                    nn.BatchNorm2d(n_filters_time),
                    activation(),
                    nn.AvgPool2d((1, pooling_size)),
                    nn.Dropout(drop_prob),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.projection = nn.Sequential(
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multi-scale convolutional module.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, 1, n_channels, n_times).

        Returns
        -------
        Tensor
            Embedded patches of shape (batch_size, n_patches, embedding_dim).
        """
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.projection(x)
        return x


class _ResidualAdd(nn.Module):
    """Residual connection with dropout and layer normalization."""

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

        Returns
        -------
        Tensor
            Output tensor after applying residual connection.
        """
        res = self.module(x)
        out = self.layernorm(self.drop(res) + x)
        return out


class _TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with post-norm residual connections."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        drop_prob: float,
        forward_expansion: int,
        attention_scale: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.attention = _ResidualAdd(
            MultiHeadAttention(emb_size, num_heads, drop_prob, scale=attention_scale),
            emb_size,
            drop_prob,
        )
        self.feed_forward = _ResidualAdd(
            FeedForwardBlock(
                emb_size,
                expansion=forward_expansion,
                drop_p=drop_prob,
                activation=activation,
            ),
            emb_size,
            drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer encoder block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after transformer encoder block.
        """
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class _TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""

    def __init__(
        self,
        num_heads: int,
        depth: int,
        emb_size: int,
        drop_prob: float,
        forward_expansion: int,
        attention_scale: float,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    drop_prob=drop_prob,
                    forward_expansion=forward_expansion,
                    attention_scale=attention_scale,
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
            Output tensor after the transformer encoder.
        """
        return self.layers(x)


class _PositionalEncoding(nn.Module):
    """Learnable positional encoding.

    Parameters
    ----------
    emb_size : int
        Embedding dimension.
    drop_prob : float
        Dropout probability applied after adding the encoding.
    n_tokens : int
        Number of tokens in the input sequence (including the class
        token). If it exceeds ``length``, the encoding table is enlarged
        to ``n_tokens`` with a warning.
    length : int, default=100
        Length of the learnable encoding table.
    """

    def __init__(
        self,
        emb_size: int,
        drop_prob: float,
        n_tokens: int,
        length: int = 100,
    ):
        super().__init__()
        if n_tokens > length:
            warn(
                "The number of tokens is larger than the default length. "
                "The length parameter will be automatically adjusted to "
                "avoid inference issues."
            )
            length = n_tokens
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

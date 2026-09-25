# Authors: Dingkun Liu (original implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/staraink/MIRepNet
"""MIRepNet downstream model."""

from __future__ import annotations

from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import FeedForwardBlock, MultiHeadAttention


class MIRepNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""MIRepNet from Liu et al. (2026) [liu2026mirepnet]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`
    :bdg-danger:`Foundation Model`

    .. figure:: https://braindecode.org/dev/_static/model/mirepnet.png
       :align: center
       :alt: MIRepNet architecture

       Original MIRepNet overview. Braindecode implements only the downstream
       embedding, Transformer encoder, pooling, and classifier shown at right.

    .. rubric:: Architecture Overview

    1. Temporal and spatial convolutions embed the input EEG.
    2. Pre-normalized Transformer blocks contextualize the resulting tokens.
    3. Token averaging and a linear head produce class logits.

    .. rubric:: Macro Components

    ``MIRepNet.embedding``
        **Operations:** A 25-sample temporal convolution, a spatial convolution
        across all channels, batch normalization, ELU, 75-sample average
        pooling with stride 15, dropout, and a 1x1 projection transform
        ``(batch, n_chans, n_times)`` into
        ``(batch, n_tokens, embed_dim)``.

        **Role:** Extract local temporal-spatial EEG features and form tokens.

    ``MIRepNet.transformer``
        **Operations:** Pre-normalized residual attention and feed-forward
        branches from the released implementation [mirepnetcode]_ process the
        token sequence.

        **Role:** Contextualize each token using the full embedded sequence.

    ``MIRepNet.final_layer``
        **Operations:** Mean pooling produces ``(batch, embed_dim)`` features,
        then a linear projection produces ``(batch, n_outputs)`` logits.

        **Role:** Convert the learned representation into class scores.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Time:** temporal convolution, temporal pooling, and attention encode
      local and global temporal structure.
    - **Channels/space:** one convolution spans all input channels.
    - **Frequency:** no explicit spectral layer is used; spectral content is
      learned from the time-domain input.

    .. rubric:: Additional Mechanisms

    Attention scores use the released source scale ``embed_dim ** -0.5``.
    With ``return_features=True``, mean-pooled tokens are returned in the
    unified feature dictionary instead of being classified.

    This class does not perform the paper's 8--30 Hz filter, 250 Hz resampling,
    channel preparation, or Euclidean alignment. Users are responsible for
    applying that preprocessing before calling the model.

    .. rubric:: Pre-trained weights

    The released downstream checkpoint is re-hosted in Braindecode format on
    the Hugging Face Hub and can be loaded with::

        model = MIRepNet.from_pretrained("braindecode/mirepnet-pretrained")

    It was trained with 45 channels, 1,000 samples at 250 Hz, and three output
    classes. Pass ``n_outputs`` to replace its classification head.

    .. versionadded:: 1.8

    Parameters
    ----------
    embed_dim : int, default=256
        Transformer embedding dimension.
    n_filters_time : int, default=64
        Number of temporal convolution filters.
    n_filters_spat : int, default=128
        Number of spatial convolution filters.
    filter_time_length : int, default=25
        Length of the temporal convolution kernel, in samples.
    pool_time_length : int, default=75
        Length of the temporal average-pooling kernel, in samples.
    pool_time_stride : int, default=15
        Stride of the temporal average pooling, in samples.
    num_layers : int, default=6
        Number of Transformer encoder blocks.
    num_heads : int, default=8
        Number of attention heads.
    feedforward_expansion : int, default=4
        Transformer feed-forward hidden-width multiplier.
    activation : type[nn.Module], default=nn.ELU
        Activation used in the convolutional patch embedding.
    activation_trans : type[nn.Module], default=nn.GELU
        Activation used in Transformer feed-forward blocks.
    drop_prob : float, default=0.5
        Dropout probability in the patch embedding.
    att_drop_prob : float, default=0.5
        Dropout probability in attention and Transformer residual branches.
    feedforward_drop_prob : float, default=0.5
        Dropout probability inside Transformer feed-forward blocks.
    attention_scale : float or None, default=None
        Attention-score multiplier. ``None`` uses the released
        ``embed_dim ** -0.5`` scale.
    return_features : bool, default=False
        Whether ``forward`` returns the unified feature dictionary by default.

    Input shape
    -----------
    ``(batch, n_chans, n_times)``.

    Output shape
    ------------
    By default, ``(batch, n_outputs)`` logits. With ``return_features=True``,
    ``{"features": (batch, embed_dim), "cls_token": None}``.

    References
    ----------
    .. [liu2026mirepnet] Liu et al. (2026). MIRepNet: A pipeline and pre-trained
       model for EEG-based motor imagery classification. Knowledge-Based
       Systems, 343, 115966. https://doi.org/10.1016/j.knosys.2026.115966
    .. [mirepnetcode] Released implementation:
       https://github.com/staraink/MIRepNet
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
        embed_dim: int = 256,
        n_filters_time: int = 64,
        n_filters_spat: int = 128,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_expansion: int = 4,
        activation: type[nn.Module] = nn.ELU,
        activation_trans: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.5,
        att_drop_prob: float = 0.5,
        feedforward_drop_prob: float = 0.5,
        attention_scale: float | None = None,
        return_features: bool = False,
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
        for name, positive_value in (
            ("embed_dim", embed_dim),
            ("n_filters_time", n_filters_time),
            ("n_filters_spat", n_filters_spat),
            ("filter_time_length", filter_time_length),
            ("pool_time_length", pool_time_length),
            ("pool_time_stride", pool_time_stride),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("feedforward_expansion", feedforward_expansion),
        ):
            if positive_value <= 0:
                raise ValueError(f"{name} must be positive.")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        for name, probability in (
            ("drop_prob", drop_prob),
            ("att_drop_prob", att_drop_prob),
            ("feedforward_drop_prob", feedforward_drop_prob),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if attention_scale is not None and attention_scale <= 0:
            raise ValueError("attention_scale must be positive or None.")

        self.embed_dim = embed_dim
        self.return_features = return_features
        self.mapping = {
            "embedding.projection.0.weight": "embedding.projection.weight",
            "embedding.projection.0.bias": "embedding.projection.bias",
            "clshead.weight": "final_layer.weight",
            "clshead.bias": "final_layer.bias",
        }
        self.embedding = _PatchEmbedding(
            n_chans=self.n_chans,
            embed_dim=embed_dim,
            n_filters_time=n_filters_time,
            n_filters_spat=n_filters_spat,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            activation=activation,
            drop_prob=drop_prob,
        )
        self.transformer = _TransformerEncoder(
            depth=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_expansion=feedforward_expansion,
            activation=activation_trans,
            att_drop_prob=att_drop_prob,
            feedforward_drop_prob=feedforward_drop_prob,
            attention_scale=attention_scale,
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: Tensor, return_features: bool | None = None):
        tokens = self.transformer(self.embedding(x))
        features = tokens.mean(dim=1)
        if return_features is None:
            return_features = self.return_features
        if return_features:
            return {"features": features, "cls_token": None}  # nosec B105
        return self.final_layer(features)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(
            self.embed_dim,
            n_outputs,
            device=self.final_layer.weight.device,
            dtype=self.final_layer.weight.dtype,
        )


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        n_chans: int,
        embed_dim: int,
        n_filters_time: int,
        n_filters_spat: int,
        filter_time_length: int,
        pool_time_length: int,
        pool_time_stride: int,
        activation: type[nn.Module],
        drop_prob: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length))
        self.conv2 = nn.Conv2d(n_filters_time, n_filters_spat, kernel_size=(n_chans, 1))
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.elu = activation()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)
        )
        self.dropout = nn.Dropout(drop_prob)
        self.projection = nn.Conv2d(n_filters_spat, embed_dim, kernel_size=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x.flatten(start_dim=2).transpose(1, 2)


class _ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_expansion: int,
        activation: type[nn.Module],
        att_drop_prob: float,
        feedforward_drop_prob: float,
        attention_scale: float | None,
    ):
        if attention_scale is None:
            attention_scale = embed_dim**-0.5
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    MultiHeadAttention(
                        emb_size=embed_dim,
                        num_heads=num_heads,
                        dropout=att_drop_prob,
                        scale=attention_scale,
                    ),
                    nn.Dropout(att_drop_prob),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    FeedForwardBlock(
                        emb_size=embed_dim,
                        expansion=feedforward_expansion,
                        drop_p=feedforward_drop_prob,
                        activation=activation,
                    ),
                    nn.Dropout(att_drop_prob),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        feedforward_expansion: int,
        activation: type[nn.Module],
        att_drop_prob: float,
        feedforward_drop_prob: float,
        attention_scale: float | None,
    ):
        super().__init__(
            *(
                _TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_expansion=feedforward_expansion,
                    activation=activation,
                    att_drop_prob=att_drop_prob,
                    feedforward_drop_prob=feedforward_drop_prob,
                    attention_scale=attention_scale,
                )
                for _ in range(depth)
            )
        )

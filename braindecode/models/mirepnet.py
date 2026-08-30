# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
# Adapted from https://github.com/staraink/MIRepNet
"""MIRepNet downstream model."""

from __future__ import annotations

from warnings import warn

from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import FeedForwardBlock, MultiHeadAttention


class MIRepNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""MIRepNet from Liu et al. (2025) [mirepnet2025]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`
    :bdg-danger:`Foundation Model`

    The input ``(batch, n_chans, n_times)`` is embedded by temporal and spatial
    convolutions into ``(batch, n_tokens, embed_dim)``, processed by pre-norm
    Transformer blocks, mean pooled across tokens to ``(batch, embed_dim)``,
    and mapped to ``(batch, n_outputs)`` logits by a linear head.

    Patch embedding uses a 25-sample temporal convolution, a spatial
    convolution across all input channels, batch normalization, ELU, 75-sample
    average pooling with stride 15, dropout, and a 1x1 projection.

    The Transformer uses pre-normalized residual attention and feed-forward
    branches. Its attention scores use the released source scale
    ``embed_dim ** -0.5``.

    Transformer tokens are mean pooled and passed to the classification head.
    With ``return_features=True``, the pooled tokens are returned instead as
    the unified feature dictionary.

    This class does not perform the paper's 8--30 Hz filter, 250 Hz resampling,
    channel preparation, or Euclidean alignment. Users are responsible for
    applying that preprocessing before calling the model.

    Parameters
    ----------
    n_outputs : int, optional
        Number of output classes.
    n_chans : int, optional
        Number of EEG channels.
    chs_info : list of dict, optional
        Channel information as MNE ``info["chs"]`` entries.
    n_times : int, optional
        Number of time samples in each input window.
    input_window_seconds : float, optional
        Input-window duration in seconds.
    sfreq : float, optional
        Sampling frequency in Hz. Released weights expect 250 Hz input.
    embed_dim : int, default=256
        Transformer embedding dimension.
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
        Dropout probability in the patch embedding and Transformer blocks.
    return_features : bool, default=False
        Whether ``forward`` returns the unified feature dictionary by default.

    Input shape
    -----------
    ``(batch, n_chans, n_times)`` with at least 99 samples.

    Output shape
    ------------
    By default, ``(batch, n_outputs)`` logits. With ``return_features=True``,
    ``{"features": (batch, embed_dim), "cls_token": None}``.

    References
    ----------
    .. [mirepnet2025] Liu et al. (2025). MIRepNet: A Pipeline and Foundation
       Model for EEG-Based Motor Imagery Classification.
    .. [mirepnetcode] Released implementation:
       https://github.com/staraink/MIRepNet
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_expansion: int = 4,
        activation: type[nn.Module] = nn.ELU,
        activation_trans: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.5,
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
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if feedforward_expansion <= 0:
            raise ValueError("feedforward_expansion must be positive.")
        if not 0 <= drop_prob <= 1:
            raise ValueError("drop_prob must be between 0 and 1.")
        if self._n_times is not None and self._n_times < 99:
            raise ValueError("n_times must be at least 99.")
        if sfreq is not None and sfreq != 250:
            warn(
                "MIRepNet's released configuration expects data resampled to 250 Hz; "
                f"received sfreq={sfreq} Hz. Resample the data before calling the model "
                "when using released weights.",
                UserWarning,
            )

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
            activation=activation,
            drop_prob=drop_prob,
        )
        self.transformer = _TransformerEncoder(
            depth=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_expansion=feedforward_expansion,
            activation=activation_trans,
            drop_prob=drop_prob,
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.apply(self._init_weights)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: Tensor, return_features: bool | None = None):
        if x.ndim != 3:
            raise ValueError(
                "MIRepNet expects input with 3 dimensions "
                "(batch, n_chans, n_times)."
            )
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"MIRepNet was configured for {self.n_chans} channels, "
                f"but received {x.shape[1]}."
            )
        if x.shape[2] < 99:
            raise ValueError(
                "MIRepNet requires at least 99 samples in the time dimension."
            )

        tokens = self.transformer(self.embedding(x))
        features = tokens.mean(dim=1)
        if return_features is None:
            return_features = self.return_features
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(
            self.embed_dim,
            n_outputs,
            device=self.final_layer.weight.device,
            dtype=self.final_layer.weight.dtype,
        )
        self._init_weights(self.final_layer)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        n_chans: int,
        embed_dim: int,
        activation: type[nn.Module],
        drop_prob: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(n_chans, 1))
        self.bn = nn.BatchNorm2d(128)
        self.elu = activation()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(drop_prob)
        self.projection = nn.Conv2d(128, embed_dim, kernel_size=(1, 1))

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
        drop_prob: float,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    MultiHeadAttention(
                        emb_size=embed_dim,
                        num_heads=num_heads,
                        dropout=drop_prob,
                        scale=embed_dim**-0.5,
                    ),
                    nn.Dropout(drop_prob),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    FeedForwardBlock(
                        emb_size=embed_dim,
                        expansion=feedforward_expansion,
                        drop_p=drop_prob,
                        activation=activation,
                    ),
                    nn.Dropout(drop_prob),
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
        drop_prob: float,
    ):
        super().__init__(
            *(
                _TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_expansion=feedforward_expansion,
                    activation=activation,
                    drop_prob=drop_prob,
                )
                for _ in range(depth)
            )
        )

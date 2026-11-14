# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Implemented from Yoo et al. (2018) and Ghonchi et al. (2020).
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin


class FNIRSLSTM(EEGModuleMixin, nn.Module):
    r"""FNIRSLSTM from Yoo et al. (2018) [Yoo2018]_ and Ghonchi et al. (2020) [Ghonchi2020]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. versionadded:: 2.1

    Recurrent [Yoo2018]_ or convolutional-recurrent [Ghonchi2020]_ fNIRS
    classifier: an optional stem (``Conv1d``, ``GroupNorm`` with ``gcd(conv_dim, 8)``
    groups, activation, ``MaxPool1d(2)``), a uni- or bidirectional LSTM or GRU,
    temporal pooling of its outputs, ``LayerNorm``, dropout and a linear head. NaN
    samples are read as zero.

    Parameters
    ----------
    use_conv_stem : bool, default=True
        Use the convolutional stem; ``False`` feeds the channels to the RNN.
    conv_dim : int, default=64
        Channels of the stem.
    kernel_size : int, default=5
        Kernel size of the stem convolution.
    rnn_type : {"lstm", "gru"}, default="lstm"
        Recurrent cell.
    hidden_dim : int, default=64
        Units per direction of each recurrent layer.
    n_layers : int, default=2
        Recurrent layers.
    bidirectional : bool, default=True
        Run the recurrent layers in both directions.
    pool_mode : {"mean_max", "mean", "last"}, default="mean_max"
        Time mean and max concatenated, time mean, or the last step (bidirectional:
        the forward state at the last step and the backward state at the first).
    drop_prob : float, default=0.3
        Dropout probability between recurrent layers and before the head.
    activation : type[nn.Module], default=nn.GELU
        Activation of the stem.

    References
    ----------
    .. [Yoo2018] Yoo, S. H., Woo, S. W., & Amad, Z. (2018). Classification of three
       categories from prefrontal cortex using LSTM networks: fNIRS study. ICCAS,
       1141-1146.
    .. [Ghonchi2020] Ghonchi, H., et al. (2020). Deep recurrent-convolutional neural
       network for classification of simultaneous EEG-fNIRS signals. IET Signal
       Processing, 14(3), 142-153.
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        use_conv_stem: bool = True,
        conv_dim: int = 64,
        kernel_size: int = 5,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        hidden_dim: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        pool_mode: Literal["mean_max", "mean", "last"] = "mean_max",
        drop_prob: float = 0.3,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.pool_mode = pool_mode
        self.stem = None
        if use_conv_stem:
            self.stem = nn.Sequential(
                nn.Conv1d(
                    self.n_chans, conv_dim, kernel_size, padding=kernel_size // 2
                ),
                nn.GroupNorm(math.gcd(conv_dim, 8), conv_dim),
                activation(),
                nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            )
        # a KeyError names an unknown rnn_type or pool_mode
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type](
            conv_dim if use_conv_stem else self.n_chans,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop_prob if n_layers > 1 else 0.0,
        )
        n_pooled = {"mean_max": 2, "mean": 1, "last": 1}[pool_mode]
        feature_dim = hidden_dim * (2 if bidirectional else 1) * n_pooled
        self.post_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.final_layer = nn.Linear(feature_dim, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self, x: Tensor, return_features: bool = False, ch_mask: Tensor | None = None
    ) -> Tensor | dict[str, Tensor | None]:
        if ch_mask is not None:
            raise ValueError(
                "FNIRSLSTM does not accept ch_mask: it mixes all channels."
            )
        x = x.nan_to_num()  # NaN samples -> 0
        if self.stem is not None:
            x = self.stem(x)  # (batch, conv_dim, ceil(n_times / 2))
        # (batch, d_in, n_steps) -> (batch, n_steps, d_rnn)
        out, _ = self.rnn(x.transpose(1, 2))
        if self.pool_mode == "mean_max":
            pooled = torch.cat((out.mean(dim=1), out.amax(dim=1)), dim=-1)
        elif self.pool_mode == "mean":
            pooled = out.mean(dim=1)
        elif self.bidirectional:
            h = self.hidden_dim
            pooled = torch.cat((out[:, -1, :h], out[:, 0, h:]), dim=-1)
        else:
            pooled = out[:, -1]
        features = self.dropout(self.post_norm(pooled))  # (batch, feature_dim)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)

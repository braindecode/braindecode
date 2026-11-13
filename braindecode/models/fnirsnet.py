# Authors: Zenghui Wang (original fNIRSNet implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/wzhlearning/fNIRSNet (Apache-2.0).
"""fNIRSNet: Rethinking Delayed Hemodynamic Responses for fNIRS Classification."""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import DepthwiseConv2d


class FNIRSNet(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""fNIRSNet from Wang et al. (2023) [wang2023fnirsnet]_.

    :bdg-success:`Convolution`

    .. versionadded:: 2.1

    A compact fNIRS network: a temporal convolution as long as the delayed
    hemodynamic response (DHR, ``conv1``), a depthwise convolution over all
    channels followed by a 1x1 convolution (``conv2``), each with batch
    normalization and ``activation``, then ``final_layer``. ``mapping`` loads the
    authors' ``fc.*`` weights. Adapted from the authors' code
    (`License <https://github.com/wzhlearning/fNIRSNet/blob/main/LICENSE>`_).

    Parameters
    ----------
    n_dhr_filters : int, default=4
        Number of DHR filters (``num_DHRConv`` in the reference).
    n_dw_filters : int, default=8
        Output channels of the spatial module (``num_DWConv``).
    dhr_kernel_length : int or None, default=None
        Width of the DHR kernel (``DHRConv_width``); ``None`` uses ``n_times``.
    n_chromophores : {1, 2}, default=1
        ``1`` (reference): all channels form one input plane. ``2``: the channels
        are HbO/HbR pairs and form two planes of ``n_chans // 2`` rows.
    channel_order : {"stacked", "interleaved"}, default="stacked"
        Pair order when ``n_chromophores=2``: ``[HbO..., HbR...]`` or
        ``[HbO_1, HbR_1, ...]`` (MNE).
    drop_prob : float, default=0.0
        Dropout probability before ``final_layer``.
    activation : type[nn.Module], default=nn.Sigmoid
        Activation after each batch normalization.

    References
    ----------
    .. [wang2023fnirsnet] Wang, Z., Fang, J., & Zhang, J. (2023). Rethinking
       Delayed Hemodynamic Responses for fNIRS Classification. IEEE
       Transactions on Neural Systems and Rehabilitation Engineering, 31,
       4528-4538. doi:10.1109/TNSRE.2023.3330911
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        n_dhr_filters: int = 4,
        n_dw_filters: int = 8,
        dhr_kernel_length: int | None = None,
        n_chromophores: int = 1,
        channel_order: Literal["stacked", "interleaved"] = "stacked",
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        kernel = dhr_kernel_length or self.n_times
        self.n_chromophores, self.channel_order = n_chromophores, channel_order
        self.conv1 = nn.Conv2d(n_chromophores, n_dhr_filters, (1, kernel))
        self.bn1 = nn.BatchNorm2d(n_dhr_filters)
        self.conv2 = nn.Sequential(
            OrderedDict(
                depth_conv=DepthwiseConv2d(
                    n_dhr_filters, 1, (self.n_chans // n_chromophores, 1)
                ),
                point_conv=nn.Conv2d(n_dhr_filters, n_dw_filters, 1),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_dw_filters)
        self.act = activation()
        self.dropout = nn.Dropout(drop_prob)
        n_features = n_dw_filters * (self.n_times - kernel + 1)
        self.final_layer = nn.Linear(n_features, self.n_outputs)
        self.mapping = {f"fc.{k}": f"final_layer.{k}" for k in ("weight", "bias")}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Logits ``(batch, n_outputs)`` of ``x`` ``(batch, n_chans, n_times)``.

        ``return_features=True`` gives ``{"features", "cls_token": None}``; a
        ``ch_mask`` raises a ``ValueError`` (the spatial kernel spans all channels).
        """
        if ch_mask is not None:
            raise ValueError("FNIRSNet does not accept ch_mask: conv2 spans n_chans.")
        # (batch, n_chans, n_times) -> (batch, n_chromophores, H, n_times)
        b, _, t = x.shape
        if self.n_chromophores == 1:
            x = x.unsqueeze(1)
        elif self.channel_order == "interleaved":
            x = x.reshape(b, -1, 2, t).transpose(1, 2).contiguous()
        else:
            x = x.reshape(b, 2, -1, t)
        x = self.act(self.bn1(self.conv1(x)))  # (batch, n_dhr_filters, H, T')
        x = self.act(self.bn2(self.conv2(x)))  # (batch, n_dw_filters, 1, T')
        features = self.dropout(x.flatten(1))  # (batch, n_dw_filters * T')
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)

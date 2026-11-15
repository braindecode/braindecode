# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/wzhlearning/fNIRSNet (Apache-2.0).
"""fNIRSNet: Rethinking Delayed Hemodynamic Responses for fNIRS Classification."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import DepthwiseConv2d


class FNIRSNet(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""fNIRSNet from Wang et al. (2023) :footcite:`wang2023fnirsnet`.

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
    activation : type[nn.Module], default=nn.Sigmoid
        Activation after each batch normalization.

    References
    ----------
    .. footbibliography::
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
        n_dhr_filters: int = 4,
        n_dw_filters: int = 8,
        dhr_kernel_length: int | None = None,
        activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        kernel = dhr_kernel_length or self.n_times
        self.conv1 = nn.Conv2d(1, n_dhr_filters, (1, kernel))
        self.bn1 = nn.BatchNorm2d(n_dhr_filters)
        self.conv2 = nn.Sequential(
            OrderedDict(
                depth_conv=DepthwiseConv2d(n_dhr_filters, 1, (self.n_chans, 1)),
                point_conv=nn.Conv2d(n_dhr_filters, n_dw_filters, 1),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_dw_filters)
        self.act = activation()
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
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        x = self.act(self.bn1(self.conv1(x)))  # (batch, n_dhr_filters, n_chans, T')
        x = self.act(self.bn2(self.conv2(x)))  # (batch, n_dw_filters, 1, T')
        features = x.flatten(1)  # (batch, n_dw_filters * T')
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)

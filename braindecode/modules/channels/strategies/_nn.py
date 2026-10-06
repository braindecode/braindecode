# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Small torch building blocks shared by the trainable strategies."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(d_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))


def _channel_stats(x: Tensor, used: Tensor) -> Tensor:
    """``(B, C, 2)`` log-variance and log line-length, centred over used channels."""
    w = used.to(x.dtype)
    stats = torch.stack(
        [
            torch.log(x.var(-1) + 1e-12),
            torch.log(x.diff(dim=-1).abs().mean(-1) + 1e-12),
        ],
        -1,
    )
    return stats - (stats * w[:, None]).sum(1, keepdim=True) / w.sum()

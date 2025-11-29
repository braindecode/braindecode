from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
from torch import nn


class StatLayer(nn.Module):
    """
    Generic layer to compute a statistical function along a specified dimension.
    Parameters
    ----------
    stat_fn : Callable
        A function like torch.mean, torch.std, etc.
    dim : int
        Dimension along which to apply the function.
    keepdim : bool, default=True
        Whether to keep the reduced dimension.
    clamp_range : tuple(float, float), optional
        Used only for functions requiring clamping (e.g., log variance).
    apply_log : bool, default=False
        Whether to apply log after computation (used for LogVarLayer).
    """

    def __init__(
        self,
        stat_fn: Callable[..., torch.Tensor],
        dim: int,
        keepdim: bool = True,
        clamp_range: Optional[tuple[float, float]] = None,
        apply_log: bool = False,
    ) -> None:
        super().__init__()
        self.stat_fn = stat_fn
        self.dim = dim
        self.keepdim = keepdim
        self.clamp_range = clamp_range
        self.apply_log = apply_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stat_fn(x, dim=self.dim, keepdim=self.keepdim)
        if self.clamp_range is not None:
            out = torch.clamp(out, min=self.clamp_range[0], max=self.clamp_range[1])
        if self.apply_log:
            out = torch.log(out)
        return out


# make things more simple
def _max_fn(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    return x.max(dim=dim, keepdim=keepdim)[0]


def _power_fn(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    # compute mean of squared values along `dim`
    return torch.mean(x**2, dim=dim, keepdim=keepdim)


MeanLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, torch.mean)
MaxLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, _max_fn)
VarLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, torch.var)
StdLayer: Callable[[int, bool], StatLayer] = partial(StatLayer, torch.std)
LogVarLayer: Callable[[int, bool], StatLayer] = partial(
    StatLayer,
    torch.var,
    clamp_range=(1e-6, 1e6),
    apply_log=True,
)

LogPowerLayer: Callable[[int, bool], StatLayer] = partial(
    StatLayer,
    _power_fn,
    clamp_range=(1e-4, 1e4),
    apply_log=True,
)

import torch
from torch import nn


class MaxNorm(nn.Module):
    def __init__(self, max_norm_val=2.0, eps=1e-5):
        super().__init__()
        self.max_norm_val = max_norm_val
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        norm = X.norm(2, dim=0, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        return X * (number / (denom + self.eps))

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        # Assuming the forward scales X by a factor s,
        # the right inverse would scale it back by 1/s.
        norm = X.norm(2, dim=0, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        scale = number / (denom + self.eps)
        return X / scale


class MaxNormParametrize(nn.Module):
    """
    Enforce a max‑norm constraint on the rows of a weight tensor via parametrization.
    """

    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Renormalize each "row" (dim=0 slice) to have at most self.max_norm
        # L2-norm. This is ``X.renorm(p=2, dim=0, maxnorm=self.max_norm)``
        # written out, with the same 1e-7 epsilon: ``renorm`` fails on Intel
        # Gaudi (HPU), which decomposes it into a broadcast that breaks for
        # some weight shapes. The rescale uses a mask, not a data-dependent
        # branch, so it also traces and compiles without graph breaks.
        # Rows are flattened so the scale broadcasts as (rows, 1); this form
        # is also TorchScript-compatible.
        rows = X.reshape(X.shape[0], -1)
        norm = rows.norm(p=2, dim=1, keepdim=True)
        over = (norm > self.max_norm).to(X.dtype)
        scale = (1.0 - over) + over * (self.max_norm / (norm + 1e-7))
        return (rows * scale).reshape_as(X)

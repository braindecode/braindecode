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
        if max_norm < 0:
            raise ValueError(f"max_norm must be >= 0, got {max_norm}.")
        self.max_norm = max_norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Renormalize each "row" (dim=0 slice) to have at most self.max_norm
        # L2-norm. This is ``X.renorm(p=2, dim=0, maxnorm=self.max_norm)``
        # written out, with the same 1e-7 epsilon: ``renorm`` fails on Intel
        # Gaudi (HPU), which decomposes it into a broadcast that breaks for
        # some weight shapes. Rows are flattened so the scale broadcasts as
        # (rows, 1), which is also TorchScript-compatible.
        if X.numel() == 0:
            return X
        rows = X.reshape(X.shape[0], -1)
        # The norm and scale are computed in float32: in float16 the
        # 1 / (norm + 1e-7) of a small row overflows to inf, which
        # ``torch.where`` would still propagate as NaN into the gradient.
        norm = rows.float().norm(p=2, dim=1, keepdim=True)
        safe_norm = torch.where(norm > self.max_norm, norm, torch.ones_like(norm))
        scale = torch.where(
            norm > self.max_norm,
            self.max_norm / (safe_norm + 1e-7),
            torch.ones_like(norm),
        )
        return (rows * scale.to(X.dtype)).reshape_as(X)

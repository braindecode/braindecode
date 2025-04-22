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
        # Renormalize each "row" (dim=0 slice) to have at most self.max_norm L2-norm
        return X.renorm(p=2, dim=0, maxnorm=self.max_norm)

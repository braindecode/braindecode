import torch
from torch import nn
from torch.nn.utils import parametrize


def _scriptable_weight(fget):
    def weight(self) -> torch.Tensor:
        if torch.jit.is_scripting():  # ParametrizationList.forward, uncached
            x = self.parametrizations.weight.original
            for f in self.parametrizations.weight:
                x = f(x)
            return x
        return fget(self)

    return property(weight)


def _scriptable_weight_norm(fget):
    def weight(self) -> torch.Tensor:
        if torch.jit.is_scripting():  # weight_norm: original0 = g, original1 = v
            p = self.parametrizations.weight
            return p[0](p.original0, p.original1)
        return fget(self)

    return property(weight)


def make_parametrizations_scriptable(module: nn.Module) -> nn.Module:
    """Let :func:`torch.jit.script` read parametrized ``weight`` tensors.

    The getter that :func:`torch.nn.utils.parametrize.register_parametrization`
    installs raises under scripting. This swaps it, on the per-instance class
    that registration creates, for one computing the same chain of
    parametrizations in scripted code; eager calls still use the original getter.
    """
    for m in module.modules():
        if not parametrize.is_parametrized(m, "weight"):
            continue
        fget = type(m).__dict__["weight"].fget
        if fget.__module__ == __name__:  # already swapped
            continue
        plist = m.parametrizations.weight
        if hasattr(plist, "original"):
            setattr(type(m), "weight", _scriptable_weight(fget))
        elif len(plist) == 1 and hasattr(plist, "original1"):
            setattr(type(m), "weight", _scriptable_weight_norm(fget))
    return module


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
        # Promote only low-precision inputs to avoid float16 overflow in
        # the reciprocal and its gradient, without downcasting float64.
        norm_rows = rows
        if X.dtype == torch.float16 or X.dtype == torch.bfloat16:
            norm_rows = rows.float()
        norm = norm_rows.norm(p=2, dim=1, keepdim=True)
        # ``torch.where`` evaluates both branches. Mask the denominator as
        # well as the scale to keep unused reciprocals and gradients safe.
        safe_norm = torch.where(norm > self.max_norm, norm, torch.ones_like(norm))
        scale = torch.where(
            norm > self.max_norm,
            self.max_norm / (safe_norm + 1e-7),
            torch.ones_like(norm),
        )
        return (rows * scale.to(X.dtype)).reshape_as(X)

"""Scaled dot-product attention with the device-specific paths in one place."""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.unused
def _hpu_eager_fused_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    fused_in_lazy: bool = False,
) -> Optional[Tensor]:
    # Lazy mode: None, the graph compiler already fuses torch's SDPA there
    # (unless the caller measured the fused kernel faster for its shapes).
    from habana_frameworks.torch.utils.internal import is_lazy

    if is_lazy() and not fused_in_lazy:
        return None
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    # Strided q/k/v views (heads split off a projection) give wrong outputs and
    # gradients at some shapes; contiguous copies give the right result.
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    return FusedSDPA.apply(query, key, value, None, 0.0, False, scale, "None", False)


@torch.jit.unused
def _hpu_lazy() -> bool:
    from habana_frameworks.torch.utils.internal import is_lazy

    return is_lazy()


def _sdpa_math_dropout(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float,
    scale: Optional[float] = None,
) -> Tensor:
    # torch's math path for SDPA with dropout, op for op, without the pass of
    # ``_safe_softmax`` that zeroes fully masked rows (none without a mask):
    # same values, same dropout draws.
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    half = math.sqrt(scale)
    attn = torch.matmul(query * half, key.transpose(-2, -1) * half)
    attn = F.dropout(attn.softmax(-1), dropout_p, True)
    return torch.matmul(attn, value)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    fused_in_lazy: bool = False,
    exact: bool = False,
) -> Tensor:
    """``F.scaled_dot_product_attention``; Gaudi's fused kernel on HPU.

    Parameters
    ----------
    query, key, value : torch.Tensor
        Shape ``(..., seq, head_dim)``; ``key``/``value`` may have another
        sequence length.
    attn_mask : torch.Tensor, optional
        Boolean (``True`` takes part) or additive mask, as in torch.
    dropout_p : float, default=0.0
        Dropout on the attention weights; pass ``0.0`` in eval mode.
    scale : float, optional
        Score scale; ``None`` is ``1 / sqrt(head_dim)``.
    fused_in_lazy : bool, default=False
        Use Gaudi's fused kernel in HPU lazy mode too.
    exact : bool, default=False
        Run the textbook op order instead of torch's kernels:
        ``softmax((q @ k^T) / sqrt(d)) @ v``, or ``softmax((q @ k^T) * scale) @ v``
        when ``scale`` is given, with dropout on the weights. Bit-identical to
        a hand-written attention; torch's kernels round differently (its math
        path scales ``q`` and ``k`` by ``sqrt(scale)`` each before the matmul,
        its fused CPU kernel runs a blocked softmax), about 1e-6 apart.

    Returns
    -------
    torch.Tensor
        Shape ``(..., seq, head_dim)``.

    In HPU eager mode, torch's SDPA runs op by op (the full
    ``(batch, heads, seq, seq)`` scores in device memory), 2-3.5x slower than
    the bridge's fused kernel on long sequences. The fused kernel is used
    without a mask or dropout (its dropout would draw other masks). In lazy
    mode only with ``fused_in_lazy=True``: the graph compiler fuses torch's
    SDPA too, faster on long sequences, slower on short ones.
    """
    if exact:
        scores = torch.matmul(query, key.transpose(-2, -1))
        if scale is None:
            scores = scores / math.sqrt(query.size(-1))
        else:
            scores = scores * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        weights = scores.softmax(dim=-1)
        if dropout_p > 0.0:
            weights = F.dropout(weights, dropout_p, True)
        return torch.matmul(weights, value)
    if (
        not torch.jit.is_scripting()
        and query.device.type == "hpu"
        and attn_mask is None
        and dropout_p == 0.0
        and (query.dtype == torch.float32 or query.dtype == torch.bfloat16)
    ):
        out = _hpu_eager_fused_sdpa(
            query, key, value, scale=scale, fused_in_lazy=fused_in_lazy
        )
        if out is not None:
            return out
    if (
        not torch.jit.is_scripting()
        and dropout_p > 0.0
        and attn_mask is None
        and (query.dtype == torch.float32 or query.dtype == torch.float64)
        and (
            query.device.type == "cpu"
            or (query.device.type == "hpu" and not _hpu_lazy())
        )
    ):
        # CPU and HPU eager run SDPA with dropout through the math path.
        return _sdpa_math_dropout(query, key, value, dropout_p, scale)
    if scale is None:  # torch 2.0's signature has no ``scale``
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
        )
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale
    )

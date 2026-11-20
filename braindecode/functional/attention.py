"""Scaled dot-product attention with the device-specific paths in one place."""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


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
    fused_in_lazy: Optional[bool] = None,
    exact: bool = False,
) -> Tensor:
    """``F.scaled_dot_product_attention``, or the textbook op order (``exact``).

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
    fused_in_lazy : bool, optional
        No effect; kept so that existing calls still work.

        .. deprecated:: 2.1
            Setting it emits a ``FutureWarning``, and it will be removed in 2.2.
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

    Notes
    -----
    On Intel Gaudi, torch's SDPA runs op by op unless Habana's attention
    setting is on; then the bridge's fused kernel runs it, with or without a
    mask (:ref:`install_gaudi`), dropout included. On CPU, dropout on float32
    or float64 inputs without a mask keeps torch's math path.
    """
    if (
        fused_in_lazy is not None
        and not torch.jit.is_scripting()
        and not torch.compiler.is_compiling()
    ):
        warnings.warn(
            "`fused_in_lazy` has no effect and will be removed in braindecode 2.2.",
            FutureWarning,
            stacklevel=2,
        )
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
        and dropout_p > 0.0
        and attn_mask is None
        and (query.dtype == torch.float32 or query.dtype == torch.float64)
        and query.device.type == "cpu"
    ):
        # The CPU runs SDPA with dropout through the math path.
        return _sdpa_math_dropout(query, key, value, dropout_p, scale)
    if scale is None:  # torch 2.0's signature has no ``scale``
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
        )
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale
    )

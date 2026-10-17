# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Models that moved from a hand-written softmax attention onto
``functional.scaled_dot_product_attention`` keep their old outputs bit for bit.

The old op order is re-implemented here per model and swapped in for the helper;
forward (eval), loss and gradients (train, same seed) must be ``torch.equal``.
"""

import math
import sys

import pytest
import torch
import torch.nn.functional as F

import braindecode.functional.attention as functional_attention
from braindecode.functional import scaled_dot_product_attention
from braindecode.models.util import models_dict, models_mandatory_parameters

from ._fixtures import _get_signal_params


def _old_div_sqrt(q, k, v, dropout_p=0.0, scale=None, **_):
    # AttnSleep, TMSANet, ATCNet: softmax((q @ k^T) / sqrt(d)) @ v
    assert scale is None
    attn = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))).softmax(-1)
    return torch.matmul(F.dropout(attn, dropout_p, True), v)


def _old_mul_scale(q, k, v, dropout_p=0.0, scale=None, **_):
    # TCFormer: softmax((q @ k^T) * scale) @ v; EEGDINO passes q * scale and scale=1
    attn = ((q @ k.transpose(-2, -1)) * scale).softmax(-1)
    return F.dropout(attn, dropout_p, True) @ v


OLD = {
    "AttnSleep": _old_div_sqrt,
    "TMSANet": _old_div_sqrt,
    "ATCNet": _old_div_sqrt,
    "TCFormer": _old_mul_scale,
    "EEGDINO": _old_mul_scale,
}


def _build(name):
    req, sp = next((r, s) for n, r, s in models_mandatory_parameters if n == name)
    torch.manual_seed(0)
    params = _get_signal_params(sp)
    model = models_dict[name](**_get_signal_params(sp, req))
    x = torch.randn(
        2, len(params["chs_info"]), params["n_times"],
        generator=torch.Generator().manual_seed(1),
    )
    return model, x


def _run(model, x, train):
    torch.manual_seed(2)
    model.train(train)
    if not train:
        with torch.no_grad():
            return model(x), None
    model.zero_grad()
    loss = model(x).square().mean()
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize("name", list(OLD))
@pytest.mark.parametrize("train", [False, True])
def test_attention_matches_old_op_order(name, train, monkeypatch):
    model, x = _build(name)
    module = sys.modules[type(model).__module__]
    if name == "EEGDINO":  # attention through modules.TimmAttention
        import braindecode.modules.attention as module

        attr = "_scaled_dot_product_attention"
    else:
        attr = "scaled_dot_product_attention"
    assert getattr(module, attr) is scaled_dot_product_attention
    new_out, new_grads = _run(model, x, train)
    monkeypatch.setattr(module, attr, OLD[name])
    old_out, old_grads = _run(model, x, train)
    assert torch.equal(new_out, old_out), (new_out - old_out).abs().max()
    if train:
        for g_new, g_old in zip(new_grads, old_grads):
            assert torch.equal(g_new, g_old), (g_new - g_old).abs().max()


def test_exact_matches_hand_written_formulas():
    torch.manual_seed(0)
    for d in (4, 8, 12, 16, 25):
        q, k, v = (torch.randn(2, 3, 11, d) for _ in range(3))
        ref = ((q @ k.transpose(-2, -1)) / math.sqrt(d)).softmax(-1) @ v
        assert torch.equal(scaled_dot_product_attention(q, k, v, exact=True), ref)
        ref = ((q @ k.transpose(-2, -1)) * (d**-0.5)).softmax(-1) @ v
        assert torch.equal(
            scaled_dot_product_attention(q, k, v, scale=d**-0.5, exact=True), ref
        )
        # dropout on the weights, same draws as the hand-written version
        torch.manual_seed(1)
        ref = F.dropout(((q @ k.transpose(-2, -1)) / math.sqrt(d)).softmax(-1), 0.3, True) @ v
        torch.manual_seed(1)
        assert torch.equal(
            scaled_dot_product_attention(q, k, v, dropout_p=0.3, exact=True), ref
        )
        assert torch.jit.script(functional_attention.scaled_dot_product_attention)


def test_exact_default_differs_only_in_rounding():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 11, 12) for _ in range(3))
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, exact=True),
        scaled_dot_product_attention(q, k, v),
        rtol=0,
        atol=1e-6,
    )

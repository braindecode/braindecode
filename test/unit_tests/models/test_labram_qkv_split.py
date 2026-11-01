"""Labram split contract against fa859551 frozen select attention (BSD-3-Clause)."""
# ruff: noqa: UP035, UP006, UP007, UP045
# Keep the frozen forward below as it was at fa859551.
import copy
from typing import Optional, Tuple, Union

import pytest
import torch
from torch import nn

from braindecode.models.labram import _Attention, _hpu_eager_fused_sdpa


class _LegacyAttention(_Attention):
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_qkv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias),
                    self.v_bias,
                )
            )
        # Call the ``qkv`` module rather than ``linear`` on its weight, so hooks
        # and adapters attached to it (e.g. LoRA) take part in the forward.
        qkv = self.qkv(x)
        if qkv_bias is not None:
            qkv = qkv + qkv_bias.to(qkv.dtype)  # keep autocast dtype
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        attn_mask: Optional[torch.Tensor] = None
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            attn_mask = (
                relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            )  # 1, nH, Wh*Ww, Wh*Ww

        # Fused kernel, same math. Not with attention dropout in training (it
        # would draw other masks). On Gaudi, torch's SDPA trains ~12 % slower
        # than the plain ops; the bridge's own fused kernel (eager mode, no
        # mask) is used instead.
        fused = not return_attention and (self.attn_drop.p == 0.0 or not self.training)
        out: Optional[torch.Tensor] = None
        if fused and q.device.type != "hpu":
            out = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=self.scale
            )
        elif (
            fused
            and attn_mask is None
            and not torch.jit.is_scripting()
            and (q.dtype == torch.float32 or q.dtype == torch.bfloat16)
        ):
            out = _hpu_eager_fused_sdpa(q, k, v, self.scale)
        if out is None:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = self.attn_drop(attn.softmax(dim=-1))
            if return_attention:
                return attn
            x = attn @ v
        else:
            x = out
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


pytestmark = pytest.mark.usefixtures("acceleration_caller_state")


def _make_model():
    from braindecode.models.labram import Labram

    torch.manual_seed(20261026)
    return Labram(n_times=400, n_chans=4, n_outputs=2, embed_dim=64,
                  num_layers=1, num_heads=4, patch_size=200, neural_tokenizer=True)


def _equal_state(a, b):
    assert list(a.state_dict()) == list(b.state_dict())
    for name, x in a.state_dict().items():
        y = b.state_dict()[name]
        assert (x.shape, x.stride(), x.dtype) == (y.shape, y.stride(), y.dtype)
        assert torch.equal(x, y), name


def _make_full_reference(candidate):
    """Explicit constructor manifest from accepted C baseline; no global patch."""
    reference = copy.deepcopy(candidate)
    installed = []
    for name, child in list(reference.named_modules()):
        if isinstance(child, _Attention):
            assert child.qkv.in_features == 200 and child.num_heads == 4
            assert child.q_bias is None and child.v_bias is None
            assert isinstance(child.q_norm, nn.LayerNorm) and child.q_norm.eps == 1e-6
            assert isinstance(child.k_norm, nn.LayerNorm) and child.k_norm.eps == 1e-6
            assert child.scale == 50 ** -0.5
            assert child.window_size is None
            assert child.attn_drop.p == child.proj_drop.p == 0.0
            legacy = _LegacyAttention(dim=200, num_heads=4, qkv_bias=False,
                                      qk_norm=nn.LayerNorm, qk_scale=None,
                                      attn_drop=0.0, proj_drop=0.0,
                                      window_size=None, attn_head_dim=None)
            legacy.load_state_dict(child.state_dict(), strict=True)
            legacy.to(device=child.qkv.weight.device, dtype=child.qkv.weight.dtype)
            legacy.train(child.training)
            parent, _, leaf = name.rpartition(".")
            reference.get_submodule(parent)._modules[leaf] = legacy
            installed.append(name)
    assert installed == ["blocks.0.attn"]
    _equal_state(candidate, reference)
    return reference


def test_labram_full_select_reference_values_gradients_updates():
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER

    candidate = _make_model().train()
    reference = _make_full_reference(candidate).train()
    names = list(LABRAM_CHANNEL_ORDER[:4])
    x = torch.randn(2, 4, 400)
    results = []
    calls = [0, 0]
    handles = []
    try:
        for side, model in enumerate((candidate, reference)):
            def hook(module, args, output, side=side):
                calls[side] += 1
            handles.append(model.blocks[0].attn.register_forward_hook(hook))
            model.zero_grad(set_to_none=True)
            xx = x.detach().clone().requires_grad_(True)
            torch.manual_seed(123)
            y = model(xx, ch_names=names)
            (y * torch.tensor([0.25, -0.75])).sum().backward()
            results.append((y.detach(), xx.grad.clone(), {
                n: None if p.grad is None else p.grad.clone()
                for n, p in model.named_parameters()}))
        assert calls == [1, 1]
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])
        for name, grad in results[0][2].items():
            other = results[1][2][name]
            assert (grad is None) == (other is None), name
            if grad is not None:
                assert torch.equal(grad, other), name
        for model in (candidate, reference):
            torch.optim.SGD(model.parameters(), lr=0.01, momentum=0).step()
        _equal_state(candidate, reference)
        candidate.load_state_dict(reference.state_dict(), strict=True)
        reference.load_state_dict(candidate.state_dict(), strict=True)
    finally:
        for handle in handles:
            handle.remove()


def test_labram_split_uses_single_unbind_backward():
    """Inspect consumed production output, not the optional unsplit qkv return."""
    model = _Attention(64, num_heads=4, qkv_bias=True, qk_norm=None)
    output = model(torch.randn(2, 17, 64, requires_grad=True))
    pending, seen, kinds = [output.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        kinds.append(type(node).__name__)
        pending.extend(n for n, _ in node.next_functions)
    assert sum(k.startswith("UnbindBackward") for k in kinds) == 1, repr(kinds)
    assert not any(k.startswith("SelectBackward") for k in kinds), repr(kinds)


def test_labram_split_reference_contract():
    """Flags, bias, normalization, head override, window and strided input."""
    from itertools import product

    for bias, norm, head, window, batch in product(
            (False, True), (None, nn.LayerNorm), (None, 8), (None, (2, 2)), (1, 3)):
        kwargs = {"dim": 64, "num_heads": 4, "qkv_bias": bias, "qk_norm": norm,
                      "attn_head_dim": head, "window_size": window}
        candidate = _Attention(**kwargs)
        if bias:
            with torch.no_grad():
                candidate.q_bias.fill_(0.125)
                candidate.v_bias.fill_(-0.25)
        reference = _LegacyAttention(**kwargs)
        reference.load_state_dict(candidate.state_dict(), strict=True)
        x = torch.randn(batch, 5 if window else 17, 128)[..., ::2]
        assert not x.is_contiguous()
        for attention, qkv in product((False, True), repeat=2):
            a = candidate(x, return_attention=attention, return_qkv=qkv)
            b = reference(x, return_attention=attention, return_qkv=qkv)
            for aa, bb in zip(a if isinstance(a, tuple) else (a,),
                              b if isinstance(b, tuple) else (b,)):
                assert torch.equal(aa, bb)
                assert aa.stride() == bb.stride()
            if attention:
                assert isinstance(a, torch.Tensor)  # attention wins over qkv
            elif qkv:
                assert a[1].shape == (3, batch, 4, x.shape[1], head or 16)
                assert a[1][0].untyped_storage().data_ptr() == a[1].untyped_storage().data_ptr()


def test_labram_split_hooks_state_rng_lifecycle():
    import numpy as np

    candidate = _Attention(64, num_heads=4, qkv_bias=True, attn_drop=0.2, proj_drop=0.1)
    reference = _LegacyAttention(64, num_heads=4, qkv_bias=True, attn_drop=0.2, proj_drop=0.1)
    reference.load_state_dict(candidate.state_dict(), strict=True)
    x = torch.randn(2, 17, 64)
    results, counts = [], []
    for model in (candidate, reference):
        handles, calls = [], {"pre": 0, "forward": 0, "backward": 0, "proj": 0}
        def pre(module, args, calls=calls):
            calls["pre"] += 1
            return (args[0] * 0.9,)
        def forward(module, args, output, calls=calls):
            calls["forward"] += 1
            return output + 0.01
        def backward(module, gin, gout, calls=calls):
            calls["backward"] += 1
        def proj(module, args, output, calls=calls):
            calls["proj"] += 1
        try:
            handles.extend([model.qkv.register_forward_pre_hook(pre),
                            model.qkv.register_forward_hook(forward),
                            model.qkv.register_full_backward_hook(backward),
                            model.proj.register_forward_hook(proj)])
            model.train().zero_grad(set_to_none=True)
            torch.manual_seed(345)
            np.random.seed(345)
            xx = x.clone().requires_grad_(True)
            y, qkv = model(xx, return_qkv=True)
            (y.square().sum() + qkv.square().sum() * 0.01).backward()
            results.append((y.detach(), xx.grad.clone(), torch.get_rng_state(),
                            np.random.get_state(), [p.grad.clone() for p in model.parameters()]))
            counts.append(calls.copy())
        finally:
            for handle in handles:
                handle.remove()
    assert counts == [{"pre": 1, "forward": 1, "backward": 1, "proj": 1}] * 2
    for i in (0, 1, 2):
        assert torch.equal(results[0][i], results[1][i])
    assert np.array_equal(results[0][3][1], results[1][3][1])
    for a, b in zip(results[0][4], results[1][4]):
        assert torch.equal(a, b)
    _equal_state(candidate, reference)

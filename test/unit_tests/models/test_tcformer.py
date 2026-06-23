import pytest
import torch
from torch import nn

from braindecode.models.tcformer import (
    _GroupedQueryAttention,
    _apply_rope,
    _build_rotary_cache,
)


def test_rotary_cache_shapes_and_values():
    cos, sin = _build_rotary_cache(head_dim=12, seq_len=17)
    assert cos.shape == (17, 12)
    assert sin.shape == (17, 12)
    # position 0 must be a no-op rotation: cos=1, sin=0
    assert torch.allclose(cos[0], torch.ones(12))
    assert torch.allclose(sin[0], torch.zeros(12))
    # cache uses duplicated halves (NeoX-style): first half == second half
    assert torch.allclose(cos[:, :6], cos[:, 6:])


def test_apply_rope_preserves_shape_and_norm():
    cos, sin = _build_rotary_cache(head_dim=12, seq_len=17)
    q = torch.randn(2, 4, 17, 12)
    k = torch.randn(2, 2, 17, 12)
    q_rot, k_rot = _apply_rope(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_gqa_forward_shape_and_grouping():
    attn = _GroupedQueryAttention(d_model=48, q_heads=4, kv_heads=2)
    # kv_proj must emit 2 (k,v) * kv_heads * head_dim = 2*2*12 = 48
    assert attn.kv_proj.out_features == 48
    assert attn.head_dim == 12
    cos, sin = _build_rotary_cache(12, 17)
    x = torch.randn(3, 17, 48)
    out = attn(x, cos, sin)
    assert out.shape == (3, 17, 48)


def test_gqa_requires_divisibility():
    with pytest.raises(AssertionError):
        _GroupedQueryAttention(d_model=48, q_heads=5, kv_heads=2)  # 48 % 5 != 0
    with pytest.raises(AssertionError):
        _GroupedQueryAttention(d_model=48, q_heads=4, kv_heads=3)  # 4 % 3 != 0


from braindecode.models.tcformer import _TransformerBlock


def test_transformer_block_shape_and_residual():
    blk = _TransformerBlock(d_model=48, q_heads=4, kv_heads=2, mlp_ratio=2,
                            drop_prob=0.0, drop_path_rate=0.0)
    # MLP hidden dim = mlp_ratio * d_model = 96
    assert blk.mlp[0].out_features == 96
    cos, sin = _build_rotary_cache(12, 17)
    x = torch.randn(2, 17, 48)
    out = blk(x, cos, sin)
    assert out.shape == (2, 17, 48)


def test_transformer_block_droppath_is_identity_in_eval():
    blk = _TransformerBlock(48, 4, 2, drop_prob=0.0, drop_path_rate=0.9).eval()
    cos, sin = _build_rotary_cache(12, 17)
    x = torch.randn(2, 17, 48)
    # in eval mode DropPath and Dropout are no-ops -> deterministic, finite
    out_a, out_b = blk(x, cos, sin), blk(x, cos, sin)
    assert torch.allclose(out_a, out_b)


from braindecode.models.tcformer import _ChannelGroupAttention


def test_group_attention_shape_and_per_group_gate():
    sa = _ChannelGroupAttention(in_channels=48, num_groups=3, reduction=4)
    # two grouped 1x1 convs: 48 -> 48//4=12 -> num_groups=3
    assert sa.att_fc1.out_channels == 12
    assert sa.att_fc2.out_channels == 3
    x = torch.randn(2, 48, 1, 17)
    out = sa(x)
    assert out.shape == x.shape


def test_group_attention_gates_are_constant_within_group():
    sa = _ChannelGroupAttention(48, 3, 4).eval()
    x = torch.ones(1, 48, 1, 5)
    ratio = (sa(x) / x)[0, :, 0, 0]  # per-channel gate
    g = 48 // 3
    for grp in range(3):
        block = ratio[grp * g:(grp + 1) * g]
        assert torch.allclose(block, block[0].expand_as(block), atol=1e-6)

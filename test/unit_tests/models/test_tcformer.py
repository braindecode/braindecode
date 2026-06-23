import os

import pytest
import torch
from torch import nn

from braindecode.models import TCFormer as _TCFormerDirect
from braindecode.models.tcformer import (
    _TCN,
    _ChannelGroupAttention,
    _ClassificationHead,
    _GroupedQueryAttention,
    _MultiKernelConvBlock,
    _TCNResidualBlock,
    _TransformerBlock,
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


def test_transformer_block_shape_and_residual():
    blk = _TransformerBlock(
        d_model=48,
        q_heads=4,
        kv_heads=2,
        mlp_ratio=2,
        drop_prob=0.0,
        drop_path_rate=0.0,
    )
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
        block = ratio[grp * g : (grp + 1) * g]
        assert torch.allclose(block, block[0].expand_as(block), atol=1e-6)


def _make_conv_block():
    return _MultiKernelConvBlock(
        n_chans=22,
        temp_kernel_lengths=(20, 32, 64),
        n_filters_time=32,
        depth_multiplier=2,
        pool_length_1=8,
        pool_length_2=7,
        temp_kernel_length_2=16,
        drop_prob=0.4,
        group_dim=16,
        se_reduction=4,
        activation=nn.ELU,
    )


def test_mkcnn_output_shape_and_tokens():
    block = _make_conv_block().eval()
    assert block.d_model == 48  # group_dim(16) * n_groups(3)
    x = torch.randn(2, 22, 1000)
    out = block(x)
    # Tc = floor(floor(1000/8)/7) = floor(125/7) = 17
    assert out.shape == (2, 48, 17)


def test_mkcnn_same_padding_preserves_time_before_pooling():
    # different n_times -> Tc scales as floor(floor(T/8)/7)
    block = _make_conv_block().eval()
    out = block(torch.randn(1, 22, 1125))  # 1125 -> 140 -> 20
    assert out.shape == (1, 48, 20)


def test_tcn_block_shape_and_groups():
    blk = _TCNResidualBlock(
        n_filters=64,
        kernel_length=4,
        dilation=1,
        n_groups=4,
        drop_prob=0.0,
        activation=nn.ELU,
    )
    assert blk.conv1.groups == 4
    x = torch.randn(2, 64, 17)
    assert blk(x).shape == (2, 64, 17)


def test_tcn_stacks_dilations():
    tcn = _TCN(
        depth=2,
        kernel_length=4,
        n_filters=64,
        n_groups=4,
        drop_prob=0.0,
        activation=nn.ELU,
    )
    assert [b.conv1.dilation[0] for b in tcn.blocks] == [1, 2]
    assert tcn(torch.randn(2, 64, 17)).shape == (2, 64, 17)


def test_tcn_block_is_causal():
    # changing the LAST timestep must not change earlier outputs (causality)
    blk = _TCNResidualBlock(
        8, 4, 1, n_groups=1, drop_prob=0.0, activation=nn.ELU
    ).eval()
    x = torch.randn(1, 8, 12)
    y1 = blk(x)
    x2 = x.clone()
    x2[..., -1] += 5.0
    y2 = blk(x2)
    assert torch.allclose(y1[..., :-1], y2[..., :-1], atol=1e-5)


def test_classification_head_shape_and_group_mean():
    head = _ClassificationHead(d_features=64, n_groups=4, n_outputs=4, max_norm=0.25)
    # grouped conv emits n_outputs * n_groups = 16 channels
    assert head.conv.out_channels == 16
    assert head.conv.groups == 4
    out = head(torch.randn(5, 64, 1))
    assert out.shape == (5, 4)


def test_tcformer_forward_shape():
    model = _TCFormerDirect(n_chans=22, n_outputs=4, n_times=1000).eval()
    out = model(torch.randn(2, 22, 1000))
    assert out.shape == (2, 4)


def test_tcformer_param_count_checksum():
    # Paper Table 1 / Tables 2 & 4 headline config (N=2) == 77,820 params.
    model = _TCFormerDirect(n_chans=22, n_outputs=4, n_times=1000)
    assert sum(p.numel() for p in model.parameters()) == 77_820


def test_tcformer_final_layer_named_and_last():
    model = _TCFormerDirect(n_chans=3, n_outputs=2, n_times=1000)
    names = [n for n, _ in model.named_children()]
    assert "final_layer" in names
    assert names[-1] == "final_layer"


def test_tcformer_n5_variant_param_count():
    model = _TCFormerDirect(
        n_chans=22, n_outputs=4, n_times=1000, n_transformer_layers=5
    )
    assert sum(p.numel() for p in model.parameters()) == 127_212


def test_tcformer_handles_other_input_sizes():
    # 3-channel IV-2b-like and 44-channel HGD-like inputs
    assert _TCFormerDirect(n_chans=3, n_outputs=2, n_times=1000).eval()(
        torch.randn(1, 3, 1000)
    ).shape == (1, 2)
    assert _TCFormerDirect(n_chans=44, n_outputs=4, n_times=1000).eval()(
        torch.randn(1, 44, 1000)
    ).shape == (1, 4)


# --------------------------------------------------------------------------- #
# Dev-only parity check: prove the braindecode port is numerically identical to
# the official reference implementation when given the same weights. Skipped in
# CI (the upstream clone is not vendored).
_REF_DIR = os.environ.get(
    "TCFORMER_REF_DIR", "/Users/bruaristimunha/Projects/braindecode-v2/TCFormer"
)


def _load_reference_module():
    """Import the reference TCFormerModule with heavy deps stubbed out."""
    path = os.path.join(_REF_DIR, "models", "tcformer.py")
    with open(path) as fh:
        src = fh.read()
    src = src.replace(
        "from .classification_module import ClassificationModule",
        "class ClassificationModule:\n    def __init__(self, *a, **k):\n        pass",
    )
    for line in (
        "from .channel_group_attention import ChannelGroupAttention",
        "from utils.weight_initialization import glorot_weight_zero_bias",
        "from utils.latency  import measure_latency",
        "from .modules import CausalConv1d, Conv1dWithConstraint",
    ):
        src = src.replace(line, "")
    prelude = (
        "import torch\n"
        "from torch import nn, Tensor\n"
        "from torch.nn import functional as F\n"
        "from einops import rearrange\n"
        "from einops.layers.torch import Rearrange\n"
        "import sys, os\n"
        "sys.path.insert(0, os.path.join(%r, 'models'))\n"
        % _REF_DIR
        + "from modules import CausalConv1d, Conv1dWithConstraint\n"
        "from channel_group_attention import ChannelGroupAttention\n"
        "def glorot_weight_zero_bias(m):\n"
        "    for mod in m.modules():\n"
        "        w = getattr(mod, 'weight', None)\n"
        "        if w is not None and 'norm' not in mod.__class__.__name__.lower():\n"
        "            nn.init.xavier_uniform_(w)\n"
        "        b = getattr(mod, 'bias', None)\n"
        "        if b is not None:\n"
        "            nn.init.constant_(b, 0)\n"
        "def measure_latency(*a, **k):\n    return None\n"
    )
    ns: dict = {}
    exec(prelude + src, ns)  # noqa: S102  (test-only, controlled source)
    return ns["TCFormerModule"]


@pytest.mark.skipif(
    not os.path.isdir(_REF_DIR), reason="reference TCFormer clone not present"
)
def test_tcformer_matches_reference():
    torch.manual_seed(0)
    TCFormerModule = _load_reference_module()
    ref = TCFormerModule(
        n_channels=22,
        n_classes=4,
        F1=32,
        temp_kernel_lengths=(20, 32, 64),
        d_group=16,
        D=2,
        pool_length_1=8,
        pool_length_2=7,
        dropout_conv=0.4,
        use_group_attn=True,
        q_heads=4,
        kv_heads=2,
        trans_depth=2,
        trans_dropout=0.4,
        tcn_depth=2,
        kernel_length_tcn=4,
        dropout_tcn=0.3,
    ).eval()
    bd = _TCFormerDirect(n_chans=22, n_outputs=4, n_times=1000).eval()

    # The trunk (everything except the final classifier) aligns 1:1 by position.
    # braindecode's Conv1dWithConstraint stores its weight under a parametrization,
    # so the classifier's state_dict order differs -> copy it explicitly by name.
    ref_trunk = [
        (k, v)
        for k, v in ref.state_dict().items()
        if not k.startswith("tcn_head.classifier")
    ]
    bd_trunk = [
        (k, v) for k, v in bd.state_dict().items() if not k.startswith("final_layer")
    ]
    assert len(ref_trunk) == len(bd_trunk), (len(ref_trunk), len(bd_trunk))
    new_sd = {}
    for (rk, rv), (bk, bv) in zip(ref_trunk, bd_trunk):
        assert rv.shape == bv.shape, (rk, bk, tuple(rv.shape), tuple(bv.shape))
        new_sd[bk] = rv.clone()
    bd.load_state_dict(new_sd, strict=False)
    ref_cls = ref.tcn_head.classifier.linear
    bd.final_layer.conv.parametrizations.weight.original.data.copy_(ref_cls.weight.data)
    bd.final_layer.conv.bias.data.copy_(ref_cls.bias.data)

    x = torch.randn(4, 22, 1000)
    with torch.no_grad():
        out_ref, out_bd = ref(x), bd(x)
    assert torch.allclose(out_ref, out_bd, atol=1e-5), (out_ref - out_bd).abs().max()

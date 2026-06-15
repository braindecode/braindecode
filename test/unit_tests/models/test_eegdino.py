import torch
import torch.nn as nn

from braindecode.models.eegdino import _TransformerEncoderLayer


def test_transformer_layer_preserves_shape_and_bias_keys():
    layer = _TransformerEncoderLayer(
        d_model=200, num_heads=8, dim_feedforward=512, activation=nn.GELU, drop_prob=0.0
    )
    x = torch.randn(2, 17, 200)  # (B, tokens, D)
    out = layer(x)
    assert out.shape == (2, 17, 200)
    keys = set(layer.state_dict().keys())
    # BEiT-style fused qkv (no bias) + separate q/v bias -> matches released checkpoint keys
    assert "attn.qkv.weight" in keys
    assert "attn.qkv.bias" not in keys
    assert {"attn.q_bias", "attn.v_bias"}.issubset(keys)
    assert layer.state_dict()["attn.qkv.weight"].shape == (600, 200)
    assert {"mlp.fc1.weight", "mlp.fc2.weight"}.issubset(keys)

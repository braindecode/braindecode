import torch
import torch.nn as nn

from braindecode.models.eegdino import (
    EEGDINO,
    EEGDINO_CONFIGS,
    _PatchEmbedding,
    _TransformerEncoderLayer,
)


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


def test_patch_embedding_shapes_and_keys_cpu():
    pe = _PatchEmbedding(
        feature_size=200, num_channels=19, patch_size=200,
        conv_channels=(25, 25, 25), groups=5, drop_prob=0.0,
    )
    x = torch.randn(3, 16, 5, 200)  # (B, C<=19, P, patch_size); runs on CPU (no .cuda())
    out = pe(x)
    assert out.shape == (3, 16, 5, 200)  # (B, C, P, D)
    keys = set(pe.state_dict().keys())
    for k in [
        "time_encoding.0.weight", "proj_in.0.weight", "proj_in.3.weight",
        "proj_in.6.weight", "spectral_proj.0.weight", "channel_embedding.weight",
    ]:
        assert k in keys
    assert pe.state_dict()["spectral_proj.0.weight"].shape == (200, 101)  # rfft(200)->101
    assert pe.state_dict()["channel_embedding.weight"].shape == (200, 19)


def test_eegdino_instantiates_small_and_has_final_layer_last():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)  # defaults = Small
    children = list(model.named_children())
    last_two = [name for name, _ in children][-2:]
    assert "final_layer" in last_two
    assert model.get_output_shape() == (1, 4)


def test_eegdino_medium_preset_dims():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000, **EEGDINO_CONFIGS["medium"])
    assert model.feature_size == 512
    assert len(model.encoder_layers) == 16

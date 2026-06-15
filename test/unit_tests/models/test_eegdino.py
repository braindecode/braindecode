import pytest
import torch
import torch.nn as nn

from braindecode.models.eegdino import (
    EEGDINO,
    EEGDINO_CONFIGS,
    _Attention,
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


@pytest.mark.parametrize("n_times", [1000, 600])  # 600 % 200 == 0
def test_forward_divisible(n_times):
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=n_times)
    out = model(torch.randn(2, 16, n_times))
    assert out.shape == (2, 4)


def test_forward_pads_non_divisible_with_warning():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=950)  # not a multiple of 200
    with pytest.warns(UserWarning, match="zero-padded"):
        out = model(torch.randn(2, 16, 950))
    assert out.shape == (2, 4)


def test_forward_accepts_prepatched_4d_input():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)
    out = model(torch.randn(2, 16, 5, 200))  # (B, C, P, patch_size)
    assert out.shape == (2, 4)


def test_return_features_dict_contract():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)
    out = model(torch.randn(2, 16, 1000), return_features=True)
    assert isinstance(out, dict)
    assert set(out) == {"features", "cls_token"}
    assert out["features"].shape[0] == 2
    assert out["features"].shape[-1] == model.feature_size
    assert isinstance(out["cls_token"], torch.Tensor)
    assert out["cls_token"].shape == (2, model.feature_size)


def test_default_forward_returns_tensor():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)
    assert isinstance(model(torch.randn(2, 16, 1000)), torch.Tensor)


def test_reset_head_changes_output_dim():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)
    model.reset_head(7)
    assert model.n_outputs == 7
    assert model(torch.randn(2, 16, 1000)).shape == (2, 7)


def test_attention_large_preset_indivisible_dims_forward():
    # Large preset: dim=1024, heads=24 -> 1024 % 24 != 0. Decoupling head_dim
    # from model dim must let the forward pass run without raising.
    attn = _Attention(dim=1024, num_heads=24)
    out = attn(torch.randn(2, 5, 1024))
    assert out.shape == (2, 5, 1024)


def test_attention_small_medium_checkpoint_shapes_unchanged():
    # For S (dim=200, heads=8) and M (dim=512, heads=16), all_head_dim == dim,
    # so released-checkpoint key shapes must remain identical.
    small = _Attention(dim=200, num_heads=8).state_dict()
    assert small["qkv.weight"].shape == (600, 200)
    assert small["q_bias"].shape == (200,)
    assert small["v_bias"].shape == (200,)
    assert small["proj.weight"].shape == (200, 200)

    medium = _Attention(dim=512, num_heads=16).state_dict()
    assert medium["qkv.weight"].shape == (1536, 512)
    assert medium["q_bias"].shape == (512,)
    assert medium["v_bias"].shape == (512,)
    assert medium["proj.weight"].shape == (512, 512)


def test_encoder_state_dict_keys_are_loadable(tmp_path):
    """The encoder submodules must accept a state_dict whose keys are the
    EEG-DINO encoder keys (patch_embedding.*, encoder_layers.*, global_tokens)."""
    model = EEGDINO(n_chans=19, n_outputs=4, n_times=1000)
    encoder_prefixes = ("patch_embedding.", "encoder_layers.", "global_tokens")
    enc_keys = {
        k for k in model.state_dict()
        if k.startswith(encoder_prefixes)
    }
    # Simulate a released checkpoint: same keys under module.teacher.
    fake_ckpt = {
        "global_step": 1,
        "state_dict": {
            f"module.teacher.{k}": model.state_dict()[k].clone() for k in enc_keys
        },
    }
    from scripts.convert_eegdino_checkpoints import extract_encoder_state_dict

    sd = extract_encoder_state_dict(fake_ckpt, source="teacher")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # every extracted key must exist in the model (no unexpected encoder keys)
    assert not unexpected
    # all extracted keys are exactly the encoder keys
    assert set(sd) == enc_keys

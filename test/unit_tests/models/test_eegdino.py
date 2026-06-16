import pytest
import torch

from braindecode.models.eegdino import (
    EEGDINO,
    EEGDINO_CONFIGS,
    _Attention,
    _PatchEmbedding,
)

# EEG-DINO-specific behaviour only. Instantiation, output shape, ``final_layer``,
# ``activation``/``drop_prob`` params and the ``return_features``/``reset_head``
# contracts are already covered by the auto-parametrized suites
# (``test_integration.py`` and ``test_return_features.py``).


def test_patch_embedding_cpu_shapes_keys_and_emb_dim():
    patch_embed = _PatchEmbedding(
        patch_size=200,
        channels_kernel_stride_padding_norm=EEGDINO_CONFIGS["small"][
            "channels_kernel_stride_padding_norm"
        ],
        num_channels=19,
        drop_prob=0.0,
    )
    assert patch_embed.emb_dim == 200  # derived from the conv configuration
    out = patch_embed(torch.randn(3, 16, 5, 200))  # runs on CPU (no .cuda())
    assert out.shape == (3, 16, 5, 200)
    keys = set(patch_embed.state_dict())
    assert {
        "proj_in.0.weight",
        "proj_in.3.weight",
        "proj_in.6.weight",
        "spectral_proj.0.weight",
        "channel_embedding.weight",
        "time_encoding.0.weight",
    } <= keys
    assert patch_embed.state_dict()["spectral_proj.0.weight"].shape == (200, 101)
    assert patch_embed.state_dict()["channel_embedding.weight"].shape == (200, 19)


def test_attention_indivisible_heads_and_beit_keys():
    # Large preset (emb_dim=1024, nhead=24) is not divisible; decoupling head_dim
    # must let the forward run. Also guard the BEiT key layout used by the weights.
    attn = _Attention(emb_dim=1024, nhead=24)
    assert attn(torch.randn(2, 5, 1024)).shape == (2, 5, 1024)
    keys = set(attn.state_dict())
    assert {"qkv.weight", "q_bias", "v_bias"} <= keys
    assert "qkv.bias" not in keys


def test_forward_pads_non_divisible_with_warning():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=950)  # 950 % 200 != 0
    with pytest.warns(UserWarning, match="zero-padded"):
        out = model(torch.randn(2, 16, 950))
    assert out.shape == (2, 4)


def test_forward_accepts_prepatched_4d_input():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)
    out = model(torch.randn(2, 16, 5, 200))  # (batch, n_chans, n_patches, patch_size)
    assert out.shape == (2, 4)


def test_medium_preset_derives_dims():
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000, **EEGDINO_CONFIGS["medium"])
    assert model.emb_dim == 512
    assert len(model.encoder_layers) == 16


def test_from_pretrained_local_roundtrip(tmp_path):
    pytest.importorskip("huggingface_hub")
    model = EEGDINO(n_chans=19, n_outputs=2, n_times=1000).eval()
    save_dir = tmp_path / "eegdino-small"
    model.save_pretrained(save_dir)
    reloaded = EEGDINO.from_pretrained(save_dir).eval()
    x = torch.randn(1, 19, 1000)
    assert torch.allclose(model(x), reloaded(x), atol=1e-5)
    assert EEGDINO.from_pretrained(save_dir, n_outputs=6)(x).shape == (1, 6)

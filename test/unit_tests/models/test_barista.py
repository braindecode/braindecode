"""BaRISTA geometry, serialization, and architecture regression checks."""

import json

import pytest
import torch
from torch import nn
from torch.export import export
from torch.nn import functional as F

from braindecode.models import BaRISTA
from braindecode.models.barista import _SamePadConv, _Transformer


def _model(**kwargs):
    return BaRISTA(
        **{
            "n_chans": 2,
            "n_outputs": 3,
            "n_times": 33,
            "patch_size": 8,
            "d_model": 8,
            "num_heads": 2,
            "n_layers": 1,
            "cnn_depth": 1,
            "cnn_channels": 2,
            "spatial_scale": "none",
            "drop_prob": 0.0,
            **kwargs,
        }
    )


def test_barista_reset_head_preserves_state_and_serialization(tmp_path):
    model = _model().double().eval()
    model.reset_head(5)
    assert model.final_layer.weight.dtype == torch.float64
    assert (
        model.final_layer.weight.device
        == next(model.temporal_encoder.parameters()).device
    )
    assert not model.final_layer.training
    assert model.n_outputs == model.get_config()["n_outputs"] == 5
    x = torch.randn(2, 2, 33, dtype=torch.float64)
    assert model(x).shape == (2, 5)
    restored = (
        BaRISTA.from_config(json.loads(json.dumps(model.get_config()))).double().eval()
    )
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x))
    with pytest.raises(ValueError, match="positive"):
        model.reset_head(0)
    assert model.n_outputs == 5

    meta_model = _model().to("meta")
    meta_model.reset_head(5)
    assert meta_model.final_layer.weight.device.type == "meta"

    # Local Hub serialization must also remember the replacement head.
    pytest.importorskip("huggingface_hub")
    model.float().save_pretrained(tmp_path)
    restored = BaRISTA.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(x.float()), model(x.float()))


@pytest.mark.parametrize("scale", ["coords", "parcels", "lobes", "none"])
@pytest.mark.parametrize("pooling", ["learned", "mean"])
def test_barista_spatial_scales_roundtrip_and_export(scale, pooling):
    kwargs = {"spatial_scale": scale, "pooling": pooling}
    if scale == "coords":
        kwargs["chs_info"] = [
            {"ch_name": f"E{i}", "loc": [0.01 * (i + 1), 0.02, 0.03] + [0.0] * 9}
            for i in range(2)
        ]
    elif scale in ("parcels", "lobes"):
        kwargs["spatial_regions"] = ["left-amygdala", "UNKNOWN"]
    model = _model(**kwargs).eval()
    x = torch.randn(1, 2, 33)
    features = model(x, return_features=True)
    assert features["cls_token"] is None
    torch.testing.assert_close(model.final_layer(features["features"]), model(x))
    # One trailing sample is discarded, while a whole extra patch is rejected.
    torch.testing.assert_close(model(x[..., :32]), model(x))
    with pytest.raises(ValueError, match="temporal patches"):
        model(torch.randn(1, 2, 40))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 3, 33))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 33))

    restored = BaRISTA.from_config(json.loads(json.dumps(model.get_config()))).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x))
    torch.testing.assert_close(torch.jit.script(model)(x), model(x))
    torch.testing.assert_close(export(model, (x,), strict=False).module()(x), model(x))
    if scale in ("parcels", "lobes"):
        assert model.spatial_emb.indices.tolist() == [[1, 0]]
        assert torch.count_nonzero(model.spatial_emb()[1]) == 0
    model.train()
    model(x).square().sum().backward()
    assert torch.isfinite(model.final_layer.weight.grad).all()
    if scale in ("parcels", "lobes"):
        assert torch.count_nonzero(model.spatial_emb.tables[0].weight.grad[0]) == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_heads": 0}, "num_heads must be positive"),
        ({"n_layers": -1}, "n_layers must be positive"),
        ({"patch_size": 0}, "patch_size must be positive"),
        ({"n_times": 7}, "at least one full patch"),
        ({"d_model": 10, "num_heads": 3}, "divisible"),
        ({"d_model": 6}, "must be even"),
        ({"cnn_depth": -1}, "non-negative"),
        ({"cnn_kernel_size": 0}, "cnn_kernel_size must be positive"),
        ({"coord_bins": 0}, "coord_bins must be positive"),
        ({"pooling": "invalid"}, "pooling must"),
        ({"spatial_scale": "invalid"}, "spatial_scale must"),
        ({"spatial_scale": "coords"}, "chs_info"),
        ({"spatial_scale": "parcels"}, "spatial_regions must"),
        ({"spatial_scale": "lobes", "spatial_regions": ["UNKNOWN"]}, "1 labels"),
    ],
)
def test_barista_rejects_invalid_geometry(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


@pytest.mark.parametrize(
    "bad_channel",
    [
        {"ch_name": "E2"},
        {"ch_name": "E2", "loc": [float("nan"), 0.02, 0.03]},
        {"ch_name": "E2", "loc": [0.01, 0.02]},
        {"ch_name": "E2", "loc": [0.01, 0.02, 0.03], "coord_frame": 5},
    ],
)
def test_barista_rejects_incomplete_or_mixed_coordinates(bad_channel):
    with pytest.raises(
        ValueError, match="position for every channel|same coordinate frame"
    ):
        _model(
            spatial_scale="coords",
            chs_info=[
                {"ch_name": "E1", "loc": [0.01, 0.02, 0.03], "coord_frame": 4},
                bad_channel,
            ],
        )


def test_barista_spatial_indices_and_unknown_labels():
    model = _model(
        spatial_scale="coords",
        chs_info=[
            {"ch_name": "E1", "loc": [0.01, -0.02, 0.03]},
            {"ch_name": "E2", "loc": [1.0, -1.0, 0.001]},
        ],
    )
    assert model.spatial_emb.indices.tolist() == [[90, 0], [120, 199], [70, 99]]
    with pytest.warns(UserWarning, match="did not recognise"):
        model = _model(
            spatial_scale="parcels", spatial_regions=["ctx-lh-G_front_middle", "typo"]
        )
    assert model.spatial_emb.indices.tolist() == [[18, 0]]
    assert torch.count_nonzero(model.spatial_emb()[1]) == 0


@pytest.mark.parametrize("kernel,dilation", [(2, 1), (3, 1), (2, 2), (4, 1)])
def test_barista_convolution_preserves_reference_alignment(kernel, dilation):
    conv = _SamePadConv(1, 2, kernel, dilation)
    x = torch.arange(9.0).reshape(1, 1, 1, 9)
    pad = ((kernel - 1) * dilation + 1) // 2
    expected = F.conv2d(
        F.pad(x, (pad, pad)), conv.conv.weight, conv.conv.bias, dilation=(1, dilation)
    )[..., : x.shape[-1]]
    torch.testing.assert_close(conv(x), expected)


def test_barista_transformer_matches_explicit_attention_and_gating():
    """Guard temporal RoPE ordering, half-split rotation, and the fused GLU."""
    torch.manual_seed(42)
    model = _Transformer(8, 1, 2, 2, 0.0, nn.GELU, 3, 2).eval()
    x = torch.randn(2, 6, 8, requires_grad=True)
    layer = model.layers[0]
    norm = layer.norm1(x)
    q, k, v = layer.self_attn.qkv_proj(norm).chunk(3, -1)
    q, k, v = [t.reshape(2, 6, 2, 4).transpose(1, 2) for t in (q, k, v)]
    positions = torch.arange(3).repeat_interleave(2)
    angles = positions[:, None] * torch.tensor([1.0, 0.01])
    angles = torch.cat((angles, angles), -1)

    def rotate(t):
        rotated = torch.cat((-t[..., 2:], t[..., :2]), -1)
        return t * angles.cos() + rotated * angles.sin()

    q, k = rotate(q), rotate(k)
    attention = ((q @ k.transpose(-1, -2)) / 2).softmax(-1) @ v
    residual = x + layer.self_attn.o_proj(attention.transpose(1, 2).reshape(2, 6, 8))
    up_weight, gate_weight = layer.mlp[0].weight.chunk(2)
    up_bias, gate_bias = layer.mlp[0].bias.chunk(2)
    hidden = layer.norm2(residual)
    hidden = F.linear(hidden, up_weight, up_bias) * F.gelu(
        F.linear(hidden, gate_weight, gate_bias)
    )
    expected = model.norm(residual + layer.mlp[3](hidden))
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-4)

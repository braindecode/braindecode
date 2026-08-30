# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import pytest
import torch

from braindecode.models.mirepnet import MIRepNet


def _model(**kwargs):
    return MIRepNet(
        n_chans=3,
        n_outputs=2,
        n_times=256,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        feedforward_expansion=2,
        **kwargs,
    )


def test_mirepnet_output_contract():
    model = _model().eval()
    x = torch.randn(2, 3, 256)

    with torch.no_grad():
        logits = model(x)
        features = model(x, return_features=True)

    assert logits.shape == (2, 2)
    assert features["features"].shape == (2, 16)
    assert features["cls_token"] is None


def test_mirepnet_constructor_return_features():
    model = _model(return_features=True).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 256))

    assert output["features"].shape == (2, 16)
    assert output["cls_token"] is None


def test_mirepnet_reset_head():
    model = _model()
    old_dtype = model.final_layer.weight.dtype
    old_device = model.final_layer.weight.device

    model.reset_head(5)

    assert model.n_outputs == 5
    assert model.final_layer.out_features == 5
    assert model.final_layer.weight.dtype == old_dtype
    assert model.final_layer.weight.device == old_device
    assert model(torch.randn(2, 3, 256)).shape == (2, 5)


def test_mirepnet_source_attention_scale_and_token_count():
    model = _model().eval()
    attention = model.transformer[0][0].fn[1]

    assert attention.scale == 16**-0.5
    with torch.no_grad():
        tokens = model.embedding(torch.randn(2, 3, 256))
    assert tokens.shape == (2, 11, 16)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"embed_dim": 0}, "embed_dim must be positive"),
        ({"num_layers": 0}, "num_layers must be positive"),
        ({"num_heads": 0}, "num_heads must be positive"),
        (
            {"embed_dim": 15, "num_heads": 4},
            "embed_dim must be divisible by num_heads",
        ),
        (
            {"feedforward_expansion": 0},
            "feedforward_expansion must be positive",
        ),
        ({"drop_prob": -0.1}, "drop_prob must be between 0 and 1"),
        ({"drop_prob": 1.1}, "drop_prob must be between 0 and 1"),
        ({"n_times": 98}, "n_times must be at least 99"),
    ],
)
def test_mirepnet_invalid_constructor_arguments(kwargs, message):
    base = {
        "n_chans": 3,
        "n_outputs": 2,
        "n_times": 256,
        "embed_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "feedforward_expansion": 2,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MIRepNet(**base)


@pytest.mark.parametrize(
    "shape, message",
    [
        ((3, 256), "3 dimensions"),
        ((2, 4, 256), "3 channels"),
        ((2, 3, 98), "at least 99 samples"),
    ],
)
def test_mirepnet_invalid_input(shape, message):
    with pytest.raises(ValueError, match=message):
        _model()(torch.randn(*shape))


def test_mirepnet_warns_for_non_released_sampling_rate():
    with pytest.warns(UserWarning, match="250 Hz"):
        MIRepNet(
            n_chans=3,
            n_outputs=2,
            n_times=256,
            sfreq=200,
            embed_dim=16,
            num_layers=2,
            num_heads=4,
            feedforward_expansion=2,
        )


def test_mirepnet_maps_released_checkpoint_keys():
    model = _model()
    source_projection_weight = torch.full_like(
        model.embedding.projection.weight,
        0.125,
    )
    source_projection_bias = torch.full_like(
        model.embedding.projection.bias,
        -0.25,
    )
    source_head_weight = torch.full_like(model.final_layer.weight, 0.375)
    source_head_bias = torch.full_like(model.final_layer.bias, -0.5)

    incompatible = model.load_state_dict(
        {
            "embedding.projection.0.weight": source_projection_weight,
            "embedding.projection.0.bias": source_projection_bias,
            "clshead.weight": source_head_weight,
            "clshead.bias": source_head_bias,
        },
        strict=False,
    )

    torch.testing.assert_close(
        model.embedding.projection.weight,
        source_projection_weight,
    )
    torch.testing.assert_close(
        model.embedding.projection.bias,
        source_projection_bias,
    )
    torch.testing.assert_close(model.final_layer.weight, source_head_weight)
    torch.testing.assert_close(model.final_layer.bias, source_head_bias)
    assert set(incompatible.unexpected_keys) == set()

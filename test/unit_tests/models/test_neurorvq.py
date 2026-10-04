# Authors: Braindecode contributors
#
# License: BSD (3-clause)
import pytest
import torch

from braindecode.models import NeuroRVQ
from braindecode.models.neurorvq import NEURORVQ_CHANNELS, _Block


@pytest.fixture
def model_kwargs():
    return {
        "n_chans": 3,
        "n_outputs": 4,
        "n_times": 600,
        "sfreq": 200,
        "channel_names": ("f3", "f4", "cz"),
        "depth": 2,
        "num_heads": 4,
        "out_chans": 4,
        "max_patches": 8,
    }


def test_neurorvq_output_and_features(model_kwargs):
    model = NeuroRVQ(**model_kwargs)
    x = torch.randn(2, model.n_chans, model.n_times)

    logits = model(x)
    features = model(x, return_features=True)

    assert logits.shape == (2, 4)
    assert features["features"].shape == (2, 100 * 4 * 3 * 3)
    assert features["cls_token"] is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_times": 601}, "n_times must be divisible by patch_size"),
        ({"sfreq": 250}, "trained at 200 Hz"),
        (
            {"channel_names": ("f3", "f4", "not-an-eeg-channel")},
            "Unsupported NeuroRVQ channel",
        ),
        ({"channel_names": ("f3", "f3", "cz")}, "channel_names must be unique"),
        ({"n_times": 1800, "max_patches": 8}, "supports at most 8 patches"),
        ({"patch_size": 100}, "requires patch_size=200"),
        ({"init_values": None}, "init_values must be a number"),
    ],
)
def test_neurorvq_invalid_configuration(model_kwargs, kwargs, message):
    with pytest.raises(ValueError, match=message):
        NeuroRVQ(**(model_kwargs | kwargs))


def test_neurorvq_reset_head(model_kwargs):
    model = NeuroRVQ(**model_kwargs)
    model.reset_head(2)

    assert model(torch.randn(1, 3, 600)).shape == (1, 2)


def test_neurorvq_channel_slots_from_chs_info(model_kwargs):
    kwargs = model_kwargs | {
        "channel_names": None,
        "chs_info": [
            {"ch_name": "F3"},
            {"ch_name": "F4"},
            {"ch_name": "Cz"},
        ],
    }
    model = NeuroRVQ(**kwargs)

    assert model.channel_names == ("f3", "f4", "cz")
    assert model.spatial_embedding_ix.tolist() == [41, 42, 36]


def test_neurorvq_default_channel_names_follow_reference_order(model_kwargs):
    kwargs = model_kwargs | {"channel_names": None}
    model = NeuroRVQ(**kwargs)

    assert model.channel_names == NEURORVQ_CHANNELS[:3]


def test_neurorvq_transformer_block_uses_sequential_residuals():
    block = _Block(
        dim=16,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        qk_norm=lambda dim: torch.nn.LayerNorm(dim, eps=1e-6),
        drop=0,
        attn_drop=0,
        drop_path=0,
        init_values=1e-5,
    ).eval()
    x = torch.randn(2, 5, 16)

    expected = x + block.gamma_1 * block.attn(block.norm1(x))
    expected = expected + block.gamma_2 * block.mlp(block.norm2(expected))

    torch.testing.assert_close(block(x), expected)

import pytest
import torch
from torch import nn

from braindecode.models import NeurIPT, amplitude_aware_mask
from braindecode.models.neuript import _TSAStage


def _small_model(**kwargs):
    parameters = {
        "n_outputs": 3,
        "n_chans": 4,
        "n_times": 17,
        "sfreq": 128.0,
        "d_model": 24,
        "n_heads": 4,
        "n_layers": 2,
        "merge_factors": (1, 2),
        "n_experts": (0, 2),
        "expert_hidden_dim": 16,
        "channel_positions": torch.randn(4, 3),
        "lobe_groups": ((0, 1), (2, 3)),
    }
    parameters.update(kwargs)
    return NeurIPT(**parameters)


def test_neuript_forward_backward_and_reset_head():
    model = _small_model()
    x = torch.randn(2, 4, 17, requires_grad=True)

    result = model(x, return_features=True)

    assert result["logits"].shape == (2, 3)
    assert result["features"].shape == (2, 2 * 2 * 24)
    assert result["aux_loss"].ndim == 0
    (result["logits"].sum() + result["aux_loss"]).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.layers[1].time_moe.router.weight.grad is not None

    model.reset_head(5)
    assert model(torch.randn(2, 4, 17)).shape == (2, 5)


def test_amplitude_aware_mask_selects_expected_count_and_is_reproducible():
    x = torch.arange(40, dtype=torch.float32).reshape(2, 2, 10)
    first = amplitude_aware_mask(
        x, mask_ratio=0.3, generator=torch.Generator().manual_seed(7)
    )
    second = amplitude_aware_mask(
        x, mask_ratio=0.3, generator=torch.Generator().manual_seed(7)
    )

    assert first.dtype == torch.bool
    assert first.shape == x.shape
    assert torch.equal(first, second)
    assert torch.all(first.sum(dim=-1) == 3)


def test_amplitude_aware_mask_samples_percentile_centers():
    x = torch.arange(10, dtype=torch.float32).view(1, 1, 10).expand(1, 4, 10)
    seed = 23
    generator = torch.Generator().manual_seed(seed)
    actual = amplitude_aware_mask(x, mask_ratio=0.3, generator=generator)

    percentile_generator = torch.Generator().manual_seed(seed)
    centers = (torch.rand((1, 4, 1), generator=percentile_generator) * 10).floor()
    starts = (centers.long() - 1).clamp(0, 7)
    for channel, start in enumerate(starts.view(-1)):
        expected = torch.arange(10).ge(start) & torch.arange(10).lt(start + 3)
        assert torch.equal(actual[0, channel], expected)


@pytest.mark.parametrize("shape", [(10, 3), (2, 4, 5, 6)])
def test_amplitude_aware_mask_requires_bct_input(shape):
    with pytest.raises(ValueError, match="batch, channels, time"):
        amplitude_aware_mask(torch.randn(*shape))


def test_neuript_validates_channel_positions_and_lobe_groups():
    with pytest.raises(ValueError, match="channel_positions"):
        _small_model(channel_positions=torch.zeros(3, 3))
    with pytest.raises(ValueError, match="lobe_groups"):
        _small_model(lobe_groups=((0, 1), (1, 2)))


def test_neuript_embeds_each_sample_without_temporal_pooling():
    model = _small_model()
    embedded = model.embed(torch.randn(2, 4, 31))

    assert embedded.shape == (2, 31, 4, 24)


def test_neuript_requires_unmerged_first_layer():
    with pytest.raises(ValueError, match="first TSA layer"):
        _small_model(merge_factors=(2, 1))


def test_neuript_drop_prob_alias_overrides_dropout():
    model = _small_model(dropout=0.1, drop_prob=0.3)

    assert model.layers[0].time_attn.dropout == 0.3
    assert model.layers[0].time_moe.shared.dropout2.p == 0.3

    with pytest.raises(ValueError, match="dropout must be in "):
        _small_model(drop_prob=1.1)


def test_tsa_attention_residuals_bypass_layer_norm():
    class ZeroAttention(nn.Module):
        def forward(self, query, key, value, need_weights=True):
            return torch.zeros_like(query), None

    class ZeroMoE(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x), x.new_zeros(())

    stage = _TSAStage(
        dim=4,
        n_heads=2,
        expert_hidden_dim=8,
        n_experts=0,
        top_k_fraction=1.0,
        dropout=0.0,
    )
    stage.time_norm = nn.Identity()
    stage.time_attn = ZeroAttention()
    stage.time_post_norm = nn.Identity()
    stage.time_moe = ZeroMoE()
    stage.channel_norm = nn.Identity()
    stage.channel_attn = ZeroAttention()
    stage.channel_post_norm = nn.Identity()
    stage.channel_moe = ZeroMoE()

    x = torch.randn(2, 3, 2, 4)
    time_position = torch.full((3, 4), 2.0)
    spatial_position = torch.full((2, 4), 3.0)
    actual, _ = stage(x, time_position, spatial_position)

    torch.testing.assert_close(actual, x)

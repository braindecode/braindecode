import pytest
import torch

from braindecode.models import NeurIPT, amplitude_aware_mask


def _small_model(**kwargs):
    parameters = {
        "n_outputs": 3,
        "n_chans": 4,
        "n_times": 17,
        "sfreq": 128.0,
        "d_model": 24,
        "n_heads": 4,
        "n_layers": 2,
        "merge_factors": (2, 2),
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


@pytest.mark.parametrize("shape", [(10, 3), (2, 4, 5, 6)])
def test_amplitude_aware_mask_requires_bct_input(shape):
    with pytest.raises(ValueError, match="batch, channels, time"):
        amplitude_aware_mask(torch.randn(*shape))


def test_neuript_validates_channel_positions_and_lobe_groups():
    with pytest.raises(ValueError, match="channel_positions"):
        _small_model(channel_positions=torch.zeros(3, 3))
    with pytest.raises(ValueError, match="lobe_groups"):
        _small_model(lobe_groups=((0, 1), (1, 2)))


def test_neuript_bounds_long_windows_to_max_tokens():
    model = _small_model(max_tokens=8)
    embedded = model._embed(torch.randn(2, 4, 17))

    assert embedded.shape == (2, 8, 4, 24)

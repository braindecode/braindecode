"""JungMAE: the validity rules, the re-hosted weights and the pair counts."""

import pytest
import torch

from braindecode.models import JungMAE


def test_jung_mae_padded_samples_are_missing():
    """``on_non_divisible="pad"`` equals the NaN-padded window: the last patch
    holds 5 of 10 samples, under the 0.8 threshold."""
    torch.manual_seed(0)
    padded = JungMAE(
        n_chans=4, n_outputs=2, n_times=55, depth=1, on_non_divisible="pad"
    ).eval()
    plain = JungMAE(n_chans=4, n_outputs=2, n_times=60, depth=1).eval()
    plain.load_state_dict(padded.state_dict(), strict=True)
    x = torch.randn(2, 4, 55)
    x_nan = torch.cat([x, torch.full((2, 4, 5), float("nan"))], dim=-1)
    with torch.no_grad():
        out = padded.encode(x)
        assert not out["valid"][..., -1].any()
        torch.testing.assert_close(padded(x), plain(x_nan), rtol=0, atol=0)


def test_jung_mae_invalid_patch_does_not_reach_the_valid_tokens():
    """The finite samples of an invalid patch (5 of its 10, under the 0.8
    threshold) are not attended: changing them changes no token, and the token
    of the invalid patch is zero."""
    torch.manual_seed(0)
    model = JungMAE(
        n_chans=4, n_outputs=2, n_times=60, embed_dim=32, depth=2, num_heads=4
    ).eval()
    x = torch.randn(2, 4, 60)
    x[:, :, 20:25] = float("nan")  # patch 2 of every pair
    other = x.clone()
    other[:, :, 25:30] = torch.randn(2, 4, 5)  # the 5 finite samples of that patch
    with torch.no_grad():
        out, out_other = model.encode(x), model.encode(other)
    assert not out["valid"][..., 2].any()
    assert out["valid"][..., [0, 1, 3, 4, 5]].all()
    assert not out["foundation"][:, :, 2].any()
    assert torch.equal(out["foundation"], out_other["foundation"])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"minimum_patch_valid_fraction": 0.0}, "minimum_patch_valid_fraction"),
        ({"minimum_patch_valid_fraction": -0.1}, "minimum_patch_valid_fraction"),
        ({"minimum_patch_valid_fraction": 1.5}, "minimum_patch_valid_fraction"),
        ({"patch_size": 0}, "patch_size must be positive, got 0"),
    ],
)
def test_jung_mae_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JungMAE(**{"n_chans": 4, "n_outputs": 2, "n_times": 60, **kwargs})


@pytest.mark.network
@pytest.mark.huggingface
def test_jung_mae_pretrained_matches_the_release():
    """The re-hosted encoder gives the features of ``jung_mae.pt`` (GeoNIRS@81f8995)."""
    pytest.importorskip("huggingface_hub")
    model = JungMAE.from_pretrained(
        "braindecode/jungmae-pretrained",
        revision="812892cae3be16620bd1fb8a0d91b7b1c61ce0ca",
        n_chans=40,
        n_outputs=3,
        n_times=60,
    ).eval()
    x = torch.arange(40 * 60.0).reshape(1, 40, 60).mul(0.05).sin()
    with torch.no_grad():
        features = model(x, return_features=True)["features"]
    assert features.shape == (1, 20 * 6 * 192)  # pairs * patches * embed_dim
    expected = torch.tensor([-0.138848, 0.800462, 0.378428, 0.906246, -1.811891])
    torch.testing.assert_close(features[0, :5], expected, rtol=0, atol=1e-5)


def test_jung_mae_encoder_accepts_other_complete_pairs():
    model = JungMAE(
        n_chans=4, n_outputs=2, n_times=30, embed_dim=16, num_heads=2, depth=1
    ).eval()
    for n_chans in (2, 6):
        x = torch.randn(2, n_chans, 30)
        encoded = model.encode(x)["foundation"]
        torch.testing.assert_close(
            model(x, return_features=True)["features"], encoded.flatten(1)
        )
        with pytest.raises(ValueError, match="channels"):
            model(x)
    with pytest.raises(ValueError, match="pairs"):
        model.encode(torch.randn(2, 1, 30))

"""GeoNIRS validity masks and mean_std read-out, without the checkpoint: the
missing-patch, threshold and population-std paths of the reference."""

import pytest
import torch

from braindecode.models import GeoNIRS
from braindecode.models.util import _fnirs_chs_info

SMALL = dict(chs_info=_fnirs_chs_info(3), n_outputs=2, embed_dim=32, num_heads=4,
             depth=1, n_latent_channels=4, feature_dim=8)
NAN = float("nan")


def test_geonirs_missing_patch_is_a_shorter_window():
    """A patch with no measured sample is left out of the attention: the
    other patches equal those of the window without it, and its tokens are 0."""
    torch.manual_seed(0)
    long = GeoNIRS(n_times=60, **SMALL).eval()
    short = GeoNIRS(n_times=50, **SMALL).eval()
    backbone = {k: v for k, v in long.state_dict().items() if not k.startswith("final_layer.")}
    short.load_state_dict(backbone)
    x = torch.randn(2, 6, 60)
    x[..., 50:] = NAN
    with torch.no_grad():
        full = long.encode(x)
        cut = short.encode(x[..., :50])
    torch.testing.assert_close(full["foundation"][..., :5, :], cut["foundation"])
    assert not full["foundation"][..., 5, :].any()
    assert not full["valid"][..., 5].any()


@pytest.mark.parametrize("fraction, measured, valid", [
    (0.8, 8, True), (0.8, 7, False), (0.75, 8, True), (0.75, 7, False),
])
def test_geonirs_patch_validity_threshold(fraction, measured, valid):
    """"at least 80% of its time points" (B.1.2): the boundary count is valid."""
    model = GeoNIRS(n_times=20, minimum_patch_valid_fraction=fraction, **SMALL).eval()
    x = torch.randn(1, 6, 20)
    x[..., 10 + measured:] = NAN
    with torch.no_grad():
        got = model.encode(x)["valid"][0, 0].tolist()
    assert got == [True, valid]


def test_geonirs_mean_std_is_the_population_std_over_valid_patches():
    model = GeoNIRS(n_times=60, pool="mean_std", **SMALL).eval()
    x = torch.randn(2, 6, 60)
    x[1, :, 40:] = NAN  # window 1: 4 valid patches of 6
    with torch.no_grad():
        features = model(x, return_features=True)["features"].reshape(2, 4, 2, 16)
        foundation = model.encode(x)["foundation"]
    joint = torch.cat([foundation[:, 0], foundation[:, 1]], dim=-1)  # (2, M, P, 2F)
    for b, n_valid in [(0, 6), (1, 4)]:
        z = joint[b, :, :n_valid]
        torch.testing.assert_close(features[b, :, 0], z.mean(dim=1))
        torch.testing.assert_close(features[b, :, 1], z.std(dim=1, unbiased=False))

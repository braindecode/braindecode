"""GeoNIRS: the validity of patches, the mean_std read-out and the released
weights."""

import pytest
import torch

from braindecode.models import GeoNIRS
from braindecode.models.util import _fnirs_chs_info

SMALL = dict(chs_info=_fnirs_chs_info(2), n_outputs=2, embed_dim=32, num_heads=4,
             depth=1, n_latent_channels=4, feature_dim=8, projection_dims=(32, 16, 8))
NAN = float("nan")


def test_geonirs_missing_samples_and_padding():
    """A patch needs 80 % of its samples measured, padding counting as
    missing. An invalid patch has zero tokens and leaves the other patches as
    in the window without it."""
    torch.manual_seed(0)
    padded = GeoNIRS(n_times=55, on_non_divisible="pad", **SMALL).eval()
    short = GeoNIRS(n_times=40, **SMALL).eval()
    encoder = {k: v for k, v in padded.state_dict().items()
               if not k.startswith("final_layer.")}
    short.load_state_dict(encoder, strict=False)  # the heads differ in width
    x = torch.randn(2, 4, 55)
    x[..., 30:32] = NAN  # patch 3: 8 of 10 samples, valid
    x[..., 40:43] = NAN  # patch 4: 7 of 10, invalid; patch 5: 5 and padding
    with torch.no_grad():
        full, cut = padded.encode(x), short.encode(x[..., :40])
    assert full["valid"][..., :4].all() and not full["valid"][..., 4:].any()
    assert not full["foundation"][..., 4:, :].any()
    torch.testing.assert_close(full["foundation"][..., :4, :], cut["foundation"])


def test_geonirs_mean_std_over_valid_patches():
    """``pool="mean_std"``: the mean and population std over the valid
    patches, zero without one, and finite gradients with a single one."""
    torch.manual_seed(0)
    model = GeoNIRS(n_times=60, pool="mean_std", **SMALL).eval()
    x = torch.randn(4, 4, 60)
    x[1, :, 40:] = NAN  # 4 valid patches of 6
    x[2, :, 10:] = NAN  # 1 valid patch
    x[3] = NAN  # none
    features = model(x, return_features=True)["features"]
    features.sum().backward()
    assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        foundation = model.encode(x)["foundation"]
    joint = torch.cat([foundation[:, 0], foundation[:, 1]], dim=-1)  # (4, M, P, 2F)
    features = features.detach().reshape(4, 4, 2, 16)  # (4, M, [mean, std], 2F)
    for b, n_valid in [(0, 6), (1, 4), (2, 1)]:
        z = joint[b, :, :n_valid]
        torch.testing.assert_close(features[b, :, 0], z.mean(dim=1))
        torch.testing.assert_close(features[b, :, 1], z.std(dim=1, unbiased=False))
    assert not features[3].any()


# features[0, :5] (HbO tokens) and features[0, -5:] (HbR tokens) of the authors'
# geonirs_base.pt and geonirs_large.pt
# (GeoNIRS@81f8995) on the input of the test, loaded as released
@pytest.mark.network
@pytest.mark.huggingface
@pytest.mark.parametrize(
    "repo_id, revision, expected, expected_hbr",
    [
        ("braindecode/geonirs-base-pretrained", "a483f904180dce2c0fdc6dd48917b95498913d9a",
         [0.067987166, -0.007574921, -0.350337952, 0.147298455, -0.141785607],
         [-0.119261660, 0.040913288, -0.013012240, -0.140069067, 0.107298315]),
        ("braindecode/geonirs-large-pretrained", "f51061d599032f726ba6732083bce2dbc7c3fcd1",
         [-3.116364002, -2.744976997, -0.223239779, -1.097928047, -0.808620572],
         [2.245513439, -1.460690856, 0.737670124, 2.605016947, -0.031923484]),
    ],
    ids=["base", "large"],
)
def test_geonirs_pretrained_matches_the_release(repo_id, revision, expected, expected_hbr):
    pytest.importorskip("huggingface_hub")
    model = GeoNIRS.from_pretrained(
        repo_id, revision=revision, chs_info=_fnirs_chs_info(4), n_outputs=3, n_times=150
    ).eval()
    x = torch.arange(8 * 150.0).reshape(1, 8, 150).mul(0.05).sin()
    with torch.no_grad():
        features = model(x, return_features=True)["features"]
    torch.testing.assert_close(features[0, :5], torch.tensor(expected), rtol=0, atol=1e-5)
    # the last features are HbR tokens, the only ones hbr_projection writes
    torch.testing.assert_close(features[0, -5:], torch.tensor(expected_hbr), rtol=0, atol=1e-5)

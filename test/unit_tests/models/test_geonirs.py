"""GeoNIRS without the checkpoint: the validity masks and mean_std read-out,
and a seeded small model and the chs_info geometry against values computed
with the authors' code (GeoNIRS@81f8995, ``Ours`` with a strict load and
``scripts/build_fair_input.py``)."""

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
    backbone = {
        k: v for k, v in long.state_dict().items() if not k.startswith("final_layer.")
    }
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


def test_geonirs_mean_std_step_with_zero_one_or_several_valid_patches():
    model = GeoNIRS(n_times=20, pool="mean_std", **SMALL).train()
    x = torch.randn(3, 6, 20)
    x[0] = NAN
    x[1, :, 10:] = NAN
    x.requires_grad_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0]))
    assert loss.isfinite()
    loss.backward()
    assert x.grad.isfinite().all()
    assert all(
        p.grad.isfinite().all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert all(p.isfinite().all() for p in model.parameters())


# The authors' Ours writes its projection widths, 192 -> 128 -> 96 -> 64, by
# hand, so the small model keeps D = 192 and F = 64 and is small elsewhere
REFERENCE_SIZE = dict(
    embed_dim=192, num_heads=4, depth=1, n_latent_channels=4, feature_dim=64
)

# encode(x)["foundation"].flatten(1)[:, ::203] of the authors' Ours on
# _window(): 8 HbO and 8 HbR values, over the 4 latent channels and the 6
# patches (patch 4 of window 1 has no valid channel, so its values are 0)
EXPECTED_FOUNDATION = [
    [
        0.6963703036308289, 0.5677199959754944, -1.0788012742996216,
        -0.04356303811073303, 0.15072523057460785, -0.749310314655304,
        0.5727516412734985, 2.4500386714935303, -1.132354497909546,
        -0.4393085241317749, -1.230041742324829, 1.4032108783721924,
        -1.348412036895752, 0.4464509189128876, -0.4862309992313385,
        0.3426509499549866,
    ],
    [
        0.6604654788970947, 0.5148303508758545, -1.1532312631607056,
        -0.0242752805352211, 0.07708635181188583, -0.7201430201530457,
        0.6330594420433044, 0.0, -1.1344952583312988,
        0.0, -1.2298455238342285, 0.0,
        -1.3693218231201172, 0.4532400369644165, -0.4585363566875458,
        0.3352893590927124,
    ],
]

# The authors' dataset_geometry for the optodes of _fnirs_chs_info(2):
# [midpoint, detector - source] of each pair, centred, in units of 25.46 mm
EXPECTED_GEOMETRY = [
    [
        0.7222108244895935, -1.3219980001449585, 0.0,
        1.178291916847229, 0.0, 0.0,
    ],
    [
        -0.7222108244895935, 1.3219980001449585, 0.0,
        1.178291916847229, 0.0, 0.0,
    ],
]


def _seeded_small_geonirs():
    """GeoNIRS of depth 1 on 3 pairs and 12 s at 5 Hz (6 patches), whose
    tensors are drawn from one seeded generator in the order of the sorted
    state-dict keys, so that they do not depend on the order the layers are
    built in."""
    model = GeoNIRS(
        chs_info=_fnirs_chs_info(3),
        n_outputs=2,
        n_times=60,
        sfreq=5.0,
        **REFERENCE_SIZE,
    ).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _, tensor in sorted(model.state_dict().items()):
            # 1 / sqrt(fan-in) for the matrices, 0.5 for the vectors
            std = tensor.shape[-1] ** -0.5 if tensor.dim() > 1 else 0.5
            tensor.copy_(std * torch.randn(tensor.shape, generator=generator))
    return model


def _window():
    """Two windows with missing samples. Window 0: HbO of pair 1 misses 7 of
    the 10 samples of patch 0 (that patch of the pair is invalid), HbR of pair
    2 one sample. Window 1: pair 0 has no data, and patch 4 has no valid
    channel."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 6, 60, generator=generator)
    x[0, 1, 2:9] = NAN
    x[0, 5, 33] = NAN
    x[1, [0, 3]] = NAN
    x[1, :, 40:50] = NAN
    return x


def test_geonirs_small_model_matches_the_authors_code():
    """The latent tokens equal those of the authors' ``Ours``
    (``model/ours.py``, GeoNIRS@81f8995) loaded strictly with the same state
    dict and fed the authors' input: zero-filled ``hbo`` and ``hbr``, the
    sample and channel validity masks, and the geometry of
    ``scripts/build_fair_input.py`` (GeoNIRS@81f8995)."""
    model = _seeded_small_geonirs()
    with torch.no_grad():
        encoded = model.encode(_window())
    got = encoded["foundation"].flatten(1)[:, ::203]
    expected = torch.tensor(EXPECTED_FOUNDATION)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    assert not encoded["valid"][1, :, 4].any()
    assert encoded["valid"][1, :, [0, 1, 2, 3, 5]].all()


def test_geonirs_default_geometry_is_the_authors_geometry():
    """``default_geometry`` of ``_fnirs_chs_info(2)`` equals, exactly, what
    ``dataset_geometry`` of the authors' ``scripts/build_fair_input.py``
    (GeoNIRS@81f8995) gives for the same optodes: centred on the mean optode,
    divided by 25.46 mm, ``[midpoint, detector - source]``."""
    model = GeoNIRS(chs_info=_fnirs_chs_info(2), n_outputs=2, n_times=150)
    expected = torch.tensor(EXPECTED_GEOMETRY)
    torch.testing.assert_close(model.default_geometry, expected, rtol=0, atol=0)

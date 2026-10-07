# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel layer: regressions A-G and the invariants every strategy keeps."""

import numpy as np
import pytest
import torch

from braindecode.models import LUNA
from braindecode.modules import ChannelLayer
from braindecode.modules.channels import STRATEGIES, _resolve

TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()
SUBSET = ["Fp1", "F3", "Fz", "C3", "Cz", "P4", "Pz", "O2"]


def chs(names, kind="eeg"):
    return [{"ch_name": n, "kind": kind, "loc": np.zeros(12)} for n in names]


def at(name, xyz):
    return {"ch_name": name, "kind": "eeg", "loc": np.r_[xyz, np.zeros(9)]}


def test_a_legacy_alias_does_not_merge_distinct_names():
    assert ChannelLayer(
        chs(["T7", "T3"]), "exact", chs(["T3", "T7"])
    ).weight.tolist() == [
        [0, 1],
        [1, 0],
    ]


def test_b_misspelt_strategy_is_declared():
    with pytest.raises(ValueError, match=r"Did you mean \['spline'\]"):
        ChannelLayer(chs(TEN_TWENTY), "splien", chs(SUBSET))


def test_c_coordinate_only_channel_matched_by_position():
    cz = _resolve(chs(["Cz"]))[1][0]
    layer = ChannelLayer(chs(["Cz"]), "exact", [at("E7", cz + [0.002, 0, 0])])
    assert layer.weight.tolist() == [[1.0]]
    with pytest.raises(ValueError, match="not in the input"):  # never relabelled
        ChannelLayer(chs(["Cz"]), "exact", [at("FCz", cz + [0.002, 0, 0])])


def test_d_non_eeg_rejected_or_dropped():
    src = chs(SUBSET) + chs(["EOG1"], kind="eog")
    with pytest.raises(ValueError, match="not EEG"):
        ChannelLayer(chs(TEN_TWENTY), "spline", src)
    layer = ChannelLayer(chs(SUBSET), "exact", src, drop_non_eeg=True)
    x = torch.randn(2, len(src), 10)
    assert torch.equal(layer(x)[0], x[:, :-1])


def test_e_legacy_names_are_copies():
    layer = ChannelLayer(
        chs(["T7", "T8", "P7", "P8"]), "spline", chs(["T3", "T4", "T5", "T6"])
    )
    assert torch.equal(layer.weight, torch.eye(4))


def test_f_sparse_montage_gain_and_targets_without_position():
    src = chs(["Fz", "Cz", "Pz", "Oz"])
    assert ChannelLayer(chs(TEN_TWENTY), "spline", src).weight.abs().sum(1).max() < 5
    with pytest.warns(UserWarning, match="row gain"):
        ChannelLayer(chs(TEN_TWENTY), "spline", src, reg=0.0)
    x, observed = ChannelLayer(chs([*TEN_TWENTY, "SCALE"]), "spline", src)(
        torch.randn(1, 4, 5)
    )
    assert not observed[-1] and (x[0, -1] == 0).all() and observed.sum() == 3


@pytest.mark.parametrize("strategy", ["spline", "field", "source"])
def test_g_fewer_than_four_positions_is_declared(strategy):
    with pytest.raises(ValueError, match="needs at least 4 channels with a position"):
        ChannelLayer(chs(TEN_TWENTY), strategy, chs(["Fz", "Cz", "Pz"]))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_permutation_is_an_exact_copy(strategy):
    x = torch.randn(2, 19, 10)
    layer = ChannelLayer(chs(TEN_TWENTY), strategy, chs(TEN_TWENTY[::-1]))
    assert torch.equal(layer(x)[0], x.flip(1))


def test_bipolar_target_is_a_derivation():
    layer = ChannelLayer(chs(["F3-C3"]), "spline", chs(SUBSET))
    x = torch.randn(1, 8, 5)
    assert torch.equal(layer(x)[0][0, 0], x[0, 1] - x[0, 3])


@pytest.mark.parametrize(
    "strategy,kw", [("field", {}), ("source", {}), ("source", {"trainable": True})]
)
def test_constant_input_gives_constant_output(strategy, kw):
    # A target on the sphere's vertical axis (biosemi64 Cz) must stay finite.
    target = chs(TEN_TWENTY) + [at("Zaxis", [0, 0, 0.12])]
    out, _ = ChannelLayer(target, strategy, chs(SUBSET), **kw)(torch.ones(1, 8, 4))
    torch.testing.assert_close(out, torch.ones_like(out))


def test_trainable_source_equals_physics_at_init():
    x = torch.randn(2, 8, 50)
    physics = ChannelLayer(chs(TEN_TWENTY), "source", chs(SUBSET))(x)[0]
    trainable = ChannelLayer(chs(TEN_TWENTY), "source", chs(SUBSET), trainable=True)
    assert torch.equal(trainable(x)[0], physics)
    trainable(x)[0].sum().backward()
    assert trainable.parcel_mix.grad.abs().sum() > 0


def test_load_clears_the_cache_and_backbone_checkpoints_load():
    kw = dict(chs_info=chs(SUBSET), n_outputs=2, n_times=800, sfreq=200)
    backbone = LUNA(**kw).state_dict()
    model = LUNA(
        **kw, channel_strategy="source", channel_strategy_kwargs={"trainable": True}
    )
    assert model.channel_layer._key is not None
    model.load_state_dict(backbone, strict=True)  # the layer keeps its init
    assert model.channel_layer._key is None
    x = torch.randn(1, 8, 800)
    assert torch.isfinite(model(x.flip(1), chs_info=chs(SUBSET[::-1]))).all()
    assert model.get_config()["channel_strategy"] == "source"


_DEVICES = ["cpu"] + [
    d
    for d, ok in [
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    ]
    if ok
]


@pytest.mark.parametrize("device", _DEVICES)
def test_maps_follow_device_and_dtype(device):
    dtype = torch.float32 if device == "mps" else torch.float64
    layer = ChannelLayer(chs(TEN_TWENTY), "source", chs(SUBSET), trainable=True)
    x = torch.randn(2, 8, 20)
    expected = layer(x)[0]
    out, observed = layer.to(device, dtype)(x.to(device, dtype))
    assert out.device.type == observed.device.type == device and out.dtype == dtype
    torch.testing.assert_close(out.cpu().float(), expected, rtol=1e-4, atol=1e-4)


def smooth_field(n, rng):
    """``(n, 19)`` random quadratic fields sampled on the 10-20 positions."""
    P = _resolve(chs(TEN_TWENTY))[1] / 0.1
    feats = np.c_[np.ones(19), P, P**2, P[:, :1] * P[:, 1:2], P[:, 1:2] * P[:, 2:]]
    return rng.standard_normal((n, feats.shape[1])) @ feats.T


def test_wiener_beats_zero_fill_and_survives_a_state_dict_round_trip():
    rng = np.random.default_rng(0)
    X, truth = smooth_field(2000, rng), torch.tensor(smooth_field(50, rng).T).float()
    idx = [TEN_TWENTY.index(n) for n in SUBSET]
    layer = ChannelLayer(chs(TEN_TWENTY), "wiener", chs(SUBSET))
    with pytest.raises(ValueError, match=r"call fit\(\)"):
        layer(truth[idx][None])
    with pytest.warns(UserWarning, match="row gain"):  # 10-dim field, 8 inputs
        out = layer.fit(X, chs(TEN_TWENTY))(truth[idx][None])[0][0]
    zero = ChannelLayer(chs(TEN_TWENTY), "zero", chs(SUBSET))(truth[idx][None])[0][0]
    assert (out - truth).norm() < 0.3 * (zero - truth).norm()
    fresh = ChannelLayer(chs(TEN_TWENTY), "wiener", chs(SUBSET))
    fresh.load_state_dict(layer.state_dict())
    assert torch.equal(fresh(truth[idx][None])[0][0], out)


def test_region_reconstructs_a_dropped_electrode():
    src = [n for n in TEN_TWENTY if n != "Cz"]
    x, observed = ChannelLayer(chs(TEN_TWENTY), "region", chs(src))(
        torch.ones(1, 18, 4)
    )
    assert not observed[9] and torch.allclose(x[0, 9], torch.ones(4))


def test_latent_is_permutation_invariant_and_trains():
    layer = ChannelLayer(chs(TEN_TWENTY), "latent", chs(SUBSET))
    x = torch.randn(2, 8, 50)
    out = layer(x)[0]
    torch.testing.assert_close(layer(x.flip(1), chs(SUBSET[::-1]))[0], out)
    out.sum().backward()
    assert all(p.grad.abs().sum() > 0 for p in layer.parameters())

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel layer: regressions A-G and the invariants every strategy keeps."""

import numpy as np
import pytest
import torch

from braindecode.modules.channels import ChannelTarget, ChannelTokenizer
from braindecode.modules.channels.resolve import match_names, standard_position

TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()
SUBSET = ["Fp1", "F3", "Fz", "C3", "Cz", "P4", "Pz", "O2"]
LINEAR = ["exact", "zero", "nearest", "idw", "spline", "field", "source", "region"]


def chs(names, loc=True, kind="eeg"):
    return [
        {
            "ch_name": n,
            "kind": kind,
            "loc": np.r_[standard_position(n), np.zeros(9)] if loc else np.zeros(12),
        }
        for n in names
    ]


def layer(strategy, src, names=TEN_TWENTY, **kw):
    target = ChannelTarget("montage", chs_info=chs(names))
    return ChannelTokenizer(target, strategy, src, **kw)


def test_a_alias_does_not_merge_distinct_vocabulary_names():
    assert match_names(["T3", "T7"], ["T7", "T3"]).tolist() == [1, 0]
    assert match_names(["T3"], ["T7"]).tolist() == [0]


def test_b_misspelt_strategy_is_declared():
    with pytest.raises(ValueError, match=r"Did you mean \['spline'\]"):
        layer("splien", chs(SUBSET))


def test_c_coordinate_only_channel_matched_by_position():
    near_cz = standard_position("Cz") + [0.002, 0, 0]
    e7 = [{"ch_name": "E7", "kind": "eeg", "loc": np.r_[near_cz, np.zeros(9)]}]
    assert layer("exact", e7, ["Cz"])(torch.ones(1, 1, 1)).weights.tolist() == [[1.0]]
    fcz = [{**e7[0], "ch_name": "FCz"}]  # a known name is never relabelled
    with pytest.raises(ValueError, match="not in the input"):
        layer("exact", fcz, ["Cz"])


@pytest.mark.parametrize("loc", [True, False])
def test_d_non_eeg_rejected_or_dropped(loc):
    src = chs(SUBSET) + chs(["EOG1"], loc=loc, kind="eog")
    with pytest.raises(ValueError, match="not EEG"):
        layer("spline", src)
    tok = layer("exact", src, SUBSET, drop_non_eeg=True)
    x = torch.randn(2, len(src), 10)
    assert torch.equal(tok(x).x, x[:, :-1])


@pytest.mark.parametrize("loc", [True, False])
def test_e_legacy_names_are_copies(loc):
    src = chs(["T3", "T4", "T5", "T6", "A1"], loc=loc)
    tok = layer("spline", src, ["T7", "T8", "P7", "P8", "M1"])
    assert torch.equal(tok(torch.eye(5)[None]).weights, torch.eye(5))


def test_f_sparse_montages_and_non_electrodes():
    src = chs(["Fz", "Cz", "Pz", "Oz"])
    gain = layer("spline", src)(torch.zeros(1, 4, 1)).weights.abs().sum(1).max()
    assert gain < 5
    with pytest.warns(UserWarning, match="row gain"):
        layer("spline", src, reg=0.0)
    target = ChannelTarget(
        "montage",
        chs_info=chs(TEN_TWENTY) + [{"ch_name": "SCALE", "loc": np.ones(12) * 0.01}],
        non_electrode=("SCALE",),
    )
    enc = ChannelTokenizer(target, "spline", src)(torch.randn(1, 4, 5))
    assert not enc.observed[-1] and (enc.x[0, -1] == 0).all()


@pytest.mark.parametrize("strategy", ["spline", "field", "source"])
def test_g_fewer_than_four_positions_is_declared(strategy):
    with pytest.raises(ValueError, match="needs at least 4 channels with a position"):
        layer(strategy, chs(["Fz", "Cz", "Pz"]))


@pytest.mark.parametrize("strategy", [*LINEAR, "latent"])
def test_permutation_is_an_exact_copy(strategy):
    perm = TEN_TWENTY[::-1]
    x = torch.randn(2, 19, 10)
    assert torch.equal(layer(strategy, chs(perm))(x).x, x.flip(1))


@pytest.mark.parametrize(
    "strategy,kw", [("field", {}), ("source", {}), ("source", {"trainable": True})]
)
def test_constant_input_gives_constant_output(strategy, kw):
    # A target on the sphere's vertical axis (biosemi64 Cz) must stay finite.
    axis = {"ch_name": "Zaxis", "kind": "eeg", "loc": np.r_[0, 0, 0.12, np.zeros(9)]}
    target = ChannelTarget("montage", chs_info=chs(TEN_TWENTY) + [axis])
    out = ChannelTokenizer(target, strategy, chs(SUBSET), **kw)(torch.ones(1, 8, 4)).x
    torch.testing.assert_close(out, torch.ones_like(out))


def test_trainable_source_equals_physics_at_init():
    x = torch.randn(2, 8, 50)
    physics = layer("source", chs(SUBSET))(x).x
    assert torch.equal(layer("source", chs(SUBSET), trainable=True)(x).x, physics)


def test_load_state_dict_clears_the_map_cache():
    dense = chs(TEN_TWENTY)
    rng = np.random.default_rng(0)
    a, b = (layer("wiener", chs(SUBSET)) for _ in range(2))
    a.fit(rng.standard_normal((200, 19)), dense)
    b.fit(rng.standard_normal((200, 19)) @ rng.standard_normal((19, 19)), dense)
    x = torch.randn(1, 8, 5)
    a(x)  # caches a's map
    a.load_state_dict(b.state_dict())
    torch.testing.assert_close(a(x).x, b(x).x)


def test_unfitted_wiener_is_declared():
    with pytest.raises(ValueError, match="call fit"):
        layer("wiener", chs(SUBSET))(torch.randn(1, 8, 5))


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
    tok = layer("source", chs(SUBSET), trainable=True)
    x = torch.randn(2, 8, 20)
    expected = tok(x).x
    enc = tok.to(device, dtype)(x.to(device, dtype), chs(SUBSET[::-1]))
    assert enc.x.device.type == device and enc.weights.dtype == dtype
    torch.testing.assert_close(
        tok(x.to(device, dtype)).x.cpu().float(), expected, rtol=1e-4, atol=1e-4
    )

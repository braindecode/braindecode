# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import math

import numpy as np
import pytest
import torch

from braindecode.models.bendr import _BENDR_TARGET_CHS_INFO
from braindecode.modules.channels import (
    ChannelTarget,
    get_channel_strategy,
    resolve_montage,
)

BENDR19 = [c for c in _BENDR_TARGET_CHS_INFO if c["ch_name"] != "SCALE"]
P19 = np.stack([np.asarray(c["loc"], float)[:3] for c in BENDR19])
MONTAGE19 = ChannelTarget("montage", chs_info=BENDR19)
SENSOR = ["exact", "zero", "nearest", "idw", "spline", "field"]


def _named(names):
    """Names-only channels (positions come from standard_1005)."""
    return [{"ch_name": n, "kind": "eeg"} for n in names]


def _build(strategy, src_chs, target=MONTAGE19, **kw):
    return get_channel_strategy(strategy, **kw).build(resolve_montage(src_chs), target)


@pytest.mark.parametrize("strategy", SENSOR)
def test_permutation_is_exact_one_hot(strategy):
    m = _build(strategy, BENDR19[::-1])
    np.testing.assert_array_equal(m.weights.numpy(), np.eye(19)[::-1])
    assert m.observed.all() and (m.support == 1).all()


def test_spline_gain_bounded():
    # Today (reg=0) the spline maps [Fz, Cz, Pz, Oz] to BENDR's 19 with a row
    # gain of ~1300: noise is amplified a thousandfold.
    src = _named(["Fz", "Cz", "Pz", "Oz"])
    gain = _build("spline", src).weights.abs().sum(1).max()
    assert gain < 5
    assert _build("spline", src, reg=0.0).weights.abs().sum(1).max() > 100


@pytest.mark.parametrize("strategy", ["spline", "field"])
def test_less_than_4_positions_declared_error(strategy):
    with pytest.raises(ValueError, match="at least 4"):
        _build(strategy, _named(["Fz", "Cz", "Pz"]))


def test_non_electrode_target_never_interpolated():
    target = ChannelTarget(
        "montage", chs_info=_BENDR_TARGET_CHS_INFO, non_electrode=("SCALE",)
    )
    m = _build("spline", BENDR19[:8], target)
    assert m.weights.shape == (20, 8)
    assert (m.weights[-1] == 0).all()
    assert not m.observed[-1] and m.support[-1] == 0
    assert m.weights[10].abs().sum() > 0  # electrodes still interpolated


def test_policy_typo():
    with pytest.raises(ValueError, match="splin.*spline"):
        get_channel_strategy("splin")


def test_exact_missing_target_is_declared_error():
    with pytest.raises(ValueError, match="O1"):
        _build("exact", BENDR19[:17])


def test_zero_marks_missing_rows():
    m = _build("zero", BENDR19[:17])
    assert m.observed.tolist() == [True] * 17 + [False, False]
    assert (m.weights[17:] == 0).all() and (m.support[17:] == 0).all()


def test_support_decays_with_distance_to_nearest_source():
    # Four sources on a ring, one target 30 mm straight above the first.
    ring = [(0.08, 0, 0), (0, 0.08, 0), (-0.08, 0, 0), (0, -0.08, 0)]
    src = [
        {"ch_name": f"E{i}", "kind": "eeg", "loc": np.r_[p, np.zeros(9)]}
        for i, p in enumerate(ring)
    ]
    target = ChannelTarget(
        "montage",
        chs_info=[{"ch_name": "X", "loc": np.r_[0.08, 0, 0.03, np.zeros(9)]}],
    )
    m = _build("nearest", src, target)
    assert m.weights.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert m.support.item() == pytest.approx(math.exp(-1), rel=1e-5)
    assert not m.observed.item()


def test_coordinates_only_channel_takes_nearest_name():
    # An unknown name 5 mm from O1 is O1 for every strategy (copied, observed).
    loc = P19[17] + np.array([0.005, 0.0, 0.0])
    src = BENDR19[:17] + [
        {"ch_name": "E99", "kind": "eeg", "loc": np.r_[loc, np.zeros(9)]},
        BENDR19[18],
    ]
    m = _build("exact", src)
    assert m.weights[17].tolist() == [0.0] * 17 + [1.0, 0.0]
    assert m.observed.all()
    assert m.support[17].item() == pytest.approx(math.exp(-5 / 30), rel=1e-4)


def test_unusable_channel_warns_and_is_ignored():
    src = _named(["Fz", "Cz", "Pz", "Oz", "Xyz"])
    with pytest.warns(UserWarning, match="Xyz"):
        m = _build("spline", src)
    assert (m.weights[:, 4] == 0).all()


def test_ids_exact_maps_vocabulary_and_lists_closest_names():
    target = ChannelTarget("ids", vocabulary=("Fp1", "Fz", "Cz", "Pz"))
    m = _build("exact", _named(["Cz", "T3", "fz"]), ChannelTarget("ids", vocabulary=("Fp1", "Fz", "Cz", "T7")))
    assert m.channel_ids.tolist() == [2, 3, 1]
    assert m.weights is None and m.observed.all()
    with pytest.raises(ValueError, match="'Cy'.*Cz"):
        _build("exact", _named(["Cz", "Cy"]), target)


def test_ids_reconstructing_strategy_fills_vocabulary():
    target = ChannelTarget("ids", vocabulary=("Fz", "Cz", "Pz", "Oz", "Nope"))
    m = _build("idw", _named(["Cz", "Pz", "C3", "C4"]), target)
    assert m.channel_ids.tolist() == [0, 1, 2, 3, 4]
    assert m.observed.tolist() == [False, True, True, False, False]
    assert m.weights[0].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (m.weights[4] == 0).all() and m.support[4] == 0  # no position known


@pytest.mark.parametrize("strategy", SENSOR)
def test_free_target_is_pass_through(strategy):
    m = _build(strategy, _named(["Cz", "Pz", "E1"]), ChannelTarget("free"))
    assert m.weights is None and m.observed.all()
    x = torch.randn(2, 3, 10)
    assert get_channel_strategy(strategy).apply(x, m) is x


def test_slots_target_uses_first_n_slots():
    target = ChannelTarget("slots", chs_info=BENDR19, n_slots=4)
    m = _build("spline", BENDR19[2:10][::-1], target)
    assert m.weights.shape == (4, 8)
    assert m.observed.tolist() == [False, False, True, True]


# ---- fidelity (Fig 6 of the design report, smooth non-dipolar fields) -------


def _smooth_fields(rng, n):
    """Random spherical-harmonic fields (degree 1-4) on the 19 BENDR sites."""
    try:
        from scipy.special import sph_harm_y
    except ImportError:  # scipy < 1.15
        from scipy.special import sph_harm

        def sph_harm_y(n_, m_, theta, phi):
            return sph_harm(m_, n_, phi, theta)

    u = P19 - P19.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    th, ph = np.arccos(np.clip(u[:, 2], -1, 1)), np.arctan2(u[:, 1], u[:, 0])
    Y = np.stack(
        [
            np.real(sph_harm_y(deg, m, th, ph)) / deg
            for deg in range(1, 5)
            for m in range(-deg, deg + 1)
        ]
    )
    X = rng.normal(size=(n, len(Y))) @ Y
    X -= X.mean(1, keepdims=True)
    return X / X.std(axis=1, keepdims=True)


def fidelity(strategy, fields, k=8, draws=20, noise=0.1, seed=7, **kw):
    """Median RMSE on the missing electrodes / their SD (1.0 = zero fill)."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(draws):
        idx = np.sort(rng.choice(19, k, replace=False))
        W = _build(strategy, [BENDR19[i] for i in idx], **kw).weights.double().numpy()
        miss = np.setdiff1d(np.arange(19), idx)
        X = fields(rng, 100)
        Xo = X[:, idx] + noise * rng.normal(size=(100, k))
        err = np.sqrt((((Xo @ W.T)[:, miss] - X[:, miss]) ** 2).mean())
        errs.append(err / X[:, miss].std())
    return float(np.median(errs))


# Bound: zero fill (1.0) + 0.05. The regularised spline does not beat zero
# fill on these fields (report probe 1.16, here 1.07); its bound 1.15 still
# fails the unregularised spline (1.28 here).
@pytest.mark.parametrize(
    "strategy,bound", [("idw", 1.05), ("field", 1.05), ("spline", 1.15)]
)
def test_fidelity_smooth_fields_k8(strategy, bound):
    assert fidelity(strategy, _smooth_fields) <= bound

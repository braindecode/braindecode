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
PHYSICS = ["source"]


def _named(names):
    """Names-only channels (positions come from standard_1005)."""
    return [{"ch_name": n, "kind": "eeg"} for n in names]


def _build(strategy, src_chs, target=MONTAGE19, **kw):
    if isinstance(strategy, str):
        strategy = get_channel_strategy(strategy, **kw)
    return strategy.build(resolve_montage(src_chs), target)


@pytest.mark.parametrize("strategy", SENSOR + PHYSICS)
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


@pytest.mark.parametrize("strategy", ["spline", "field", "source"])
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


# ---- source strategy (direction C) ------------------------------------------


def _dipole_fields_factory():
    """3 random dipoles in a head that differs from the strategy's sphere.

    4-shell sphere, origin shifted 8 mm, radius 95 mm, skull conductivity
    halved, 10 mm grid (Fig 6 of the design report).
    """
    import mne

    sphere = mne.make_sphere_model(
        r0=(0.0, 0.008, 0.045),
        head_radius=0.095,
        relative_radii=(0.90, 0.92, 0.97, 1.0),
        sigmas=(0.33, 1.0, 0.002, 0.33),
        verbose=False,
    )
    src = mne.setup_volume_source_space(
        sphere=sphere, pos=10.0, mindist=5.0, exclude=20.0, verbose=False
    )
    names = [c["ch_name"] for c in BENDR19]
    info = mne.create_info(names, 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(dict(zip(names, P19)), coord_frame="head")
    )
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
    )
    L = fwd["sol"]["data"]
    L = L - L.mean(0, keepdims=True)

    def fields(rng, n):
        S = np.zeros((n, L.shape[1]))
        for i in range(n):
            S[i, rng.choice(L.shape[1], 3, replace=False)] = rng.normal(size=3)
        X = S @ L.T
        return X / X.std(axis=1, keepdims=True)

    return fields


def test_source_fidelity_dipoles_mismatched_head_k8():
    # 0.83 here; the report probe reached 0.71 by inverting data already
    # average-referenced over all 19 sites (an oracle reference). The strategy
    # re-references over the observed channels instead, as real data need.
    # Field mapping, the best head-model-free baseline, is 0.92.
    assert fidelity("source", _dipole_fields_factory()) <= 0.85


@pytest.mark.parametrize(
    "target", [MONTAGE19, ChannelTarget("free")], ids=["montage", "free"]
)
def test_source_init_equals_physics(target):
    src = resolve_montage(BENDR19[:8])
    physics = get_channel_strategy("source")
    learned = get_channel_strategy("source", trainable=True)
    x = torch.randn(3, 8, 50)
    out_p = physics.apply(x, physics.build(src, target))
    out_l = learned.apply(x, learned.build(src, target))
    assert (out_p - out_l).abs().max().item() == 0.0
    assert len(list(learned.parameters())) > 0 and not list(physics.parameters())


def test_source_gradient_reaches_attention_after_one_step():
    strategy = get_channel_strategy("source", trainable=True)
    m = strategy.build(resolve_montage(BENDR19[:8]), MONTAGE19)
    opt = torch.optim.SGD(strategy.parameters(), lr=0.1)
    x = torch.randn(4, 8, 30)
    grads = []
    for _ in range(2):
        opt.zero_grad()
        strategy.apply(x, m).pow(2).mean().backward()
        grads.append(strategy.queries.grad.abs().sum().item())
        opt.step()
    assert grads[0] == 0.0  # zero gate: the physics solution at init
    assert grads[1] > 0.0  # once the gate opens, attention learns


def test_source_free_target_has_fixed_size_for_any_montage():
    strategy = get_channel_strategy("source", n_parcels=32)
    sizes = set()
    for chs in (BENDR19[:5], BENDR19, _named(["C3", "C4", "Cz", "FC1", "CP2"])):
        m = strategy.build(resolve_montage(chs), ChannelTarget("free"))
        assert m.weights.shape == (32, len(chs))
        assert m.observed.all() and ((m.support >= 0) & (m.support <= 1)).all()
        sizes.add(strategy.apply(torch.randn(1, len(chs), 4), m).shape[1])
    assert sizes == {32}


# ---- data-driven and learned strategies --------------------------------------


def _fitted_wiener(**kw):
    rng = np.random.default_rng(0)  # training fields independent of the test's
    return get_channel_strategy("wiener", **kw).fit(_smooth_fields(rng, 2000), BENDR19)


def test_wiener_fidelity_smooth_fields_k8():
    # Report probe: 0.60 (covariance fitted on 2000 dense training fields).
    assert fidelity(_fitted_wiener(), _smooth_fields) <= 0.70


def test_wiener_requires_fit():
    with pytest.raises(ValueError, match=r"fit\(\)"):
        _build("wiener", BENDR19[:8])


def test_wiener_fit_travels_in_the_state_dict():
    src = resolve_montage(BENDR19[:8])
    fitted = _fitted_wiener()
    fresh = get_channel_strategy("wiener")
    fresh.load_state_dict(fitted.state_dict())
    torch.testing.assert_close(
        fresh.build(src, MONTAGE19).weights, fitted.build(src, MONTAGE19).weights
    )


def test_tokenizer_fit_refreshes_its_maps():
    from braindecode.modules import ChannelTokenizer

    tok = ChannelTokenizer(MONTAGE19, "wiener", src_chs_info=BENDR19[:8])
    x = torch.randn(1, 8, 5)
    with pytest.raises(ValueError, match=r"fit\(\)"):
        tok(x)
    tok.fit(_smooth_fields(np.random.default_rng(0), 2000), BENDR19)
    assert tok(x).x.shape == (1, 19, 5)


def test_region_averages_the_40_mm_neighbourhood():
    # FC3, CP3 and C1 lie within 40 mm of C3; Oz is the only source near O1;
    # nothing observed is near Fp1.
    m = _build("region", _named(["FC3", "CP3", "C1", "Oz"]))
    c3, o1, fp1 = 8, 17, 0
    np.testing.assert_allclose(m.weights[c3].numpy(), [1 / 3, 1 / 3, 1 / 3, 0])
    np.testing.assert_allclose(m.weights[o1].numpy(), [0, 0, 0, 1])
    assert (m.weights[fp1] == 0).all()
    assert not m.observed[fp1] and m.support[fp1] == 0


LATENT_TARGETS = {
    "montage": (MONTAGE19, 19),
    "ids": (ChannelTarget("ids", vocabulary=("Fz", "Cz", "Pz", "Oz", "Nope")), 5),
    "slots": (ChannelTarget("slots", chs_info=BENDR19, n_slots=6), 6),
    "positions": (ChannelTarget("positions", chs_info=BENDR19[:12]), 12),
    "free": (ChannelTarget("free"), 16),
}


@pytest.mark.parametrize("name", LATENT_TARGETS)
def test_latent_output_shape_for_each_interface(name):
    target, K = LATENT_TARGETS[name]
    strategy = get_channel_strategy("latent", n_latents=16)
    m = strategy.build(resolve_montage(BENDR19[3:10]), target)
    out = strategy.apply(torch.randn(2, 7, 11), m)
    assert out.shape == (2, K, 11) and torch.isfinite(out).all()
    assert len(list(strategy.parameters())) > 0


@pytest.mark.parametrize("name", ["montage", "free"])
def test_latent_ignores_input_order(name):
    target, _ = LATENT_TARGETS[name]
    strategy = get_channel_strategy("latent", n_latents=16)
    chs = BENDR19[3:10]
    perm = [4, 0, 6, 2, 1, 5, 3]
    x = torch.randn(2, 7, 11)
    out = strategy.apply(x, strategy.build(resolve_montage(chs), target))
    out_perm = strategy.apply(
        x[:, perm], strategy.build(resolve_montage([chs[i] for i in perm]), target)
    )
    torch.testing.assert_close(out_perm, out)

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel layer: regressions A-G and the invariants every strategy keeps."""

import copy
import inspect
import pickle  # nosec B403

import mne
import numpy as np
import pytest
import torch

from braindecode.datasets import pad_channels_collate
from braindecode.functional import exact_derivation
from braindecode.models import BENDR, LUNA, Labram, SignalJEPA
from braindecode.modules import ChannelLayer
from braindecode.modules.channels import STRATEGIES

TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()
SUBSET = ["Fp1", "F3", "Fz", "C3", "Cz", "P4", "Pz", "O2"]


def chs(names, kind="eeg"):
    return [{"ch_name": n, "kind": kind, "loc": np.zeros(12)} for n in names]


def at(name, xyz):
    return {"ch_name": name, "kind": "eeg", "loc": np.r_[xyz, np.zeros(9)]}


def standard_info(names):
    """``mne.Info`` with ``names`` at their 10-05 positions, as ``raw.set_montage`` places them."""
    info = mne.create_info(names, 100.0, "eeg")
    info.set_montage("colin27_1005")
    return info


def standard_positions(names):
    return np.array([ch["loc"][:3] for ch in standard_info(names)["chs"]])


def mne_spline(x, inputs, targets, reg):
    """``targets`` (``A`` or ``A-B``) from the signals ``x`` of the electrodes ``inputs``
    as MNE's ``interpolate_to`` gives them; a measured electrode stays its signal."""
    electrodes = list(dict.fromkeys(e for t in targets for e in t.split("-")))
    raw = mne.io.RawArray(x, standard_info(inputs), verbose="error")
    sensors = standard_info(electrodes).get_montage()
    with mne.utils.use_log_level("ERROR"):  # the sphere fit of a sparse montage warns
        interpolated = raw.interpolate_to(sensors, method="spline", reg=reg).get_data()
    value = {
        e: x[inputs.index(e)] if e in inputs else row
        for e, row in zip(electrodes, interpolated)
    }
    pairs = [t.partition("-") for t in targets]
    return np.stack([value[a] - (value[b] if b else 0) for a, _, b in pairs])


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
    cz = standard_positions(["Cz"])[0]
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


def test_d_non_eeg_ch_type_without_kind_rejected():
    """A serialized chs_info may carry ``ch_type`` and no ``kind``."""
    src = [{"ch_name": c["ch_name"], "ch_type": "mag"} for c in chs(SUBSET)]
    with pytest.raises(ValueError, match="not EEG"):
        ChannelLayer(chs(TEN_TWENTY), "spline", src)
    layer = ChannelLayer(chs(SUBSET), "exact", chs(SUBSET))
    x = torch.randn(1, len(SUBSET), 10)
    layer(x, chs_info=chs(SUBSET))
    with pytest.raises(ValueError, match="not EEG"):  # same names, other type
        layer(x, chs_info=[dict(c, kind=None, ch_type="mag") for c in chs(SUBSET)])


@pytest.mark.parametrize("per_sample", [False, True])
def test_dropped_non_eeg_channels_do_not_propagate_nans(per_sample):
    inputs = chs(["F3"]) + chs(["EOG"], kind="eog")
    x = torch.tensor([[[7.0, 3.0], [float("nan"), float("nan")]]], requires_grad=True)
    layer = ChannelLayer(chs(["F3"]), "exact", inputs, drop_non_eeg=True)
    values, observed = layer(x, [inputs] if per_sample else inputs)
    torch.testing.assert_close(values, x[:, :1])
    assert observed.all()
    values.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))


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
    with pytest.warns(UserWarning, match=r"\['SCALE'\] .* zero-filled"):
        layer = ChannelLayer(chs([*TEN_TWENTY, "SCALE"]), "spline", src)
    x, observed = layer(torch.randn(1, 4, 5))
    assert not observed[-1] and (x[0, -1] == 0).all() and observed.sum() == 3


@pytest.mark.parametrize("strategy", ["spline", "field", "source"])
def test_g_fewer_than_four_positions_is_declared(strategy):
    with pytest.raises(ValueError, match="needs at least 4 channels with a position"):
        ChannelLayer(chs(TEN_TWENTY), strategy, chs(["Fz", "Cz", "Pz"]))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_permutation_is_an_exact_copy(strategy):
    names = [*TEN_TWENTY, "F3-C3"]  # a measured bipolar channel is copied too
    x = torch.randn(2, 20, 10)
    layer = ChannelLayer(chs(names), strategy, chs(names[::-1]))
    assert torch.equal(layer(x)[0], x.flip(1))


LEFT, RIGHT = [0.03, 0.05, 0.08], [-0.03, 0.05, 0.08]


def swap_locations(inputs):
    inputs[0]["loc"][:3], inputs[1]["loc"][:3] = RIGHT, LEFT


@pytest.mark.parametrize(
    "change",
    [
        list.reverse,  # the order of the channels
        swap_locations,
        lambda inputs: inputs[0].update(kind="eog"),  # the type, as ``kind``
        lambda inputs: inputs[0].update(kind=None, ch_type="mag"),  # or as ``ch_type``
    ],
    ids=["order", "location", "kind", "ch_type"],
)
def test_maps_follow_in_place_changes_of_the_input_montage(change):
    """The cached maps are keyed by what the montage holds, not by the list object."""
    inputs = [at("E1", LEFT), at("E2", RIGHT)]
    target = [at("T1", LEFT), at("T2", RIGHT)]  # each is copied from the input beside it
    layer = ChannelLayer(target, "nearest", inputs, drop_non_eeg=True)
    x = torch.tensor([[[1.0], [2.0]]])
    before = layer(x)[0]
    change(inputs)
    fresh = ChannelLayer(target, "nearest", copy.deepcopy(inputs), drop_non_eeg=True)
    torch.testing.assert_close(layer(x)[0], fresh(x)[0])
    assert not torch.equal(layer(x)[0], before)


@pytest.mark.parametrize("literal", ["F3-C3-P3", "F3-", "-C3"])
def test_only_two_electrode_names_form_a_bipolar_derivation(literal):
    names = ["F3", "C3", "P3"]
    targets = ["F3", literal]
    _, exact = exact_derivation(targets, names)
    with pytest.warns(UserWarning, match="zero-filled"):
        layer = ChannelLayer(chs(targets), "zero", chs(names))
    signal = torch.tensor([[[5.0], [20.0], [2.0]]])
    values, observed = layer(signal)
    torch.testing.assert_close(observed, exact)
    torch.testing.assert_close(values, torch.tensor([[[5.0], [0.0]]]))
    measured = ChannelLayer(chs(targets), "exact", chs(targets))
    torch.testing.assert_close(measured(signal[:, :2])[0], signal[:, :2])


def test_bipolar_target_is_a_derivation():
    layer = ChannelLayer(chs(["F3-C3"]), "spline", chs(SUBSET))
    x = torch.randn(1, 8, 5)
    assert torch.equal(layer(x)[0][0, 0], x[0, 1] - x[0, 3])


@pytest.mark.parametrize(
    "strategy,kw", [("field", {}), ("source", {}), ("source", {"trainable": True})]
)
def test_constant_input_gives_constant_output(strategy, kw):
    # A target on the vertical axis of the template head (centre -1, 15, 39 mm) must
    # stay finite.
    target = chs(TEN_TWENTY) + [at("Zaxis", [-0.001, 0.0148, 0.1192])]
    out, _ = ChannelLayer(target, strategy, chs(SUBSET), **kw)(torch.ones(1, 8, 4))
    torch.testing.assert_close(out, torch.ones_like(out))


def test_source_fidelity_on_another_head():
    # Truth from a head the inversion does not use: 4 shells, other centre,
    # radius and skull conductivity. Measured 0.759 (MNE's default sphere: 0.820).
    info = standard_info(TEN_TWENTY)
    with mne.utils.use_log_level("ERROR"):
        sph = mne.make_sphere_model((0, 0.008, 0.045), 0.095, sigmas=(0.33, 1, 0.002, 0.33))
        src = mne.setup_volume_source_space(sphere=sph, pos=10.0, exclude=20.0)
        L = mne.make_forward_solution(info, None, src, sph, meg=False)["sol"]["data"]
    rng = np.random.default_rng(0)
    S = np.zeros((L.shape[1], 300))  # 3 dipoles of random orientation per sample
    for i, d in enumerate(rng.integers(0, L.shape[1] // 3, (300, 3))):
        S[(3 * d[:, None] + np.arange(3)).ravel(), i] = rng.standard_normal(9)
    truth = torch.tensor(L @ S).float()
    idx = [TEN_TWENTY.index(n) for n in SUBSET]
    miss = [i for i in range(19) if i not in idx]
    out = ChannelLayer(chs(TEN_TWENTY), "source", chs(SUBSET))(truth[idx][None])[0][0]
    out, truth = out - out.mean(0), truth - truth.mean(0)
    err = (out[miss] - truth[miss]).norm() / truth[miss].norm()
    assert err < 0.78


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
    flipped = layer(x.to(device, dtype).flip(1), chs(SUBSET[::-1]))[0]  # new montage
    assert layer.weight.device.type == device and layer.weight.dtype == dtype
    torch.testing.assert_close(flipped, out, rtol=1e-4, atol=1e-4)
    assert torch.equal(layer(x.to(device, dtype))[0], out)  # back from the cache


def smooth_field(n, rng):
    """``(n, 19)`` random quadratic fields sampled on the 10-20 positions."""
    P = standard_positions(TEN_TWENTY) / 0.1
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


def test_a_map_built_under_inference_mode_trains():
    layer = ChannelLayer(chs(TEN_TWENTY), "latent", chs(SUBSET))
    x, new = torch.randn(2, 8, 50), chs(SUBSET[::-1])
    with torch.inference_mode():  # e.g. a validation batch on a new montage
        layer(x, new)
    layer(x, new)[0].sum().backward()


@pytest.mark.parametrize("strategy", ["exact", "zero"])
def test_measured_bipolar_target_is_copied(strategy):
    target = [at("F3-C3", [-0.06, 0.02, 0.1]), chs(["Fz"])[0]]  # LaBraM: a midpoint
    x = torch.randn(1, 2, 5)
    layer = ChannelLayer(target, strategy, chs(["F3-C3", "Fz"]))
    assert torch.equal(layer(x)[0], x) and layer(x)[1].all()


def test_trainable_source_with_nothing_to_reconstruct():
    layer = ChannelLayer(chs(["Cz"]), "source", chs(["Cz"]), trainable=True)
    x = torch.randn(1, 1, 5)
    assert torch.equal(layer(x)[0], x)


@pytest.mark.parametrize(
    "strategy,kw",
    [("zero", {}), ("spline", {}), ("latent", {}), ("source", {"trainable": True})],
)
def test_no_target_observed_is_declared(strategy, kw):
    # BrainBERT-like target without a position, 10-20 input: only zeros.
    with pytest.raises(ValueError, match="would see only zeros"):
        ChannelLayer(chs(["E1"]), strategy, chs(SUBSET), **kw)


@pytest.mark.parametrize("cls", [BENDR, LUNA, Labram])
def test_forward_applies_the_layer_once(cls):
    model = cls(chs_info=chs(SUBSET), n_outputs=2, n_times=800, sfreq=200,
                channel_strategy="spline").eval()
    calls = []
    model.channel_layer.register_forward_hook(lambda *a: calls.append(1))
    x = torch.randn(1, 8, 800)
    with torch.no_grad():
        y = model(x)
        assert torch.equal(model.forward(x), y)
        y_perm = model.forward(x.flip(1), chs_info=chs(SUBSET[::-1]))
        torch.testing.assert_close(y_perm, y)
    assert len(calls) == 3
    assert "forward" not in vars(cls(chs_info=chs(SUBSET), n_outputs=2, n_times=800,
                                     sfreq=200))  # native: the class forward


def test_strategy_model_survives_deepcopy_and_pickle():
    model = LUNA(chs_info=chs(SUBSET), n_outputs=2, n_times=800, sfreq=200,
                 channel_strategy="spline").eval()
    x = torch.randn(1, 8, 800)
    y = model(x)
    clones = (copy.deepcopy(model), pickle.loads(pickle.dumps(model)))  # nosec B301
    for clone in clones:
        assert clone.forward.__self__ is clone
        assert torch.equal(clone(x), y) and torch.equal(clone.forward(x), y)


def test_latent_half_precision_stays_finite():
    layer = ChannelLayer(chs(TEN_TWENTY), "latent", chs(SUBSET)).half()
    for x in (torch.zeros(1, 8, 20), 1e-5 * torch.randn(1, 8, 20)):  # var underflows
        assert torch.isfinite(layer(x.half())[0]).all()


@pytest.mark.parametrize("cls", [BENDR, LUNA])  # forward(x, ...) / forward(X, ...)
def test_input_by_keyword(cls):
    model = cls(chs_info=chs(SUBSET), n_outputs=2, n_times=800, sfreq=200,
                channel_strategy="spline").eval()
    x = torch.randn(1, 8, 800)
    name = list(inspect.signature(type(model).forward).parameters)[1]
    assert torch.equal(model(**{name: x}, chs_info=chs(SUBSET)), model(x))


def test_labram_ch_names_describe_the_input():
    model = Labram(chs_info=chs(SUBSET), n_outputs=2, n_times=800, sfreq=200,
                   channel_strategy="spline").eval()
    x = torch.randn(1, 8, 800).flip(1)
    expected = model(x, chs_info=chs(SUBSET[::-1]))
    assert torch.equal(model(x, ch_names=SUBSET[::-1]), expected)


def test_config_saves_the_input_montage_not_the_target(tmp_path):
    small = dict(encoder_h=16, contextualizer_hidden=32, transformer_layers=1,
                 transformer_heads=4)  # the saved weights stay small
    model = BENDR(chs_info=chs(SUBSET), n_outputs=2, n_times=1280, sfreq=256,
                  channel_strategy="spline", **small).eval()
    x = torch.randn(1, 8, 1280)
    config = model.get_config()
    assert [c["ch_name"] for c in config["chs_info"]] == SUBSET
    clone = BENDR.from_config(config).eval()
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone(x), model(x))
    pytest.importorskip("huggingface_hub")
    model.save_pretrained(tmp_path)
    assert torch.equal(BENDR.from_pretrained(tmp_path).eval()(x), model(x))


def test_signal_jepa_strategy_targets_the_pretraining_channels():
    kw = dict(n_outputs=2, n_times=256, sfreq=128)
    full = SignalJEPA(**kw, channel_embedding="pretrain_aligned").state_dict()
    names_only = [{"ch_name": "Fz"}, {"ch_name": "Cz", "loc": None}]
    model = SignalJEPA(chs_info=names_only, **kw, channel_strategy="zero")
    model.load_state_dict(full, strict=True)  # the 62-channel checkpoint
    assert torch.isfinite(model(torch.randn(1, 2, 256))).all()
    with pytest.raises(ValueError, match="not in the input"):
        SignalJEPA(chs_info=names_only, **kw, channel_strategy="exact")


def test_direct_bipolar_match_precedes_reverse_when_both_are_measured():
    layer = ChannelLayer(chs(["F3-C3"]), "exact", chs(["C3-F3", "F3-C3"]))
    x = torch.tensor([[[2.0, 5.0], [-2.0, -5.0]]])
    y, observed = layer(x)
    assert torch.equal(y, x[:, 1:2])
    assert layer.weight.tolist() == [[0.0, 1.0]]
    assert observed.tolist() == [True]


def test_reverse_bipolar_beats_midpoint_spatial_match():
    # Reverse bipolar measurements have the SAME midpoint as the requested
    # derivation. Copying by geometry would silently give the wrong polarity.
    midpoint = [0.01, 0.02, 0.03]
    target = [at("F3-C3", midpoint)]
    input_chs = [at("C3-F3", midpoint)]
    layer = ChannelLayer(target, "exact", input_chs)
    x = torch.tensor([[[3.5, 0.0, -1.25]]])
    y, observed = layer(x)
    assert layer.weight.tolist() == [[-1.0]]
    assert torch.equal(y, -x)
    assert observed.tolist() == [True]


def test_targets_from_any_combination_of_measured_channels():
    # Potentials of 6 electrodes; the input is a TCP-style chain and a pair
    # against the recording reference; every target is exact without positions.
    v = torch.randn(1, 6, 50, dtype=torch.float64)
    idx = dict(zip("Fp1 F7 T7 P7 O1 C3".split(), range(6)))
    inputs = ["Fp1-F7", "F7-T7", "T7-P7", "P7-O1", "C3-REF", "P7-REF"]
    x = torch.cat(
        [v[:, [idx[a]]] - (v[:, [idx[b]]] if b in idx else 0)
         for a, b in (n.split("-") for n in inputs)], 1
    )
    targets = ["Fp1-T7", "O1-T7", "C3-P7", "C3-O1"]
    layer = ChannelLayer(chs(targets), "exact", chs(inputs)).double()
    y, observed = layer(x)
    expected = torch.cat([v[:, [idx[a]]] - v[:, [idx[b]]] for a, b in (n.split("-") for n in targets)], 1)
    torch.testing.assert_close(y, expected, rtol=0, atol=1e-12)
    assert observed.all()


def without(names, *dropped):
    return [n for n in names if n not in dropped]


@pytest.mark.filterwarnings("ignore:Strategy 'spline'")  # row gain of a sparse montage
@pytest.mark.parametrize("reg", [1e-3, 0.0])
@pytest.mark.parametrize(
    "inputs, recorded, targets",
    [
        (SUBSET, [n.swapcase() for n in SUBSET], TEN_TWENTY),
        (  # legacy names: T3 is T7, T5 is P7
            ["Fp1", "F3", "Fz", "T7", "Cz", "P7", "Pz", "O2"],
            ["Fp1", "F3", "Fz", "T3", "Cz", "T5", "Pz", "O2"],
            TEN_TWENTY,
        ),
        (SUBSET, [f"EEG {n.upper()}-REF" for n in SUBSET], TEN_TWENTY),
        (without(TEN_TWENTY, "F4", "C4"), None, ["F3-C3", "F4-P4", "F4-C4"]),
        (without(TEN_TWENTY, "Cz", "Pz"), None, TEN_TWENTY),
    ],
    ids=["mixed case", "alias", "EEG X-REF", "bipolar", "two targets missing"],
)
def test_spline_matches_mne_interpolate_to(inputs, recorded, targets, reg):
    """Whatever the input names, the targets are what MNE's ``interpolate_to`` gives."""
    x = np.random.default_rng(0).standard_normal((len(inputs), 40))
    montage = [
        dict(ch, ch_name=name)  # the standard position, recorded under another name
        for ch, name in zip(standard_info(inputs)["chs"], recorded or inputs)
    ]
    layer = ChannelLayer(chs(targets), "spline", montage, reg=reg).double()
    out, _ = layer(torch.from_numpy(x)[None])
    expected = mne_spline(x, inputs, targets, reg)
    np.testing.assert_allclose(out[0].numpy(), expected, atol=1e-6, rtol=0)


def test_one_montage_per_sample():
    layer = ChannelLayer(chs(TEN_TWENTY), "spline", chs(TEN_TWENTY))
    x = torch.randn(2, len(TEN_TWENTY), 30)
    montages = [chs(TEN_TWENTY), chs(SUBSET)]
    padded = x.clone()
    padded[1, len(SUBSET):] = 0
    y, observed = layer(padded, montages)
    torch.testing.assert_close(y[1], layer(padded[1:, : len(SUBSET)], montages[1])[0][0])
    torch.testing.assert_close(y[0], x[0])
    assert observed.shape == (2, len(TEN_TWENTY)) and observed[0].all()
    mask = torch.zeros(2, len(TEN_TWENTY), dtype=torch.bool)  # channels anywhere
    mask[0], mask[1, -len(SUBSET):] = True, True
    shifted = padded.clone()
    shifted[1] = padded[1].roll(len(TEN_TWENTY) - len(SUBSET), 0)
    torch.testing.assert_close(layer(shifted, montages, mask)[0], y)
    with pytest.raises(ValueError, match="one montage per sample"):
        layer(x, chs(TEN_TWENTY), mask)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_padded_montages_ignore_nan_in_signal_and_gradient(explicit_mask):
    montages = [chs(["F3", "C3"]), chs(["F3"])]
    windows = [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 0, [0, 0, 2]),
        (np.array([[5.0, 6.0]]), 1, [0, 0, 2]),
    ]
    signal, _, _, mask = pad_channels_collate(windows, pad_value=float("nan"))
    if explicit_mask:
        signal[1] = signal[1].flip(0)
        mask[1] = mask[1].flip(0)
    signal.requires_grad_()
    layer = ChannelLayer(chs(["F3"]), "exact")
    values, observed = layer(signal, montages, mask if explicit_mask else None)
    torch.testing.assert_close(values, torch.tensor([[[1.0, 2.0]], [[5.0, 6.0]]]))
    assert observed.all()
    values.sum().backward()
    assert (signal.grad[~mask] == 0).all()


def test_padded_montages_require_a_real_channel_and_preserve_measured_nan():
    montage = chs(["F3"])
    layer = ChannelLayer(montage, "exact")
    x = torch.tensor([[[float("nan")]]])
    values, _ = layer(x, [montage], torch.ones(1, 1, dtype=torch.bool))
    assert torch.isnan(values).all()
    with pytest.raises(ValueError, match="at least one real channel"):
        layer(x, [montage], torch.zeros(1, 1, dtype=torch.bool))


def test_strategy_model_takes_pad_channels_collate_batches():
    # pos + ch_mask (no names): the montage of each sample from its positions.
    model = Labram(chs_info=chs(SUBSET), n_outputs=2, n_times=800, sfreq=200,
                   channel_strategy="spline").eval()
    pos = torch.as_tensor(standard_positions(SUBSET)).expand(2, -1, -1).clone()
    mask = torch.ones(2, len(SUBSET), dtype=torch.bool)
    mask[1, 5:] = False
    x = torch.randn(2, len(SUBSET), 800) * mask[..., None]
    named = [chs(SUBSET), chs(SUBSET[:5])]
    with torch.no_grad():
        torch.testing.assert_close(
            model(x, pos=pos, ch_mask=mask), model(x, chs_info=named, ch_mask=mask)
        )
        torch.testing.assert_close(
            model(x, pos=pos, ch_mask=mask.int()), model(x, pos=pos, ch_mask=mask)
        )
        # pos without ch_mask (set_return_ch_pos, no padding): the construction
        # montage maps x, the backbone never gets the input's positions
        torch.testing.assert_close(model(x[:1], pos=pos[:1]), model(x[:1]))

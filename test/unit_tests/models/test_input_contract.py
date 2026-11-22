"""Contract tests for declared input axes, minimum length, and length-grid snapping."""

import inspect
import warnings

import numpy as np
import pytest
import torch

import braindecode.models.base as base_mod
from braindecode.datasets import pad_channels_collate
from braindecode.models import (
    AXON,
    CORTEG,
    DIVER1,
    LUNA,
    BrainBERT,
    CBraMod,
    EEGConformer,
    FBCNet,
    FBMSNet,
    IFNet,
    SensingDynamics,
    TinyMyo,
)
from braindecode.models.util import models_dict, models_mandatory_parameters
from test.unit_tests.models._fixtures import (
    _TOKENIZERS,
    PARAMS,
    _build_case,
    _materialize,
    _shapes,
    _summary_row,
    _tensor_leaves,
    non_classification_models,
)

# channel_tokens is an upper-bound table (1 <= n_chans_in <= n_chans); see #1542.
_UPPER_BOUND_CHANNEL_MODELS = {"BIOT"}


def _matches_free_time(out, ref, *, longer: bool) -> bool:
    if _shapes(out) == _shapes(ref):
        return True
    out_leaves = list(_tensor_leaves(out))
    ref_leaves = list(_tensor_leaves(ref))
    if len(out_leaves) != len(ref_leaves):
        return False
    for lo, lr in zip(out_leaves, ref_leaves):
        if lo.ndim != lr.ndim or lo.ndim < 3 or lo.shape[0] != lr.shape[0]:
            return False
        diffs = [d for d in range(1, lo.ndim) if lo.shape[d] != lr.shape[d]]
        if len(diffs) > 1:
            return False
        if diffs:
            d = diffs[0]
            if longer and lo.shape[d] <= lr.shape[d]:
                return False
    return True


def _matches_free_channels(out, ref, n_chans: int) -> bool:
    if _shapes(out) == _shapes(ref):
        return True
    out_leaves = list(_tensor_leaves(out))
    ref_leaves = list(_tensor_leaves(ref))
    if len(out_leaves) != len(ref_leaves):
        return False
    for lo, lr in zip(out_leaves, ref_leaves):
        if lo.ndim != lr.ndim or lo.ndim < 3:
            return False
        if (
            lo.shape[0] != lr.shape[0]
            or lo.shape[1] != n_chans
            or lo.shape[2:] != lr.shape[2:]
        ):
            return False
    return True


def test_every_registered_model_declares_axes_in_own_class_body():
    for name, cls in models_dict.items():
        assert cls.__dict__.get("_time_axis") in {"fixed", "free", "dense"}, name
        assert cls.__dict__.get("_channel_axis") in {"fixed", "free"}, name


@PARAMS
def test_time_axis_matches_forward(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        ref = model(x)
        calls.clear()
        if model._time_axis == "fixed":
            with pytest.raises(
                ValueError, match=rf"built for n_times={T} and got {2 * T} samples"
            ):
                model(torch.randn(2, C, 2 * T))
            assert calls == []
        elif model._time_axis == "free":
            lengths = [2 * T, T + max(1, model._n_times_step)]
            if model._min_n_times is not None and T // 2 >= model._min_n_times:
                lengths.append(T // 2)
            for t in lengths:
                out = model(torch.randn(2, C, t))
                if model_name in non_classification_models:
                    assert _matches_free_time(out, ref, longer=(t == 2 * T))
                else:
                    assert _shapes(out) == _shapes(ref)
        else:
            longer = model(torch.randn(2, C, 2 * T))
            assert (
                next(_tensor_leaves(longer)).numel() > next(_tensor_leaves(ref)).numel()
            )


@PARAMS
def test_min_n_times_runs_and_one_less_refuses(
    model_name, required_params, signal_params
):
    model, x = _build_case(model_name, required_params, signal_params)
    if model._time_axis == "fixed":
        pytest.skip("fixed length")
    _materialize(model, x)
    m = model._min_n_times
    assert isinstance(m, int) and m >= 1
    with torch.no_grad():
        model(torch.randn(2, x.shape[1], m))
    if m > 1:
        calls = []
        model.register_forward_pre_hook(lambda mod, a: calls.append(1))
        with pytest.raises(ValueError, match=rf"needs at least {m} time samples"):
            model(torch.randn(2, x.shape[1], m - 1))
        assert calls == []


def test_lazy_model_keeps_channel_count_of_first_call():
    model = CBraMod(n_outputs=2, n_times=400).eval()  # no n_chans: lazy head
    with torch.no_grad():
        model(torch.randn(2, 3, 400))
        with pytest.raises(ValueError, match="built for n_chans=3 and got 4"):
            model(torch.randn(2, 4, 400))


def test_length_grid_snaps_within_cap_and_warns_once():
    model = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.warns(UserWarning, match="snapped"):
        out = model(torch.randn(2, 3, 1001))
    assert out.shape == (2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_pad = model(torch.randn(2, 3, 999))
    assert out_pad.shape == (2, 2)
    assert not [w for w in caught if "snapped" in str(w.message)]
    with pytest.raises(ValueError, match="built for n_times=1000"):
        model(torch.randn(2, 3, 1020))

    strict = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    strict.on_non_divisible = "error"
    with pytest.raises(ValueError, match="built for n_times=1000"):
        strict(torch.randn(2, 3, 1001))

    luna = LUNA(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.warns(UserWarning, match=r"LUNA snapped input length 1 to 40"):
        assert luna(torch.randn(2, 3, 1)).shape == (2, 2)


@PARAMS
def test_channel_axis_matches_forward(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        ref = model(x)
        calls.clear()
        candidates = [C + 1] + ([C - 1] if C > 1 else [])
        if model._channel_axis == "fixed":
            for c in candidates:
                if model_name in _UPPER_BOUND_CHANNEL_MODELS and c <= C:
                    assert _shapes(model(torch.randn(2, c, T))) == _shapes(ref)
                    continue
                calls.clear()
                with pytest.raises(
                    ValueError,
                    match=rf"built for n_chans={C} and got {c} channels",
                ):
                    model(torch.randn(2, c, T))
                assert calls == []
        else:
            for c in candidates:
                out = model(torch.randn(2, c, T))
                if model_name in _TOKENIZERS:
                    assert _matches_free_channels(out, ref, c)
                else:
                    assert _shapes(out) == _shapes(ref)


@PARAMS
def test_check_reads_keyword_input(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    name = list(inspect.signature(type(model).forward).parameters)[1]
    if model._channel_axis == "fixed":
        calls = []
        model.register_forward_pre_hook(lambda m, a: calls.append(1))
        with pytest.raises(ValueError, match="channels"):
            model(**{name: torch.randn(2, C + 1, T)})
        assert calls == []
    with torch.no_grad():
        assert _shapes(model(**{name: x})) == _shapes(model(x))


def test_padded_batch_passes_channel_check():
    tested = 0
    for model_name, required_params, signal_params in models_mandatory_parameters:
        cls = models_dict[model_name]
        if cls._channel_axis != "free":
            continue
        if "ch_mask" not in inspect.signature(cls.forward).parameters:
            continue
        model, x = _build_case(model_name, required_params, signal_params)
        _materialize(model, x)
        C, T = x.shape[1:]
        c_short = max(1, C - 2)
        windows = [
            (torch.randn(C, T).numpy(), 0, [0, 0, T]),
            (torch.randn(c_short, T).numpy(), 1, [0, 0, T]),
        ]
        X, _, _, mask = pad_channels_collate(windows)
        with torch.no_grad():
            out = model(X, ch_mask=mask)
        leaves = list(_tensor_leaves(out))
        assert leaves, model_name
        for leaf in leaves:
            assert torch.isfinite(leaf).all(), model_name
        tested += 1
    assert tested >= 5


@PARAMS
def test_eval_is_deterministic_and_batch_independent(
    model_name, required_params, signal_params
):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.no_grad():
        a = model(x)
        b = model(x)
        one = model(x[:1])
    a_leaves = list(_tensor_leaves(a))
    b_leaves = list(_tensor_leaves(b))
    one_leaves = list(_tensor_leaves(one))
    assert len(a_leaves) == len(b_leaves) == len(one_leaves)
    for la, lb in zip(a_leaves, b_leaves):
        assert torch.equal(la, lb)
    for la, l1 in zip(a_leaves, one_leaves):
        if la.ndim >= 1 and la.shape[0] == x.shape[0] and l1.shape[0] == 1:
            tol = 1e-4 * max(1.0, l1.abs().max().item())
            torch.testing.assert_close(la[:1], l1, rtol=1e-4, atol=tol)


def test_no_hf_hub_backfill_and_diver1_on_non_divisible(monkeypatch):
    seeg_chs = [
        {
            "ch_name": f"A{i + 1}",
            "kind": "seeg",
            "loc": np.array([0.01 * (i + 1), 0.02, -0.01, *([0.0] * 9)]),
        }
        for i in range(3)
    ]
    orig_eeg_init = base_mod.EEGModuleMixin.__init__
    had_hub_config = []

    def _init_without_hub_config(self, *args, **kwargs):
        self.__dict__.pop("_hub_mixin_config", None)
        had_hub_config.append(getattr(self, "_hub_mixin_config", None))
        return orig_eeg_init(self, *args, **kwargs)

    monkeypatch.setattr(base_mod, "HAS_HF_HUB", False)
    monkeypatch.setattr(base_mod.EEGModuleMixin, "__init__", _init_without_hub_config)

    luna = LUNA(
        n_chans=3, n_outputs=2, n_times=1000, embed_dim=16, depth=1, num_heads=2
    ).eval()
    assert had_hub_config == [None]
    assert luna.on_non_divisible == "pad"
    assert luna._min_n_times == 1
    with torch.no_grad(), pytest.warns(UserWarning, match="LUNA snapped"):
        assert luna(torch.randn(2, 3, 1)).shape == (2, 2)

    tinymyo = TinyMyo(
        n_chans=3, n_outputs=2, n_times=100, embed_dim=24, depth=1, num_heads=2
    ).eval()
    assert tinymyo.on_non_divisible == "pad"
    assert tinymyo._min_n_times == 1
    with torch.no_grad(), pytest.warns(UserWarning, match="TinyMyo snapped"):
        assert tinymyo(torch.randn(2, 3, 1)).shape == (2, 2)

    brainbert = BrainBERT(
        n_chans=2,
        n_outputs=2,
        n_times=2048,
        hidden_dim=32,
        ffn_dim=64,
        n_layers=1,
        n_heads=2,
    ).eval()
    assert brainbert.on_non_divisible == "pad"
    assert brainbert._min_n_times == 1

    diver = DIVER1(
        chs_info=seeg_chs,
        n_outputs=2,
        n_times=1000,
        sfreq=500.0,
        d_model=64,
        n_layers=1,
    ).eval()
    assert diver.on_non_divisible == "pad"


def test_axon_crop_suppresses_warning_and_pad_snaps_patches():
    chs_info = [
        {"ch_name": name, "kind": "eeg", "loc": np.zeros(12)}
        for name in ("Fz", "Cz", "Pz")
    ]
    axon = AXON(
        chs_info=chs_info,
        n_outputs=2,
        n_times=None,
        embed_dim=32,
        depth=1,
        num_heads=2,
    ).eval()
    x = torch.randn(2, 3, 1000)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crop_out = axon(x, return_features=True)
    assert not [w for w in caught if "snapped" in str(w.message)]
    assert crop_out["tokens"].shape == (2, 3, 5, 32)

    axon.on_non_divisible = "pad"
    with (
        torch.no_grad(),
        pytest.warns(UserWarning, match=r"AXON snapped input length 1000 to 1100"),
    ):
        pad_out = axon(x, return_features=True)
    assert pad_out["tokens"].shape == (2, 3, 6, 32)


def test_secondary_time_inputs_snap_positional_and_keyword():
    ecog_chs = [
        {
            "ch_name": f"G{i + 1}",
            "kind": "ecog",
            "loc": np.array([0.02 * (i + 1), 0.03, 0.05, *([0.0] * 9)]),
        }
        for i in range(2)
    ]
    corteg = CORTEG(
        chs_info=ecog_chs,
        n_outputs=2,
        n_times=64,
        embed_dim=32,
        depth=2,
        num_heads=2,
        n_merge_blocks=1,
        knn_k=2,
        on_non_divisible="pad",
    ).eval()
    seen_corteg = []
    corteg.register_forward_pre_hook(
        lambda _m, a, kw: seen_corteg.append((a, kw)), with_kwargs=True
    )
    x = torch.randn(2, 2, 30)
    x_hi = torch.randn(2, 2, 48)
    with torch.no_grad():
        out_pos = corteg(x, x_hi)
        out_kw = corteg(x=x, x_hi=x_hi)
    assert seen_corteg[0][0][0].shape == (2, 2, 32)
    assert seen_corteg[0][0][1].shape == (2, 2, 50)
    assert seen_corteg[1][1]["x"].shape == (2, 2, 32)
    assert seen_corteg[1][1]["x_hi"].shape == (2, 2, 50)
    torch.testing.assert_close(out_pos, out_kw)

    sd = SensingDynamics(
        n_chans=16,
        n_outputs=3,
        n_times=200,
        sfreq=2000.0,
        temporal_channels=8,
        mid_channels=8,
        spatial_channels=8,
        mlp_hidden=16,
    ).eval()
    sd._time_axis = "fixed"
    seen_sd = []
    sd.register_forward_pre_hook(
        lambda _m, a, kw: seen_sd.append((a, kw)), with_kwargs=True
    )
    x_emg = torch.randn(2, 16, 201)
    x_lp = torch.randn(2, 16, 201)
    with (
        torch.no_grad(),
        pytest.warns(
            UserWarning,
            match=r"SensingDynamics snapped input length 201 to n_times=200",
        ),
    ):
        sd_pos = sd(x_emg, x_lp)
    with torch.no_grad():
        sd_kw = sd(x=x_emg, x_lowpass=x_lp)
    assert seen_sd[0][0][0].shape == (2, 16, 200)
    assert seen_sd[0][0][1].shape == (2, 16, 200)
    assert seen_sd[1][1]["x"].shape == (2, 16, 200)
    assert seen_sd[1][1]["x_lowpass"].shape == (2, 16, 200)
    torch.testing.assert_close(sd_pos, sd_kw)


@pytest.mark.parametrize("cls,stride_factor", [(FBCNet, 4), (FBMSNet, 4), (IFNet, 8)])
def test_filterbank_models_pad_after_convs_and_script_dynamic_lengths(
    cls, stride_factor
):
    model_nondiv = cls(
        n_chans=3, n_outputs=2, n_times=1003, stride_factor=stride_factor, sfreq=250
    ).eval()
    x_1003 = torch.randn(2, 3, 1003)
    with torch.no_grad():
        assert torch.equal(model_nondiv(x_1003), model_nondiv.forward(x_1003))

    model_div = cls(
        n_chans=3, n_outputs=2, n_times=1000, stride_factor=stride_factor, sfreq=250
    ).eval()
    scripted = torch.jit.script(model_div)
    x_1001 = torch.randn(2, 3, 1001)
    with torch.no_grad(), pytest.warns(UserWarning, match="snapped"):
        eager_out = model_div(x_1001)
    with torch.no_grad():
        torch.testing.assert_close(scripted(x_1001), eager_out, atol=0, rtol=0)
    model_div.on_non_divisible = "crop"
    with torch.no_grad():
        # IFNet's 0.01-std init hides the trailing samples at unit scale
        crop_out = model_div(100.0 * x_1001)
        crop_scripted = torch.jit.script(model_div)(100.0 * x_1001)
    assert crop_out.shape == (2, 2)
    torch.testing.assert_close(crop_scripted, crop_out, atol=0, rtol=0)


_VALID_OUTPUT_TAGS = {
    "(batch, outputs)",
    "(batch, outputs, time)",
    "(batch, time, outputs)",
    "reconstruction",
    "(batch, tokens, dim)",
}


def test_summary_csv_matches_generator():
    from braindecode.models._summary import check_summary_csv

    diffs = check_summary_csv()
    assert diffs == [], "\n".join(diffs)


def test_summary_csv_declares_axes():
    for name, cls in models_dict.items():
        row = _summary_row(name)
        assert row["Time axis"] == cls._time_axis, name
        assert row["Channel axis"] == cls._channel_axis, name
        assert row["Output"] in _VALID_OUTPUT_TAGS, name
        for cap_col in (
            "Cropped decoding",
            "adapt_input",
            "reset_head",
            "return_features",
        ):
            assert row[cap_col] in {"yes", "no"}, (name, cap_col)


def test_summary_generator_subset_and_cli(tmp_path):
    import shutil

    from braindecode.models._summary import (
        SUMMARY_CSV_PATH,
        check_summary_csv,
        generate_summary_rows,
        main,
    )

    rows = generate_summary_rows(names=["EEGNet"])
    assert len(rows) == 1
    assert rows[0]["Model"] == "EEGNet"
    assert int(rows[0]["#Parameters"]) > 0
    assert rows[0]["Time axis"] == models_dict["EEGNet"]._time_axis

    temp_csv = tmp_path / "summary.csv"
    shutil.copyfile(SUMMARY_CSV_PATH, temp_csv)
    assert main(["--check", "--csv", str(temp_csv)]) == 0

    text = temp_csv.read_text(encoding="utf-8")
    corrupted = text.replace(f",{rows[0]['#Parameters']},", ",999999,", 1)
    temp_csv.write_text(corrupted, encoding="utf-8")

    diffs = check_summary_csv(temp_csv)
    assert any(d.startswith("-EEGNet,") and "999999" in d for d in diffs)
    assert main(["--check", "--csv", str(temp_csv)]) == 1
    assert (
        main(["--files", "braindecode/models/eegnet.py", "--csv", str(temp_csv)]) == 0
    )
    assert check_summary_csv(temp_csv) == []

    missing_editorial_csv = tmp_path / "missing_editorial.csv"
    missing_editorial_csv.write_text(
        "\n".join(
            line
            for line in temp_csv.read_text(encoding="utf-8").splitlines()
            if not line.startswith("EEGNet,")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="EEGNet is missing editorial column 'Application'"
    ):
        generate_summary_rows(names=["EEGNet"], csv_path=missing_editorial_csv)

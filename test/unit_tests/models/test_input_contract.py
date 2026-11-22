"""Declared input axes, minimum window length, length-grid snapping and eval
determinism across every registered model."""

import ast
import inspect
import shutil
import time
import warnings
from pathlib import Path

import pytest
import torch

from braindecode.datasets import pad_channels_collate
from braindecode.models import (
    CORTEG,
    LUNA,
    EEGConformer,
    EEGNet,
    MVPFormer,
    SensingDynamics,
)
from braindecode.models import util as _models_util
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
            assert calls == []  # refused before forward ran
        elif model._time_axis == "free":
            for t in (2 * T, T + max(1, model._n_times_step)):
                out = model(torch.randn(2, C, t))
                if model_name not in non_classification_models or model_name in {
                    "DANCE",
                    "CareSleepNet",
                }:
                    assert _shapes(out) == _shapes(ref)
                else:
                    assert len(_shapes(out)) == len(_shapes(ref))
                    for got_s, ref_s in zip(_shapes(out), _shapes(ref)):
                        assert len(got_s) == len(ref_s) and got_s[0] == ref_s[0]
                    if t == 2 * T:
                        assert (
                            next(_tensor_leaves(out)).numel()
                            > next(_tensor_leaves(ref)).numel()
                        )
        else:  # dense: a longer input gives more output steps
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
        out = model(torch.randn(2, x.shape[1], m))
    assert all(
        torch.isfinite(leaf).all()
        for leaf in _tensor_leaves(out)
        if leaf.is_floating_point() or leaf.is_complex()
    )
    if m > 1:
        with pytest.raises(ValueError, match=rf"needs at least {m} time samples"):
            model(torch.randn(2, x.shape[1], m - 1))


def test_length_grid_snaps_within_cap_and_warns_once():
    model = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.warns(UserWarning, match="snapped"):
        out = model(torch.randn(2, 3, 1001))
    assert out.shape == (2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.randn(2, 3, 1001))
    assert not [w for w in caught if "snapped" in str(w.message)]
    with pytest.raises(ValueError, match="built for n_times=1000"):
        model(torch.randn(2, 3, 1020))  # > 1% of 1000: refused


@PARAMS
def test_channel_axis_matches_forward(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    candidates = [C + 1, *([C - 1] if C > 1 else [])]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        ref = model(x)
        for c in candidates:
            calls.clear()
            if model._channel_axis == "fixed":
                with pytest.raises(
                    ValueError,
                    match=rf"built for n_chans={C} and got {c} channels",
                ):
                    model(torch.randn(2, c, T))
                assert calls == []
            elif model_name in _TOKENIZERS:
                expected = [(s[0], c, *s[2:]) for s in _shapes(ref)]
                assert _shapes(model(torch.randn(2, c, T))) == expected
            else:
                assert _shapes(model(torch.randn(2, c, T))) == _shapes(ref)


@PARAMS
def test_check_reads_keyword_input(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    params = inspect.signature(type(model).forward).parameters
    name = list(params)[1]
    takes_ch_mask = "ch_mask" in params
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        if model._channel_axis == "fixed":
            with pytest.raises(ValueError, match="channels"):
                model(**{name: torch.randn(2, C + 1, T)})
            assert calls == []
            if takes_ch_mask:
                with pytest.raises(ValueError, match="channels"):
                    model(
                        **{
                            "ch_mask": torch.ones(2, C + 1, dtype=torch.bool),
                            name: torch.randn(2, C + 1, T),
                        }
                    )
                assert calls == []
        elif takes_ch_mask:
            mask = torch.ones(2, C, dtype=torch.bool)
            assert _shapes(model(**{"ch_mask": mask, name: x})) == _shapes(
                model(x, ch_mask=mask)
            )
        assert _shapes(model(**{name: x})) == _shapes(model(x))


def test_secondary_time_inputs_snap_and_reject_short_crops():
    corteg = CORTEG(
        n_chans=3,
        n_outputs=2,
        n_times=64,
        patch_size=16,
        hi_patch_size=25,
        embed_dim=32,
        depth=1,
        num_heads=2,
        n_merge_blocks=1,
    ).eval()
    pos = 0.1 * torch.rand(3, 3)
    seen = []
    corteg.register_forward_pre_hook(
        lambda m, a, kw: seen.append((a[0].shape, kw["x_hi"].shape)),
        with_kwargs=True,
    )
    with torch.no_grad():
        with pytest.warns(UserWarning, match="snapped"):
            out = corteg(
                torch.randn(2, 3, 66),
                x_hi=torch.randn(2, 3, 98),
                pos=pos,
            )
    assert out.shape == (2, 2)
    assert seen == [(torch.Size([2, 3, 80]), torch.Size([2, 3, 100]))]

    corteg_crop = CORTEG(
        n_chans=3,
        n_outputs=2,
        n_times=64,
        patch_size=16,
        hi_patch_size=25,
        embed_dim=32,
        depth=1,
        num_heads=2,
        n_merge_blocks=1,
        on_non_divisible="crop",
    ).eval()
    crop_calls = []
    corteg_crop.register_forward_pre_hook(lambda m, a: crop_calls.append(1))
    with pytest.raises(ValueError, match="x_hi"):
        corteg_crop(torch.randn(2, 3, 64), x_hi=torch.randn(2, 3, 20), pos=pos)
    assert crop_calls == []

    sd = SensingDynamics(
        n_chans=16,
        n_outputs=4,
        n_times=480,
        sfreq=2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=8,
        mlp_hidden=16,
    ).eval()
    sd._n_times_step = 8
    sd.on_non_divisible = "pad"
    sd_seen = []
    sd.register_forward_pre_hook(
        lambda m, a, kw: sd_seen.append((a[0].shape, kw["x_lowpass"].shape)),
        with_kwargs=True,
    )
    with torch.no_grad():
        with pytest.warns(UserWarning, match="snapped"):
            sd_out = sd(
                torch.randn(2, 16, 481),
                x_lowpass=torch.randn(2, 16, 481),
            )
    assert sd_out.shape == (2, 488, 4)
    assert sd_seen == [(torch.Size([2, 16, 488]), torch.Size([2, 16, 488]))]

    sd_fixed = SensingDynamics(
        n_chans=16,
        n_outputs=4,
        n_times=480,
        sfreq=2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=8,
        mlp_hidden=16,
    ).eval()
    sd_fixed.__dict__["_time_axis"] = "fixed"
    with torch.no_grad():
        with pytest.warns(UserWarning, match="snapped"):
            fixed_out = sd_fixed(
                torch.randn(2, 16, 481),
                x_lowpass=torch.randn(2, 16, 481),
            )
    assert fixed_out.shape == (2, 480, 4)
    fixed_calls = []
    sd_fixed.register_forward_pre_hook(lambda m, a: fixed_calls.append(1))
    with pytest.raises(ValueError, match="x_lowpass"):
        sd_fixed(torch.randn(2, 16, 480), x_lowpass=torch.randn(2, 16, 495))
    assert fixed_calls == []


def test_padded_batch_passes_channel_check():
    checked = 0
    for model_name, required_params, signal_params in models_mandatory_parameters:
        cls = models_dict[model_name]
        if cls._channel_axis != "free":
            continue
        if "ch_mask" not in inspect.signature(cls.forward).parameters:
            continue
        model, x = _build_case(model_name, required_params, signal_params)
        _materialize(model, x)
        C, T = x.shape[1:]
        c_full = C + 2
        c_short = max(1, C - 1)
        batch = [
            (torch.randn(c_full, T).numpy(), 0, [0, T, 0]),
            (torch.randn(c_short, T).numpy(), 1, [0, T, 0]),
        ]
        X, _, _, mask = pad_channels_collate(batch)
        assert X.shape[1] != model.n_chans
        with torch.no_grad():
            out = model(X, ch_mask=mask)
        leaves = list(_tensor_leaves(out))
        assert leaves
        assert all(
            torch.isfinite(leaf).all()
            for leaf in leaves
            if leaf.is_floating_point() or leaf.is_complex()
        ), model_name
        checked += 1
    assert checked >= 7

    luna = LUNA(
        n_chans=3,
        n_outputs=2,
        n_times=80,
        patch_size=20,
        embed_dim=32,
        num_queries=2,
        depth=1,
    ).eval()
    pos_batch = [
        (torch.randn(5, 80).numpy(), 0, [0, 80, 0], (0.1 * torch.rand(5, 3)).numpy()),
        (torch.randn(2, 80).numpy(), 1, [0, 80, 0], (0.1 * torch.rand(2, 3)).numpy()),
    ]
    X_pos, _, _, pos, mask_pos = pad_channels_collate(pos_batch)
    assert X_pos.shape[1] == 5 and pos.shape == (2, 5, 3)
    with torch.no_grad():
        luna_out = luna(X_pos, channel_locations=pos, ch_mask=mask_pos)
    assert luna_out.shape == (2, 2)
    assert torch.isfinite(luna_out).all()


def test_to_dense_prediction_model_updates_min_n_times_with_dilated_padding():
    for pool_mode, expected_min in (("max", 1035), ("mean", 1014)):
        net = EEGNet(
            n_chans=3,
            n_outputs=2,
            n_times=1000,
            pool_mode=pool_mode,
            final_layer_with_constraint=False,
        ).eval()
        net.to_dense_prediction_model()
        assert net._min_n_times == expected_min
        calls = []
        net.register_forward_pre_hook(lambda m, a: calls.append(1))
        with torch.no_grad():
            out = net(torch.randn(1, 3, expected_min))
        assert torch.isfinite(out).all()
        calls.clear()
        with pytest.raises(
            ValueError, match=rf"needs at least {expected_min} time samples"
        ):
            net(torch.randn(1, 3, expected_min - 1))
        assert calls == []


def test_mvpformer_concat_pooling_enforces_fixed_channels():
    assert MVPFormer.__dict__["_channel_axis"] == "free"
    model = MVPFormer(
        n_chans=4,
        n_outputs=2,
        n_times=256,
        segment_len=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_head_kv=2,
        d_inner=64,
        local_window=2,
        pooling="concat",
    ).eval()
    assert model._effective_channel_axis() == "fixed"
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with pytest.raises(ValueError, match="built for n_chans=4 and got 5 channels"):
        model(torch.randn(1, 5, 256))
    assert calls == []
    with torch.no_grad():
        feats = model(torch.randn(1, 5, 256), return_features=True)
    assert torch.isfinite(feats["features"]).all()


@PARAMS
def test_eval_is_deterministic_and_batch_independent(
    model_name, required_params, signal_params
):
    torch.manual_seed(0)
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.no_grad():
        a = model(x)
        b = model(x)
        one = model(x[:1])
    leaves_a = list(_tensor_leaves(a))
    leaves_b = list(_tensor_leaves(b))
    leaves_one = list(_tensor_leaves(one))
    assert len(leaves_a) == len(leaves_b) == len(leaves_one)
    for leaf_a, leaf_b in zip(leaves_a, leaves_b):
        assert torch.equal(leaf_a, leaf_b)
    for leaf_a, leaf_one in zip(leaves_a, leaves_one):
        if leaf_a.ndim >= 1 and leaf_a.shape[0] == x.shape[0]:
            torch.testing.assert_close(leaf_a[:1], leaf_one, atol=5e-5, rtol=1e-5)


def test_summary_csv_matches_generator():
    from braindecode.models._summary import check_summary_csv

    diffs = check_summary_csv()
    assert diffs == [], "\n".join(diffs)
    for name in models_dict:
        expr = _summary_row(name)["get_#Parameters"]
        assert expr.startswith(f"{name}("), f"{name}: {expr!r}"
        ast.parse(expr, mode="eval")


def test_summary_csv_check_and_incremental_files_update(tmp_path):
    from braindecode.models._summary import check_summary_csv, main

    src_csv = Path(_models_util.__file__).resolve().with_name("summary.csv")
    csv_copy = tmp_path / "summary.csv"
    shutil.copyfile(src_csv, csv_copy)
    assert check_summary_csv(csv_copy) == []

    text = csv_copy.read_text(encoding="utf-8")
    mutated_lines = []
    for line in text.splitlines():
        if line.startswith("EEGNet,"):
            line = line.replace(",2146,", ",999999,", 1)
        elif line.startswith("ShallowFBCSPNet,"):
            line = line.replace(",dense,", ",fixed,", 1)
        mutated_lines.append(line)
    csv_copy.write_text("\n".join(mutated_lines) + "\n", encoding="utf-8")

    diffs = check_summary_csv(csv_copy)
    assert any("EEGNet" in d for d in diffs)
    assert any("ShallowFBCSPNet" in d for d in diffs)
    assert main(["--check", "--csv", str(csv_copy)]) == 1

    t0 = time.perf_counter()
    rc = main(["--files", "braindecode/models/eegnet.py", "--csv", str(csv_copy)])
    elapsed = time.perf_counter() - t0
    assert rc == 0
    assert elapsed < 0.5, f"Incremental --files update took {elapsed:.3f}s"

    remaining = check_summary_csv(csv_copy)
    assert not any("EEGNet" in d for d in remaining)
    assert any("ShallowFBCSPNet" in d for d in remaining)

    assert (
        main(
            [
                "--files",
                "braindecode/models/shallow_fbcsp.py",
                "--csv",
                str(csv_copy),
            ]
        )
        == 0
    )
    assert check_summary_csv(csv_copy) == []
    assert main(["--check", "--csv", str(csv_copy)]) == 0

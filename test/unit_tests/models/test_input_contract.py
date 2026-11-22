"""Declared input axes, minimum window length, length-grid snapping and eval
determinism across every registered model."""

import ast
import copy
import inspect
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from braindecode.datasets import pad_channels_collate
from braindecode.models import (
    CORTEG,
    LUNA,
    MAPA,
    AttentionBaseNet,
    BrainModule,
    BrainOmni,
    CTNet,
    Deep4Net,
    EEGConformer,
    EEGNet,
    EEGSimpleConv,
    FBCNet,
    FBLightConvNet,
    FBMSNet,
    IFNet,
    MVPFormer,
    NeuroRVQ,
    SensingDynamics,
    ShallowFBCSPNet,
)
from braindecode.models import base as _models_base
from braindecode.models import util as _models_util
from braindecode.models.util import (
    _draw_chs_info,
    models_dict,
    models_mandatory_parameters,
)
from braindecode.modules import ChannelLayer
from braindecode.modules.channels import STRATEGIES
from braindecode.models.util import models_dict, models_mandatory_parameters
from braindecode.modules.util import dense_convert
from test.unit_tests.models._fixtures import (
    _TOKENIZERS,
    PARAMS,
    _build_case,
    _get_signal_params,
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
    # Average pooling is dilated too, so both pool modes need the same length.
    for pool_mode, expected_min in (("max", 1035), ("mean", 1035)):
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


@PARAMS
def test_dense_conversion_contract(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    T = x.shape[-1]
    dense = copy.deepcopy(model)
    if _summary_row(model_name)["Cropped decoding"] == "no":
        with pytest.raises(ValueError, match="cannot give dense predictions"):
            dense.to_dense_prediction_model()
        return
    dense.to_dense_prediction_model()
    long = torch.randn(2, x.shape[1], 3 * T)
    with torch.no_grad():
        out = dense(long)
        assert out.numel() > dense(long[..., : 2 * T]).numel()  # more steps
        window = model(long[..., :T])
        # The first dense step is the first window's output, unless zero
        # padding at the window edge (EEGNet) sees samples the window did not.
        padded = any(
            m.padding not in (0, (0,), (0, 0), "valid")
            for m in model.modules()
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d))
            or "Pool" in type(m).__name__
        )
        if window.ndim == 2 and not padded:
            torch.testing.assert_close(out[..., 0], window, atol=1e-5, rtol=1e-4)


def test_dense_conversion_refuses_a_second_call():
    net = EEGNet(n_chans=3, n_outputs=2, n_times=1000)
    net.to_dense_prediction_model()
    with pytest.raises(ValueError, match="it is already converted"):
        net.to_dense_prediction_model()


def test_dense_conversion_multiplies_existing_dilation():
    net = torch.nn.Sequential(
        torch.nn.Conv1d(3, 4, 3, stride=2),
        torch.nn.Conv1d(4, 4, 3, dilation=2),
        torch.nn.Conv1d(4, 2, 3, stride=3, dilation=2),
    )
    dense_convert(net, axis=(2, 3))
    assert [m.stride for m in net] == [(1,), (1,), (1,)]
    assert [m.dilation for m in net] == [(1,), (4,), (4,)]


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


def test_summary_csv_fills_empty_sfreq_from_constructor_default(tmp_path):
    from braindecode.models._summary import generate_summary_rows

    src_csv = Path(_models_util.__file__).resolve().with_name("summary.csv")
    csv_copy = tmp_path / "summary.csv"
    lines = src_csv.read_text(encoding="utf-8").splitlines()
    # Blank the hand-written 200 Hz of TFMTokenizer (constructor sfreq=200).
    lines = [
        line.replace(",200,", ",,", 1) if line.startswith("TFMTokenizer,") else line
        for line in lines
    ]
    csv_copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (row,) = generate_summary_rows(["TFMTokenizer"], csv_path=csv_copy)
    assert row["Sampling Frequency (Hz)"] == "200"


@pytest.mark.parametrize("cls", [FBCNet, FBMSNet, IFNet])
def test_filterbank_stride_models_defer_snap_until_after_filtering(cls):
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(n_chans=3, n_outputs=2, n_times=1002, sfreq=250.0).eval()
    x = torch.randn(2, 3, 1002)
    with torch.no_grad():
        eager = model(x)
        direct = model.forward(x)
        scripted = torch.jit.script(model)(x)
    assert torch.equal(eager, direct)
    assert torch.equal(eager, scripted)

    x_scaled = 100.0 * x
    with torch.no_grad():
        pad_scaled = model(x_scaled)
    model.on_non_divisible = "crop"
    with torch.no_grad():
        crop_eager = model(x_scaled)
        crop_direct = model.forward(x_scaled)
        crop_scripted = torch.jit.script(model)(x_scaled)
    assert torch.equal(crop_eager, crop_direct)
    assert torch.equal(crop_eager, crop_scripted)
    assert not torch.equal(crop_eager, pad_scaled)


def test_fblightconvnet_fixed_time_axis_and_non_divisible_build():
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FBLightConvNet(
            n_chans=3, n_outputs=2, n_times=1010, sfreq=250.0, win_len=250
        ).eval()
    assert model._time_axis == "fixed"
    x1010 = torch.randn(2, 3, 1010)
    with torch.no_grad():
        eager = model(x1010)
        direct = model.forward(x1010)
        scripted = torch.jit.script(model)(x1010)
    assert torch.equal(eager, direct)
    assert torch.equal(eager, scripted)
    with pytest.raises(
        ValueError, match=r"built for n_times=1010 and got 2020 samples"
    ):
        model(torch.randn(2, 3, 2 * 1010))


def test_eegsimpleconv_short_input_emits_no_snap_warning():
    model = EEGSimpleConv(n_chans=3, n_outputs=2, sfreq=250.0).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = model(torch.randn(2, 3, 1))
    assert out.shape == (2, 2)
    assert not [w for w in caught if "snapped" in str(w.message)]


_HEAD_POOLABLE_FIXED_MODELS = (
    "AttentionBaseNet",
    "CBraMod",
    "CSBrain",
    "CTNet",
    "CodeBrain",
    "DIVER1",
    "EEGConformer",
    "EEGITNet",
    "EEGInceptionERP",
    "EEGNeX",
    "EEGPT",
    "Guetschel2026",
    "JungMAE",
    "REVE",
    "SCCNet",
    "SignalJEPA_Contextual",
    "SignalJEPA_PostLocal",
    "SignalJEPA_PreLocal",
    "SincShallowNet",
    "SleepStagerBlanco2020",
    "SleepStagerChambon2018",
    "SyncNet",
    "TMSANet",
)

_INNER_FIXED_MODELS = (
    "AttnSleep",
    "BaRISTA",
    "Brant",
    "DGCNN",
    "DeepSleepNet",
    "DuIN",
    "EEGMiner",
    "EEGSym",
    "EpiNT",
    "FBLightConvNet",
    "FNIRS4D",
    "FNIRSNet",
    "FNIRSPreT",
    "FNIRST",
    "GeoNIRS",
    "MEDFormer",
    "MSVTNet",
    "NICE",
    "NeuroRVQ",
    "PBT",
    "PopulationTransformer",
    "SSTDPN",
    "TSception",
    "ZUNA",
)

STRATEGY_MODELS = tuple(
    name
    for name, cls in sorted(models_dict.items())
    if "channel_strategy" in inspect.signature(cls.__init__).parameters
    and name not in {"SleepFM", "SleepFMStager"}
)


def _is_eeg_montage_model(model_name: str) -> bool:
    modalities = {
        m.strip() for m in _summary_row(model_name)["Modality"].split(",") if m.strip()
    }
    return bool({"EEG", "iEEG", "ECoG"} & modalities) and model_name not in {
        "SleepFM",
        "SleepFMStager",
    }


def _build_with_montage(
    model_name: str, required_params: list, signal_params
) -> tuple[torch.nn.Module, torch.Tensor, list[dict]]:
    sp = _get_signal_params(signal_params)
    kw = _get_signal_params(signal_params, required_params)
    kw.pop("n_chans", None)
    kw["chs_info"] = sp["chs_info"]
    model = models_dict[model_name](**kw).eval()
    x = torch.randn(2, len(sp["chs_info"]), sp["n_times"])
    _materialize(model, x)
    return model, x, sp["chs_info"]


@PARAMS
def test_adapt_input_maps_a_permuted_montage(
    model_name, required_params, signal_params
):
    model, x, chs = _build_with_montage(model_name, required_params, signal_params)
    C = x.shape[1]
    perm = list(reversed(range(C)))
    perm_chs = [chs[i] for i in perm]
    if not _is_eeg_montage_model(model_name):
        with pytest.raises(
            ValueError,
            match=r"not an EEG montage|channel layer maps electrodes|montage adaptation",
        ):
            copy.deepcopy(model).adapt_input(chs_info=perm_chs)
        return
    adapted = copy.deepcopy(model).adapt_input(chs_info=perm_chs)
    with torch.no_grad():
        expected = list(_tensor_leaves(model(x)))
        actual = list(_tensor_leaves(adapted(x[:, perm])))
    assert len(actual) == len(expected)
    for got_leaf, exp_leaf in zip(actual, expected):
        torch.testing.assert_close(got_leaf, exp_leaf)


def test_adapt_input_corteg_dual_stream_and_modality_refusals():
    sp = _get_signal_params(None)
    chs = sp["chs_info"]
    perm = list(reversed(range(len(chs))))
    perm_chs = [chs[i] for i in perm]

    corteg = CORTEG(
        chs_info=chs,
        n_outputs=2,
        n_times=64,
        patch_size=16,
        hi_patch_size=25,
        embed_dim=32,
        depth=1,
        num_heads=2,
        n_merge_blocks=1,
    ).eval()
    x = torch.randn(2, len(chs), 64)
    x_hi = torch.randn(2, len(chs), 100)
    adapted_corteg = copy.deepcopy(corteg).adapt_input(chs_info=perm_chs)
    with torch.no_grad():
        torch.testing.assert_close(
            adapted_corteg(x[:, perm], x_hi=x_hi[:, perm]),
            corteg(x, x_hi=x_hi),
        )

    meg_chs = [
        {"ch_name": f"MEG{i}", "kind": 1, "coil_type": 3022, "loc": ch["loc"]}
        for i, ch in enumerate(chs)
    ]
    with pytest.raises(ValueError, match="electrodes"):
        BrainOmni(
            chs_info=meg_chs,
            n_outputs=2,
            n_times=512,
            sfreq=256.0,
            emb_dim=16,
            lm_dim=16,
            n_neuro=4,
            n_filters=4,
            ratios=(4, 2),
            depth=2,
            num_heads=2,
            tokenizer_num_heads=2,
            codebook_dim=16,
            codebook_size=16,
            num_quantizers=2,
        ).adapt_input(chs_info=meg_chs)
    with pytest.raises(ValueError, match="electrodes"):
        BrainModule(
            chs_info=meg_chs, n_outputs=2, n_times=100, sfreq=250.0
        ).adapt_input(chs_info=meg_chs)
    with pytest.raises(ValueError, match="modality"):
        NeuroRVQ(
            chs_info=[{"ch_name": f"c{i + 1}"} for i in range(3)],
            modality="emg",
            n_outputs=2,
            n_times=200,
            sfreq=1000.0,
            depth=1,
            num_heads=2,
            out_chans=4,
        ).adapt_input(chs_info=chs)
    with pytest.raises(ValueError, match="spectrograms|normalization"):
        MAPA(
            chs_info=[{"ch_name": "LA1", "kind": "seeg"}],
            n_outputs=2,
            n_times=32,
            sfreq=32.0,
            normalization="session",
            d_model=64,
        ).adapt_input(chs_info=[{"ch_name": "LA1", "kind": "seeg"}])


def test_adapt_input_mixed_montage_batch():
    chs = _draw_chs_info(6, np.random.default_rng(6))
    C, T = len(chs), 1000
    model = EEGNet(chs_info=chs, n_outputs=2, n_times=T, sfreq=250.0).eval()
    adapted = copy.deepcopy(model).adapt_input(channel_strategy="spline")
    batch = [
        (
            torch.randn(C, T).numpy(),
            0,
            [0, T, 0],
            np.stack([c["loc"][:3] for c in chs]),
        ),
        (
            torch.randn(C - 1, T).numpy(),
            1,
            [0, T, 0],
            np.stack([c["loc"][:3] for c in chs[: C - 1]]),
        ),
    ]
    X, _, _, pos, ch_mask = pad_channels_collate(batch)
    with torch.no_grad():
        out = adapted(X, pos=pos, ch_mask=ch_mask)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


@PARAMS
def test_adapt_input_resamples(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    sfreq = model._signal_param("sfreq")
    if sfreq is None or float(sfreq) != round(float(sfreq)):
        pytest.skip("requires integer fixture sfreq")
    C, T = x.shape[1:]
    with pytest.warns(UserWarning, match="resampled inside the model"):
        adapted = copy.deepcopy(model).adapt_input(sfreq=2 * float(sfreq))
    with torch.no_grad():
        ref = model(x)
        out = adapted(torch.randn(2, C, 2 * T))
    assert _shapes(out) == _shapes(ref)
    if model._effective_time_axis() == "fixed" and not model._bypass_time_check(
        x, (x,), {}
    ):
        bad_len = 2 * (T + max(20, T // 50 + 2))
        with pytest.raises(ValueError, match="after resampling from"):
            adapted(torch.randn(2, C, bad_len))


def test_adapt_input_n_times_is_identity_at_build_length():
    base = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    x0 = torch.randn(2, 3, 1000)
    adapted_same = copy.deepcopy(base).adapt_input(n_times=1000)
    with torch.no_grad():
        torch.testing.assert_close(adapted_same(x0), base(x0), atol=0, rtol=0)
        longer = copy.deepcopy(base).adapt_input(n_times=1500)
        assert longer(torch.randn(2, 3, 1500)).shape == (2, 2)


@pytest.mark.parametrize("model_name", _HEAD_POOLABLE_FIXED_MODELS)
def test_adapt_input_n_times_head_poolable_models(model_name):
    req, raw_sp = next(
        (r, s) for n, r, s in models_mandatory_parameters if n == model_name
    )
    sp = _get_signal_params(raw_sp)
    kw = _get_signal_params(raw_sp, req)
    if "n_times" not in kw:
        kw["n_times"] = sp["n_times"]
    model = models_dict[model_name](**kw).eval()
    C, T = len(sp["chs_info"]), sp["n_times"]
    x0 = torch.randn(2, C, T)
    _materialize(model, x0)
    adapted_same = copy.deepcopy(model).adapt_input(n_times=T)
    step = getattr(model, "_n_times_step", 1)
    longer_t = T + T // 2
    if getattr(model, "on_non_divisible", "pad") == "error" and step > 1:
        longer_t = T + ((T // 2 + step - 1) // step) * step
    adapted_long = copy.deepcopy(model).adapt_input(n_times=longer_t)
    with torch.no_grad():
        torch.testing.assert_close(adapted_same(x0), model(x0), atol=0, rtol=0)
        out_long = adapted_long(torch.randn(2, C, longer_t))
        assert out_long.shape == model(x0).shape
        assert torch.isfinite(out_long).all()


@pytest.mark.parametrize("model_name", _INNER_FIXED_MODELS)
def test_adapt_input_n_times_refuses_inner_fixed_models(model_name):
    req, raw_sp = next(
        (r, s) for n, r, s in models_mandatory_parameters if n == model_name
    )
    model, x = _build_case(model_name, req, raw_sp)
    _materialize(model, x)
    with pytest.raises(ValueError, match=r"fixes .+ to n_times|adapt_input\(n_times="):
        copy.deepcopy(model).adapt_input(n_times=x.shape[-1] + 100)


def test_adapt_input_n_times_special_bounds_and_free_models():
    for mode in ("fca", "ge"):
        attn_model = AttentionBaseNet(
            n_chans=3, n_outputs=2, n_times=1000, attention_mode=mode
        ).eval()
        with pytest.raises(ValueError, match="attention_mode"):
            attn_model.adapt_input(n_times=1500)

    ctnet = CTNet(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.raises(ValueError, match="position"):
        ctnet.adapt_input(n_times=7000)

    for cls in (EEGNet, Deep4Net, ShallowFBCSPNet):
        m = cls(n_chans=3, n_outputs=2, n_times=1000).eval()
        adapted = copy.deepcopy(m).adapt_input(n_times=1500)
        assert adapted._adapted_n_times == 1500
        with torch.no_grad():
            assert torch.isfinite(adapted(torch.randn(2, 3, 1500))).all()


def test_adapt_input_keeps_state_dict_keys():
    chs = _get_signal_params(None)["chs_info"]
    model = EEGNet(chs_info=chs, n_outputs=2, n_times=1000, sfreq=250.0).eval()
    keys = set(model.state_dict())
    for strategy in _models_base._ADAPT_STRATEGIES:
        adapted = copy.deepcopy(model).adapt_input(
            chs_info=chs, channel_strategy=strategy
        )
        assert set(adapted.state_dict()) == keys
    for strategy in set(STRATEGIES) - set(_models_base._ADAPT_STRATEGIES) - {"native"}:
        with pytest.raises(ValueError, match="adds learned or fitted state"):
            copy.deepcopy(model).adapt_input(chs_info=chs, channel_strategy=strategy)
    with pytest.raises(ValueError, match="adds learned or fitted state"):
        copy.deepcopy(model).adapt_input(
            chs_info=chs,
            channel_strategy="source",
            channel_strategy_kwargs={"trainable": True},
        )


def test_adapt_input_twice_and_reload(tmp_path):
    sp = _get_signal_params(None)
    chs, C, T = sp["chs_info"], len(sp["chs_info"]), sp["n_times"]
    chs_a = list(reversed(chs))
    chs_b = [chs[(i + 1) % C] for i in range(C)]
    model = EEGNet(chs_info=chs, n_outputs=2, n_times=T, sfreq=250.0).eval()
    with pytest.warns(UserWarning, match="resampled inside the model"):
        adapted = copy.deepcopy(model).adapt_input(chs_info=chs_a)
        adapted.adapt_input(chs_info=chs_b, sfreq=2 * model.sfreq)
    assert adapted.channel_layer.chs_info == chs_b
    plain = copy.deepcopy(model)
    plain.load_state_dict(adapted.state_dict(), strict=True)
    assert "input_adaptation" not in model.get_config()
    cfg = adapted.get_config()
    assert "input_adaptation" in cfg
    assert [c["ch_name"] for c in cfg["chs_info"]] == [c["ch_name"] for c in chs]
    assert [c["ch_name"] for c in cfg["input_adaptation"]["chs_info"]] == [
        c["ch_name"] for c in chs_b
    ]
    with pytest.warns(UserWarning, match="resampled inside the model"):
        rebuilt = type(model).from_config(cfg).eval()
    rebuilt.load_state_dict(adapted.state_dict(), strict=True)
    x2 = torch.randn(2, C, 2 * T)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x2), adapted(x2))
    if _models_base.HAS_HF_HUB:
        adapted.save_pretrained(tmp_path)
        with pytest.warns(UserWarning, match="resampled inside the model"):
            hf_reloaded = type(model).from_pretrained(tmp_path).eval()
        with torch.no_grad():
            torch.testing.assert_close(hf_reloaded(x2), adapted(x2))


def test_adapt_input_bfloat16():
    sp = _get_signal_params(None)
    chs, C, T = sp["chs_info"], len(sp["chs_info"]), sp["n_times"]
    chs_b = [chs[(i + 1) % C] for i in range(C)]
    model = EEGNet(chs_info=chs, n_outputs=2, n_times=T, sfreq=250.0).eval()
    with pytest.warns(UserWarning, match="resampled inside the model"):
        adapted = (
            copy.deepcopy(model)
            .adapt_input(chs_info=chs_b, sfreq=2 * model.sfreq)
            .to(torch.bfloat16)
        )
    with torch.no_grad():
        out = adapted(torch.randn(2, C, 2 * T, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


def test_adapt_input_needs_the_build_montage():
    chs = _get_signal_params(None)["chs_info"]
    with pytest.raises(ValueError, match="needs the montage the model was built with"):
        EEGNet(n_chans=3, n_outputs=2, n_times=1000).adapt_input(chs_info=chs)


@pytest.mark.parametrize("model_name", STRATEGY_MODELS)
def test_identity_mapping_matches_native(model_name):
    req, raw_sp = next(
        (r, s) for n, r, s in models_mandatory_parameters if n == model_name
    )
    sp = _get_signal_params(raw_sp)
    kw = _get_signal_params(raw_sp, req)
    cls = models_dict[model_name]
    target_spec = (
        cls._channel_target if cls._channel_target is not None else sp["chs_info"]
    )
    target_dicts = [
        c if isinstance(c, dict) else {"ch_name": c, "kind": "eeg", "loc": np.zeros(12)}
        for c in target_spec
    ]
    target = ChannelLayer(target_dicts, "exact").target
    kw.pop("n_chans", None)
    kw["chs_info"] = target
    torch.manual_seed(0)
    native = cls(**kw).eval()
    x = torch.randn(2, len(target), sp["n_times"])
    _materialize(native, x)
    mapped = cls(**kw, channel_strategy="nearest").eval()
    _materialize(mapped, x)
    mapped.load_state_dict(native.state_dict(), strict=True)
    with torch.no_grad():
        native_leaves = list(_tensor_leaves(native(x)))
        mapped_leaves = list(_tensor_leaves(mapped(x)))
    assert len(mapped_leaves) == len(native_leaves)
    for got_leaf, ref_leaf in zip(mapped_leaves, native_leaves):
        torch.testing.assert_close(got_leaf, ref_leaf)


# Free or dense models whose export pins the time axis: model name to the first
# line of its export error (T is the input length). Shrink-only: fix the forward
# and remove the entry. With torch 2.15, export specializes the output length of
# nn.MaxPool1d alone: CareSleepNet and SeizureTransformer specialize at their
# MaxPool1d call; FNIRSCNN1D and USleep pool with it too.
_EXPORT_STATIC_TIME: dict[str, str] = {
    "ATCNet": "unprovable guard (2 % ((-8) + 2*(T // 56))) != 0",
    "AXON": "unprovable guard T != (180 + 180*(((-200) + T) // 180))",
    "BBL": "unprovable guard on torch.sym_max(0, 3 - ((T - 1) // 2)) padding",
    "BENDR": "unprovable guard (2 % (2 + 2*(((-1) + T) // 96))) != 0",
    "BIOT": "unprovable guard (2 % (2 + 2*(((-250) + T) // 100))) != 0",
    "BrainBERT": "unprovable guard (T % 50) != 0",
    "BrainOmni": "AssertionError: Cannot create Dim with inconsistent min=-2, max=-10",
    "BrainTokenizer": "unprovable guard (T // 512) == 1",
    "CareSleepNet": "AssertionError: Guard failed: 1 + (-1 + T // 20) // 12 == 25",
    "ContraWR": "unprovable guard 1000 <= T and T <= 1279",
    "DANCE": "time specialized to 200",
    "DeepSOZ": "AssertionError: Guard failed: T // 256 == 20",
    # avg_pool2d over the whole time axis reads its kernel from the shape;
    # adaptive pooling exports but changes outputs by 1e-7.
    "EEGInceptionMI": "time specialized to 1000",
    "EEGSimpleConv": "AssertionError: Guard failed: 8 * T / 25 == 320.32",
    "EMG2QwertyNet": "unprovable guard (((-3) + (T // 16)) % ((-18) + 6*(T // 16))) != 0",
    "FBCNet": "AssertionError: Guard failed: (1 + (331 + T) // 2) // 2 % 2 != 0",
    "FBMSNet": "AssertionError: Guard failed: (1 + (331 + T) // 2) // 2 % 2 != 0",
    "FNIRSCNN1D": "time specialized to 60",
    "IFNet": "AssertionError: Guard failed: (331 + T) // 2 % 5 == 0",
    "MAPA": "GuardOnDataDependentSymNode: Could not extract specialized integer u2",
    "MEGGPT": "time (100*k) specialized to 1000",
    "MEGXL": "unprovable guard on torch.sym_max(0, 3 - T) padding",
    "MetaNeuromotorHand": "unprovable guard (33 % (33*(((-11) + (((-64) + T) // 10)) // 4))) != 0",
    "MSCFormer": "unprovable guard (1 % (T // 44)) != 0",
    "MtReNIRS": "time (100*k) specialized to 100",
    "MVPFormer": "time (2560*k) specialized to 2560",
    "NeuroPose": "unprovable guard (2 % (640*(T // 320))) != 0",
    "NeuroRVQTokenizer": "GuardOnDataDependentSymNode: Could not guard on zuf0",
    "Pratibha2": "unprovable guard T != (180 + 180*(((-200) + T) // 180))",
    "SAMBA": "time specialized to 128",
    "SeizureTransformer": "time specialized to 2048",
    "SignalJEPA": "unprovable guard (3 % (3 + 3*(... (T // 8) ...))) != 0",
    "SleepFM": "time (640*k) specialized to 640",
    "SleepFMStager": "time (640*k) specialized to 640",
    "SPARCNet": "NotImplementedError: Eq(63*(((x/4)//128)) - 63, 0) cannot be solved",
    "TCFormer": "unprovable guard (2 % (2*(T // 56))) != 0",
    "TFMTokenizer": "unprovable guard (6 % ((-6) + 6*(T // 100))) != 0",
    # Also nn.MaxPool1d; with a branch-free odd-length pad instead of the
    # `if x.shape[-1] % 2` branch, export of the 12 levels ran past 15 minutes.
    "USleep": "time specialized to 2000",
    "VEMG2Pose": "unprovable guard (2 % ((-14) + 2*(((-18) + (((-11) + T) // 10)) // 4))) != 0",
}


@PARAMS
def test_export_with_dynamic_time(model_name, required_params, signal_params, request):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    if model._time_axis == "fixed":
        pytest.skip("fixed length")
    if model_name in _EXPORT_STATIC_TIME:
        request.applymarker(
            pytest.mark.xfail(reason=_EXPORT_STATIC_TIME[model_name], strict=True)
        )
    C, T = x.shape[1:]
    step, offset = model._n_times_step, model._n_times_offset
    # A dense model returns one step at its build length and a time axis for
    # longer inputs, so its range starts where the output has several steps.
    low = 2 * T if model._time_axis == "dense" else T
    # Lengths on the model's grid, offset + step * k, from low to 4 T.
    k_min = -(-(low - offset) // step)
    k_max = max((4 * T - offset) // step, k_min + 1)
    k = torch.export.Dim("k", min=k_min, max=k_max)
    time = k if step == 1 else step * k + offset
    example = torch.randn(2, C, offset + step * k_min)
    program = torch.export.export(
        model, (example,), dynamic_shapes=({2: time},), strict=False
    )
    longer = torch.randn(2, C, offset + step * ((k_min + k_max + 1) // 2))
    with torch.no_grad():
        torch.testing.assert_close(program.module()(longer), model(longer))

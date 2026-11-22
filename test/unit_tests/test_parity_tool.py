"""Tests for the model parity recorder/checker and shared model fixture helpers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from braindecode.models import util as _models_util
from braindecode.models.util import models_mandatory_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = REPO_ROOT / "benchmarks" / "parity.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("parity", PARITY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_round_trip(tmp_path: Path) -> None:
    parity = _load_tool()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        parity.check(empty_dir)

    names = ["ShallowFBCSPNet", "EEGNet"]
    assert parity.record(tmp_path, names) == names
    for name in names:
        pt_path = tmp_path / f"{name}.pt"
        assert pt_path.is_file()
        saved = torch.load(pt_path, map_location="cpu", weights_only=True)
        assert set(saved.keys()) == {"keys", "state", "x", "out"}

    assert parity.check(tmp_path, names) == {
        "ShallowFBCSPNet": "same",
        "EEGNet": "same",
    }
    assert parity.check(tmp_path) == {
        "EEGNet": "same",
        "ShallowFBCSPNet": "same",
    }


def test_parity_reports_changed_output_and_tolerances(tmp_path: Path) -> None:
    parity = _load_tool()
    parity.record(tmp_path, ["EEGNet"])
    pt_path = tmp_path / "EEGNet.pt"
    original = torch.load(pt_path, map_location="cpu", weights_only=True)

    saved = {**original, "out": [original["out"][0] + 5e-4, *original["out"][1:]]}
    torch.save(saved, pt_path)

    status = parity.check(tmp_path, ["EEGNet"])["EEGNet"]
    assert status.startswith("diff=")
    assert float(status.removeprefix("diff=")) == pytest.approx(5e-4, rel=1e-2)
    assert parity.check(tmp_path, ["EEGNet"], atol=1e-4)["EEGNet"].startswith("diff=")
    assert parity.check(tmp_path, ["EEGNet"], atol=1e-3) == {"EEGNet": "same"}

    scaled = {**original, "out": [original["out"][0] * 1.05, *original["out"][1:]]}
    torch.save(scaled, pt_path)
    assert parity.check(tmp_path, ["EEGNet"], rtol=0.01)["EEGNet"].startswith("diff=")
    assert parity.check(tmp_path, ["EEGNet"], rtol=0.1) == {"EEGNet": "same"}

    nan_out = original["out"][0].clone()
    nan_out.view(-1)[0] = float("nan")
    torch.save({**original, "out": [nan_out, *original["out"][1:]]}, pt_path)
    assert parity.check(tmp_path, ["EEGNet"]) == {"EEGNet": "diff=nan"}
    assert parity.check(tmp_path, ["EEGNet"], atol=1.0, rtol=1.0) == {
        "EEGNet": "diff=nan"
    }


def test_parity_loads_saved_weights_not_fresh_seed(tmp_path: Path) -> None:
    from test.unit_tests.models._fixtures import (
        _build_case,
        _materialize,
        _tensor_leaves,
    )

    parity = _load_tool()
    parity.record(tmp_path, ["EEGNet"])
    pt_path = tmp_path / "EEGNet.pt"
    saved = torch.load(pt_path, map_location="cpu", weights_only=True)

    torch.manual_seed(99)
    alt_model, _ = _build_case("EEGNet", ["n_chans", "n_outputs", "n_times"], None)
    _materialize(alt_model, saved["x"])
    saved["state"] = {k: v.detach().clone() for k, v in alt_model.state_dict().items()}
    with torch.no_grad():
        saved["out"] = [
            leaf.detach().clone() for leaf in _tensor_leaves(alt_model(saved["x"]))
        ]
    torch.save(saved, pt_path)

    assert parity.check(tmp_path, ["EEGNet"]) == {"EEGNet": "same"}


def test_parity_state_key_mismatch(tmp_path: Path) -> None:
    parity = _load_tool()
    parity.record(tmp_path, ["EEGNet"])
    pt_path = tmp_path / "EEGNet.pt"
    saved = torch.load(pt_path, map_location="cpu", weights_only=True)
    saved["keys"] = [*saved["keys"], "extra_head.weight"]
    torch.save(saved, pt_path)

    assert parity.check(tmp_path, ["EEGNet"]) == {"EEGNet": "keys"}


def test_parity_shape_mismatch(tmp_path: Path) -> None:
    parity = _load_tool()
    parity.record(tmp_path, ["EEGNet"])
    pt_path = tmp_path / "EEGNet.pt"
    original = torch.load(pt_path, map_location="cpu", weights_only=True)

    sliced = {**original, "out": [original["out"][0][:, :1], *original["out"][1:]]}
    torch.save(sliced, pt_path)
    assert parity.check(tmp_path, ["EEGNet"]) == {"EEGNet": "shape"}

    extra_leaf = {**original, "out": [*original["out"], original["out"][0]]}
    torch.save(extra_leaf, pt_path)
    assert parity.check(tmp_path, ["EEGNet"]) == {"EEGNet": "shape"}


def test_parity_cli_exit_codes(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    rec = subprocess.run(
        [
            sys.executable,
            str(PARITY_SCRIPT),
            "record",
            "--out",
            str(tmp_path),
            "--models",
            "EEGNet",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rec.returncode == 0, rec.stderr

    ok = subprocess.run(
        [
            sys.executable,
            str(PARITY_SCRIPT),
            "check",
            "--out",
            str(tmp_path),
            "--models",
            "EEGNet",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert "EEGNet: same" in ok.stdout

    pt_path = tmp_path / "EEGNet.pt"
    saved = torch.load(pt_path, map_location="cpu", weights_only=True)
    saved["out"] = [saved["out"][0] + 1.0, *saved["out"][1:]]
    torch.save(saved, pt_path)

    bad = subprocess.run(
        [
            sys.executable,
            str(PARITY_SCRIPT),
            "check",
            "--out",
            str(tmp_path),
            "--models",
            "EEGNet",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "EEGNet: diff=" in bad.stdout

    tol = subprocess.run(
        [
            sys.executable,
            str(PARITY_SCRIPT),
            "check",
            "--out",
            str(tmp_path),
            "--models",
            "EEGNet",
            "--atol",
            "2.0",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tol.returncode == 0, tol.stderr


def test_shared_fixture_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test.unit_tests.models._fixtures import (
        _TOKENIZERS,
        PARAMS,
        _build_case,
        _materialize,
        _shapes,
        _summary_row,
        _tensor_leaves,
    )

    assert _TOKENIZERS == ("BrainTokenizer", "NeuroRVQTokenizer", "TFMTokenizer")
    assert PARAMS.mark.name == "parametrize"
    assert PARAMS.mark.args[0] == "model_name,required_params,signal_params"
    assert list(PARAMS.mark.args[1]) == list(models_mandatory_parameters)
    assert PARAMS.mark.kwargs.get("ids") == [m[0] for m in models_mandatory_parameters]

    nested = {"logits": torch.zeros(2, 4), "aux": (torch.ones(2, 3, 5), "ignored")}
    assert _shapes(nested) == [(2, 4), (2, 3, 5)]

    row = _summary_row("ATCNet")
    assert row["Model"] == "ATCNet"
    assert "Categorization" in row
    assert "Modality" in row

    fake_util = tmp_path / "util.py"
    fake_util.write_text("", encoding="utf-8")
    fake_csv = tmp_path / "summary.csv"
    fake_csv.write_text(
        "\ufeffModel, Application, Modality\nATCNet, General, EEG\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_models_util, "__file__", str(fake_util))
    assert _summary_row("ATCNet") == {
        "Model": "ATCNet",
        "Application": "General",
        "Modality": "EEG",
    }
    fake_csv.write_text(
        "\ufeffModel, Application, Modality\nATCNet, Motor Imagery, MEG\n",
        encoding="utf-8",
    )
    stat = fake_csv.stat()
    os.utime(fake_csv, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert _summary_row("ATCNet")["Application"] == "Motor Imagery"
    assert _summary_row("ATCNet")["Modality"] == "MEG"
    monkeypatch.undo()

    model, x = _build_case("EEGNet", ["n_chans", "n_outputs", "n_times"], None)
    assert not model.training
    assert x.shape == (2, 3, 1000)
    _materialize(model, x)
    with torch.no_grad():
        leaves = list(_tensor_leaves(model(x)))
    assert len(leaves) == 1
    assert leaves[0].shape == (2, 2)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    lazy_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import test.unit_tests.models._fixtures; "
                "from braindecode.models import util; "
                "assert not util._models_dict; "
                "assert 'braindecode.models.eegnet' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lazy_proc.returncode == 0, lazy_proc.stderr

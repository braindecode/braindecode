"""Tests for the model parity recorder and checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

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


def test_parity_cli_exit_codes(tmp_path: Path, capsys) -> None:
    parity = _load_tool()
    args = ["--out", str(tmp_path), "--models", "EEGNet"]
    assert parity.main(["record", *args]) == 0
    assert parity.main(["check", *args]) == 0
    assert "EEGNet: same" in capsys.readouterr().out

    pt_path = tmp_path / "EEGNet.pt"
    saved = torch.load(pt_path, map_location="cpu", weights_only=True)
    saved["out"] = [saved["out"][0] + 1.0, *saved["out"][1:]]
    torch.save(saved, pt_path)
    assert parity.main(["check", *args]) == 1
    assert "EEGNet: diff=" in capsys.readouterr().out
    assert parity.main(["check", *args, "--atol", "2.0"]) == 0

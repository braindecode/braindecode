"""Record and check per-model state_dict and forward-output parity.

Run from the repository root::

    python benchmarks/parity.py record --out /tmp/bd-parity-base
    python benchmarks/parity.py check --out /tmp/bd-parity-base
    python benchmarks/parity.py check --out /tmp/bd-parity-base --models EEGNet --atol 1e-5
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braindecode.models.util import models_mandatory_parameters
from test.unit_tests.models._fixtures import (
    _build_case,
    _materialize,
    _tensor_leaves,
)

_CASES = {
    name: (name, required_params, signal_params)
    for name, required_params, signal_params in models_mandatory_parameters
}


def record(out_dir: Path, names: list[str] | None = None) -> list[str]:
    """Record initial weights, synthetic input, and eval outputs per model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(_CASES) if names is None else list(names)
    for name in selected:
        torch.manual_seed(0)
        model, x = _build_case(*_CASES[name])
        _materialize(model, x)
        state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            out = [leaf.detach().clone() for leaf in _tensor_leaves(model(x))]
        payload = {
            "keys": list(state_dict),
            "state": state_dict,
            "x": x.detach().clone(),
            "out": out,
        }
        torch.save(payload, out_dir / f"{name}.pt")
        del model, x, state_dict, out, payload
        gc.collect()
    return selected


def _compare_leaves(
    got_leaves: list[torch.Tensor],
    want_leaves: list[torch.Tensor],
    atol: float,
    rtol: float,
) -> str:
    if len(got_leaves) != len(want_leaves):
        return "shape"
    for got, want in zip(got_leaves, want_leaves):
        if got.shape != want.shape:
            return "shape"

    max_diff = 0.0
    ok = True
    for got, want in zip(got_leaves, want_leaves):
        if got.dtype == torch.bool:
            diff = (got != want).float()
        elif not got.is_floating_point() and not got.is_complex():
            diff = (got.to(torch.int64) - want.to(torch.int64)).abs()
        else:
            diff = (got - want).abs()
        leaf_max = float(diff.max().item()) if diff.numel() else 0.0
        if math.isnan(leaf_max):
            max_diff = float("nan")
        elif not math.isnan(max_diff) and leaf_max > max_diff:
            max_diff = leaf_max
        if atol == 0.0 and rtol == 0.0:
            if not torch.equal(got, want):
                ok = False
        else:
            if want.dtype == torch.bool:
                want_mag = want.float().abs()
            elif not want.is_floating_point() and not want.is_complex():
                want_mag = want.to(torch.int64).abs()
            else:
                want_mag = want.abs()
            bound = atol + rtol * want_mag
            if not bool((diff <= bound).all().item()):
                ok = False
    return "same" if ok else f"diff={max_diff:g}"


def check(
    out_dir: Path,
    names: list[str] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, str]:
    """Compare current model state keys and outputs against recorded files."""
    out_dir = Path(out_dir)
    if names is None:
        selected = [
            path.stem for path in sorted(out_dir.glob("*.pt")) if path.stem in _CASES
        ]
        if not selected:
            raise FileNotFoundError(f"No recorded .pt files found in {out_dir}")
    else:
        selected = list(names)

    results: dict[str, str] = {}
    for name in selected:
        saved = torch.load(
            out_dir / f"{name}.pt", map_location="cpu", weights_only=True
        )
        torch.manual_seed(0)
        model, _ = _build_case(*_CASES[name])
        x = saved["x"]
        _materialize(model, x)
        if list(model.state_dict()) != saved["keys"]:
            results[name] = "keys"
            del model, saved, x
            gc.collect()
            continue
        model.load_state_dict(saved["state"], strict=True)
        with torch.no_grad():
            got_leaves = list(_tensor_leaves(model(x)))
        results[name] = _compare_leaves(got_leaves, saved["out"], atol=atol, rtol=rtol)
        del model, saved, x, got_leaves
        gc.collect()
    return results


def _parse_models(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "check"))
    parser.add_argument("--out", required=True, type=Path, help="Baseline directory")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of model names (default: all)",
    )
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance")
    args = parser.parse_args(argv)

    names = _parse_models(args.models)
    if args.command == "record":
        recorded = record(args.out, names=names)
        print(f"Recorded {len(recorded)} models to {args.out}")
        return 0

    results = check(args.out, names=names, atol=args.atol, rtol=args.rtol)
    for name, status in results.items():
        print(f"{name}: {status}")
    return 0 if all(status == "same" for status in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

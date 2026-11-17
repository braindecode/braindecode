# Authors: Zhibo Zhang
# License: BSD-3

"""Read-only pretrained tensor-origin and feature-scope audit (internal experimental helper).

The caller explicitly declares the feature tensor scope. No claim is made
about execution graph dependency, checkpoint parity, or scientific validity.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(name == p or (p.endswith(".") and name.startswith(p)) for p in patterns)


def _rows(model, loaded, mixed, expected_new, feature_scope):
    objects = {}
    for kind, iterator in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in iterator:
            if id(tensor) not in objects:
                objects[id(tensor)] = {
                    "name": name, "aliases": [name], "tensor": tensor, "kind": kind,
                }
            else:
                entry = objects[id(tensor)]
                if entry["kind"] != kind:
                    raise ValueError("parameter and buffer alias the same object")
                entry["aliases"].append(name)

    rows = []
    for entry in objects.values():
        tensor = entry["tensor"]
        names = entry["aliases"]
        unbuilt = isinstance(tensor, (UninitializedParameter, UninitializedBuffer))
        status = (
            "uninitialized" if unbuilt else
            "adapted" if any(n in mixed for n in names) else
            "loaded" if any(n in loaded for n in names) else "new"
        )
        rows.append({
            "name": entry["name"], "aliases": list(names),
            "kind": entry["kind"], "status": status,
            "count": None if unbuilt else int(tensor.numel()),
            "trainable": bool(tensor.requires_grad) if entry["kind"] == "parameter" else False,
            "expected_new": any(_matches(n, expected_new) for n in names),
            "feature_scoped": any(_matches(n, feature_scope) for n in names),
        })
    return sorted(rows, key=lambda r: (r["kind"], r["name"]))


def audit_model(
    model: torch.nn.Module, *,
    expected_new: Iterable[str] = (),
    feature_scope: Iterable[str] | None = None,
    include_buffers: bool = False,
) -> dict:
    """Classify provenance without changing tensors, hooks or training mode.

    expected_new identifies permitted newly initialized task components.
    It does not make these components safe for frozen feature extraction.
    feature_scope is a manually declared set of tensor names or prefixes,
    not a discovered forward execution graph.
    """
    provenance = vars(model).get("_pretrained_provenance")
    if not isinstance(provenance, dict) or "loaded" not in provenance:
        raise ValueError("no Braindecode loaded-parameter provenance; cannot audit checkpoint coverage")
    loaded = set(provenance.get("loaded") or ())
    mixed = set(provenance.get("mixed") or ())
    allowed = tuple(expected_new)
    features = None if feature_scope is None else tuple(feature_scope)
    if any(not isinstance(s, str) or not s for s in (*allowed, *(features or ()))):
        raise ValueError("expected_new and feature_scope entries must be nonempty strings")
    rows_all = _rows(model, loaded, mixed, allowed, features or ())
    rows = [r for r in rows_all if include_buffers or r["kind"] == "parameter"]
    counts = {
        k: sum(r["count"] or 0 for r in rows if r["status"] == k)
        for k in ("loaded", "adapted", "new")
    }
    total = sum(counts.values())
    changed = ("new", "adapted")
    unexpected = [
        r["name"] for r in rows if r["status"] in changed and not r["expected_new"]
    ]
    frozen_unexpected = [
        r["name"] for r in rows if r["kind"] == "parameter"
        and r["status"] in changed and not r["trainable"] and not r["expected_new"]
    ]
    feature_unloaded = None if features is None else [
        r["name"] for r in rows if r["feature_scoped"] and r["status"] != "loaded"
    ]
    feature_unloaded_frozen = None if features is None else [
        r["name"] for r in rows if r["kind"] == "parameter"
        and r["feature_scoped"] and r["status"] in changed and not r["trainable"]
    ]
    present = {name for r in rows_all for name in r["aliases"]}
    return {
        "schema_version": 2, "model_class": type(model).__name__,
        "repo_id": provenance.get("repo_id"),
        "revision": provenance.get("commit") or provenance.get("revision"),
        "counts": {**counts, "total_materialized": total},
        "loaded_fraction": None if total == 0 else counts["loaded"] / total,
        "unexpected_new_or_adapted": unexpected,
        "frozen_unexpected": frozen_unexpected,
        "stale_loaded_keys": sorted(loaded - present),
        "stale_adapted_keys": sorted(mixed - present),
        "feature_scope_status": "not_assessed" if features is None else "declared",
        "feature_unloaded": feature_unloaded,
        "feature_unloaded_frozen": feature_unloaded_frozen,
        "uninitialized": [r["name"] for r in rows if r["status"] == "uninitialized"],
        "parameters": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline synthetic provenance audit")
    parser.add_argument("--output", help="Write JSON report")
    args = parser.parse_args()
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    model._pretrained_provenance = {
        "loaded": {"0.weight", "0.bias"}, "mixed": set(), "repo_id": "synthetic/example",
    }
    content = json.dumps(
        audit_model(model, expected_new=("1.",), feature_scope=("0.",)),
        indent=2, ensure_ascii=False,
    ) + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()

"""Paired CPU Mixup component and full AugmentedDataLoader traversal benchmark.

Run from a clean checkout with --output pointing outside Git. Each pair uses
fresh serial processes in alternating order, with identical synthetic inputs.
The scalar function is extracted from the pinned baseline only for this explicit
integration benchmark; unit-test oracles do not depend on Git history. The
production default remains scalar because this candidate regressed on CPU. No GPU
qualification or general hardware crossover is inferred from these CPU cases.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import platform
import resource
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

import braindecode
from benchmarks.acceleration._common import measure_step, write_result
from benchmarks.acceleration._qualification import historical_only, require_linux_rss
from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation.transforms import Mixup

BASELINE = "0dd2980f45682129d8458ac3a83f567573b8a2fa"
EXPERIMENTAL = "e77d2587623b52de96d4ac25df9b1aaa7e63affe"
ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _historical_mixup(revision):
    try:
        source = _git("show", f"{revision}:braindecode/augmentation/functional.py")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Historical-only Mixup replay requires exact Git object {revision}. "
            "Use the retained private full-history clone, or explicitly fetch the "
            "pinned object from an authorized historical remote. Never substitute "
            "current Mixup: the rejected e77d2587 candidate is not current production."
        ) from exc
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "mixup")
    namespace = {"torch": torch}
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), revision, "exec"),
        namespace,
    )
    digest = hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest()
    return namespace["mixup"], digest


def _input(name, value):
    return {"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
            "strides": list(value.stride()), "stride_unit": "elements"}


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    else:
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _equal(left, right)


def _worker(args):
    require_linux_rss()
    if _git("status", "--porcelain"):
        raise RuntimeError("Finish edits and commit before measuring")
    assert Path(braindecode.__file__).resolve().is_relative_to(ROOT)
    torch.set_num_threads(1)
    reference, digest = _historical_mixup(BASELINE)
    candidate, candidate_digest = _historical_mixup(EXPERIMENTAL)
    function = reference if args.implementation == "reference" else candidate
    generator = torch.Generator().manual_seed(231)
    X_all = torch.randn(128, 22, 1000, generator=generator)
    y_all = torch.arange(256, dtype=torch.float64).reshape(128, 2).add(0.25)
    X, y = X_all[:args.batch], y_all[:args.batch]
    lam = torch.rand(args.batch, generator=generator)
    idx = torch.randperm(args.batch, generator=generator)
    _equal(reference(X, y, lam, idx), candidate(X, y, lam, idx))
    dataset = TensorDataset(X_all, y_all)

    def loader_for(operation):
        transform = Mixup(0.5, beta_per_sample=True, random_state=231)
        transform.operation = operation
        loader = AugmentedDataLoader(
            dataset, transforms=transform, batch_size=args.batch, shuffle=False,
            num_workers=0, generator=torch.Generator().manual_seed(231),
        )
        state = transform.rng.get_state()
        generator_state = loader.generator.get_state()

        def traverse():
            # Explicit common-random-numbers policy, reset included in timing.
            transform.rng.set_state(state)
            loader.generator.set_state(generator_state)
            return list(loader)

        return traverse, transform

    expected_loader, reference_transform = loader_for(reference)
    actual_loader, candidate_transform = loader_for(candidate)
    _equal(expected_loader(), actual_loader())
    np.testing.assert_array_equal(reference_transform.rng.rand(20),
                                  candidate_transform.rng.rand(20))
    traverse, _ = loader_for(function)
    for stage, run, inputs in (
        ("component", lambda: function(X, y, lam, idx),
         [("X", X), ("y", y), ("lam", lam), ("idx_perm", idx)]),
        ("loader", traverse, [("dataset_X", X_all), ("dataset_y", y_all)]),
    ):
        with torch.no_grad():
            measurement = measure_step(run, device=torch.device("cpu"), warmup=5,
                                       min_run_time=args.min_run_time)
            gc.collect()
            tracemalloc.start()
            output = run()
            retained, peak = tracemalloc.get_traced_memory()
            del output
            tracemalloc.stop()
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        record = {
            "schema_version": 1,
            "case_id": f"mixup/batch{args.batch}/{stage}",
            "status": "ok", "implementation": args.implementation,
            "config": {
                "inputs": [_input(name, value) for name, value in inputs],
                "seed": 231, "threads": 1, "workers": 0,
                "cache_policy": "fresh process; both paths verified first; warmup/calibration; OS cache uncontrolled",
                "autograd": "torch.no_grad; inputs require_grad=False",
                "precision": "float32 signal/coefficients; float64 fractional targets",
                "parameters": {"batch": args.batch, "pair": args.pair,
                               "stage": stage, "dataset_windows": 128,
                               "baseline": BASELINE, "baseline_function_sha256": digest,
                               "experimental_revision": EXPERIMENTAL,
                               "experimental_function_sha256": candidate_digest,
                               "alpha": 0.5, "beta_per_sample": True,
                               "shuffle": False, "n_augmentation": 0},
            },
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {"torch": str(torch.__version__),
                                 "braindecode": str(braindecode.__version__),
                                 "numpy": str(np.__version__)},
                "device": "cpu", "device_name": args.device_name,
            },
            "source": {"commit": _git("rev-parse", "HEAD"), "dirty": False,
                       "dirty_source_hashes": {}},
            "correctness": {
                "status": "passed",
                "method": "torch.equal and dtype on component and every full-loader signal/target/lam; identical NumPy next-state",
                "metrics": {"loader_windows": 128},
            },
            **measurement,
            "notes": [
                "Production remains scalar; candidate is the archived exact broadcast experiment, not enabled by default. Only Mixup operation differs. Component inputs fixed; loader resets transform NumPy and DataLoader generator states on each traversal (included).",
                "Loader timing includes iteration, sample retrieval, collation, cloning, transform parameter draws, Mixup, output retention/destruction; construction, synthetic input creation, disk I/O and training excluded.",
                "Alternating serial fresh-process pairs; 4 allocated CPUs, 1 torch/BLAS thread, 0 workers; hardware not exclusive. No CUDA/MPS executed.",
                "CPU memory: tracemalloc excludes most native tensor buffers; cumulative process RSS includes imports, setup, both correctness paths and previous stage, not isolated incremental allocation. CUDA peak is inapplicable, not zero.",
                "CPU allocation evidence: " + json.dumps({
                    "traced_retained_bytes": retained, "traced_peak_bytes": peak,
                    "process_high_water_rss_bytes": rss}, sort_keys=True),
            ],
        }
        historical_only(record, "Pinned rejected candidate replay only; not a new Mixup qualification.")
        write_result(record, args.output /
                     f"batch{args.batch}-pair{args.pair}-{args.implementation}-{stage}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--implementation", choices=["reference", "candidate"])
    parser.add_argument("--batch", type=int, choices=[1, 16, 64])
    parser.add_argument("--pair", type=int)
    args = parser.parse_args()
    if args.implementation:
        _worker(args)
        return
    if args.pairs < 5:
        parser.error("Retained comparisons require at least five independent pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    for batch in (1, 16, 64):
        for pair in range(args.pairs):
            order = ("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference")
            for implementation in order:
                subprocess.run([
                    sys.executable, "-m", "benchmarks.acceleration.bench_mixup",
                    "--output", str(args.output.resolve()), "--device-name", args.device_name,
                    "--min-run-time", str(args.min_run_time), "--batch", str(batch),
                    "--pair", str(pair), "--implementation", implementation,
                ], check=True)


if __name__ == "__main__":
    main()

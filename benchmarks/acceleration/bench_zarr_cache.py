"""Compare Zarr cache conversion with a preserved git baseline, locally only.

Example (all generated stores/results stay outside the checkout)::

    python -m benchmarks.acceleration.bench_zarr_cache \
        --output /tmp/zarr-evidence --pairs 5

Each pair runs reference/candidate in separate fresh processes, alternating
order. Each process measures cache-cold conversion, warm path lookup, and full
memmap traversal separately using the shared v1 contract. Cache-cold does NOT
mean cold OS page cache. Cold timing includes creating/cleaning a private cache
workspace per call; no published cache outside that workspace is ever deleted.
RSS is process high-water RSS (including imports/setup and destination pages),
not the conversion-array bound. Read tracing runs outside the timed sections.

The candidate explicitly opts into ``bounded_conversion=True``; ordinary
library loads retain the historical first-axis slab conversion by default.
This private-lab experiment qualifies a conversion-array objective, NOT a speed
or total-RSS objective: each float64 conversion selection is at most 64 MiB,
independent of recording length, and remains within a source chunk. Scheduling
uses O(ndim) state. It is not approval for public/default-production promotion.
More chunk-contained reads and repeated decoding of oversized compressed chunks
can cost throughput and increase cumulative RSS. Initial diagnostic five-pair
runs observed median cold candidate/reference ratios 2.08 and 1.38, and oversized
chunk RSS medians 738 MB versus 796 MB (decimal). Those runs overlapped tests and
are NOT clean causal estimates; their raw negative evidence is preserved. Fresh
qualification must run both scenarios sequentially after tests and edits stop,
retain every pair, and record the exact measured source. No safe throughput
mitigation is claimed: batching across chunks violates this contract, and full
chunk decoding can violate the float64 allocation bound. Compressed-decode peak
remains unknown, separate from the bounded conversion array. See accompanying
external manifests for controlled rerun costs and workload/cache conditions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import mne
import numpy as np
import torch

import braindecode
from benchmarks.acceleration._common import measure_step, write_result
from braindecode.datasets import base

BASELINE = "aa58e84052b3fffbdcbd77e7b64c9482ccfee172"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "braindecode/datasets/base.py",
    "test/unit_tests/datasets/test_zarr_to_memmap_concurrency.py",
    "benchmarks/acceleration/bench_zarr_cache.py",
    "benchmarks/acceleration/_common.py",
)


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _converter(implementation, baseline):
    if implementation == "candidate":
        return partial(base._zarr_to_memmap, bounded_conversion=True)
    # Extract the actual preserved function, not a reimplementation of it.
    source = _git("show", f"{baseline}:braindecode/datasets/base.py")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_zarr_to_memmap"
    )
    namespace = {"Path": Path, "os": os, "np": np, "zarr": base.zarr}
    # Trusted local repository commit explicitly selected by the caller.
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), baseline, "exec"),
        namespace,
    )
    return namespace["_zarr_to_memmap"]


def _provenance():
    changed = set(_git("diff", "HEAD", "--name-only").splitlines())
    changed.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
        if name in changed
    }
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(hashes),
        "dirty_source_hashes": hashes,
    }


def _peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _worker(args):
    if base.zarr is False or base.zarr is None:
        raise RuntimeError(
            "Zarr unavailable: dependency unverified; no performance result"
        )
    zarr = base.zarr
    torch.set_num_threads(args.threads)
    converter = _converter(args.implementation, args.baseline)
    source = np.arange(int(np.prod(args.shape)), dtype=np.float32).reshape(args.shape)
    source /= 13
    with tempfile.TemporaryDirectory(
        prefix="zarr-bench-", dir=args.output
    ) as directory:
        directory = Path(directory)
        path = directory / "source.zarr"
        root = zarr.open(str(path), mode="w")
        group = root.create_group("recording_0")
        try:
            array = group.create_array(
                "data", shape=source.shape, chunks=args.chunks, dtype="f4"
            )
        except AttributeError:
            array = group.zeros(
                "data", shape=source.shape, chunks=args.chunks, dtype="f4"
            )
        array[:] = source
        stored_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        selections = []
        original = zarr.Array.__getitem__

        def traced_read(self, selection):
            parts = selection if isinstance(selection, tuple) else (selection,)
            parts += (slice(None),) * (len(self.shape) - len(parts))
            lengths = [
                len(range(*part.indices(size))) for part, size in zip(parts, self.shape)
            ]
            selections.append(lengths)
            return original(self, selection)

        zarr.Array.__getitem__ = traced_read
        try:
            cache = converter(path, "recording_0")
        finally:
            zarr.Array.__getitem__ = original
        actual = np.load(cache, mmap_mode="r")
        equal = np.array_equal(actual, source.astype(np.float64))
        assert equal and actual.dtype == np.float64
        checksum = float(actual.sum())
        cache_bytes = cache.stat().st_size
        cache_inode = cache.stat().st_ino
        del actual
        max_elements = max(int(np.prod(lengths)) for lengths in selections)
        metrics = {
            "exact_float64_equal": equal,
            "cache_bytes": cache_bytes,
            "source_store_bytes": stored_bytes,
            "read_count": len(selections),
            "read_slice_shapes": selections,
            "max_read_float64_bytes": max_elements * 8,
            "max_read_source_bytes": max_elements * source.dtype.itemsize,
            "source_chunk_uncompressed_bytes": int(np.prod(args.chunks)) * 4,
            "compressed_chunk_decompression_peak_bytes": None,
            "full_traversal_checksum": checksum,
        }
        cold_calls = 0

        def cold():
            nonlocal cold_calls
            # Per-call directory owns both symlink and all resulting cache files.
            with tempfile.TemporaryDirectory(prefix="cold-", dir=directory) as call_dir:
                alias = Path(call_dir) / "source.zarr"
                alias.symlink_to(path, target_is_directory=True)
                converter(alias, "recording_0")
            cold_calls += 1

        def warm():
            return converter(path, "recording_0")

        def traverse():
            return float(np.load(cache, mmap_mode="r").sum())

        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "numpy": np.__version__,
                "zarr": zarr.__version__,
                "torch": str(torch.__version__),
                "mne": mne.__version__,
                "braindecode": braindecode.__version__,
            },
            "device": "cpu",
            "device_name": args.device_name,
        }
        for phase, run in (("cold", cold), ("warm", warm), ("traversal", traverse)):
            measurement = measure_step(
                run,
                device=torch.device("cpu"),
                warmup=0,
                min_run_time=args.min_run_time,
            )
            assert cache.stat().st_ino == cache_inode
            assert traverse() == checksum
            phase_metrics = dict(metrics, peak_rss_bytes=_peak_rss())
            record = {
                "schema_version": 1,
                "case_id": f"zarr-cache-{phase}-pair-{args.pair}",
                "status": "ok",
                "implementation": args.implementation,
                "config": {
                    "inputs": [
                        {
                            "name": "source",
                            "shape": list(source.shape),
                            "dtype": str(source.dtype),
                            "strides": list(source.strides),
                            "stride_unit": "bytes",
                        }
                    ],
                    "seed": None,
                    "threads": args.threads,
                    "workers": 0,
                    "cache_policy": "cache-cold per-call private workspace; OS cache uncontrolled"
                    if phase == "cold"
                    else "existing cache; OS cache warmed by prior checks",
                    "autograd": "not_applicable",
                    "precision": "float32 source; exact float64 output",
                    "parameters": {
                        "phase": phase,
                        "chunks": args.chunks,
                        "baseline_commit": args.baseline,
                        "pair": args.pair,
                        "cold_calls": cold_calls,
                    },
                },
                "environment": env,
                "source": _provenance(),
                "correctness": {
                    "status": "passed",
                    "method": "np.array_equal against source.astype(float64); inode stability; complete sum",
                    "metrics": phase_metrics,
                },
                **measurement,
                "notes": [
                    "Reference function is AST-extracted unchanged from baseline_commit; source identifies harness checkout.",
                    "Generated deterministic arange/13 input, no RNG. Source generation and validation excluded from timing.",
                    "Cold includes per-call private directory/symlink creation and cleanup; never deletes shared or live-reader caches.",
                    "RSS is cumulative process high-water including imports, setup, previous phases, and resident memmap pages; not isolated conversion allocation.",
                    "Compressed chunk decoding allocation is backend-dependent and unmeasured separately: null is unknown, not zero. Source chunk uncompressed bytes is an extent, not a peak bound.",
                    f"Zarr metadata: {array.metadata!s}",
                    "Traversal opens the memmap and sums every value; warm phase is path lookup only.",
                    "Private-lab objective: float64 conversion arrays <=64 MiB independent of recording length, not lower total RSS or faster execution; no public/default-production promotion.",
                    "Qualification requires no concurrent lane tests/edits; the accompanying evidence manifest attests run scheduling, not this timing utility.",
                ],
            }
            write_result(
                record,
                args.output / f"pair-{args.pair}-{args.implementation}-{phase}.json",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--shape", type=int, nargs="+", default=[16, 262144])
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 4096])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument(
        "--device-name", default=platform.processor() or platform.machine()
    )
    parser.add_argument("--implementation", choices=["reference", "candidate"])
    parser.add_argument("--pair", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error("generated data/results must be outside the checkout")
    if len(args.shape) != len(args.chunks) or any(
        n <= 0 for n in args.shape + args.chunks
    ):
        parser.error("positive shape/chunks must have matching dimensions")
    args.baseline = _git("rev-parse", f"{args.baseline}^{{commit}}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.implementation:
        _worker(args)
        return
    if args.pairs < 5:
        parser.error("at least 5 independent pairs are required")
    for pair in range(args.pairs):
        order = (
            ["reference", "candidate"] if pair % 2 == 0 else ["candidate", "reference"]
        )
        for implementation in order:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.acceleration.bench_zarr_cache",
                    "--output",
                    str(args.output),
                    "--implementation",
                    implementation,
                    "--pair",
                    str(pair),
                    "--shape",
                    *map(str, args.shape),
                    "--chunks",
                    *map(str, args.chunks),
                    "--threads",
                    str(args.threads),
                    "--min-run-time",
                    str(args.min_run_time),
                    "--baseline",
                    args.baseline,
                    "--device-name",
                    args.device_name,
                ],
                cwd=ROOT,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
                check=True,
                timeout=300,
            )
    summary = {
        "pairs": args.pairs,
        "order": "alternating reference/candidate",
        "baseline_commit": args.baseline,
        "phases": {},
    }
    for phase in ("cold", "warm", "traversal"):
        records = {
            kind: [
                json.loads((args.output / f"pair-{i}-{kind}-{phase}.json").read_text())
                for i in range(args.pairs)
            ]
            for kind in ("reference", "candidate")
        }
        values = {
            kind: [r["timing"]["median_s"] for r in rows]
            for kind, rows in records.items()
        }
        ratios = np.array(values["candidate"]) / np.array(values["reference"])
        summary["phases"][phase] = {
            "invocation_medians_s": values,
            "candidate_over_reference_ratios": ratios.tolist(),
            "ratio_median": float(np.median(ratios)),
            "ratio_iqr": float(np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)),
            "ratio_range": [float(ratios.min()), float(ratios.max())],
        }
    (args.output / "paired-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

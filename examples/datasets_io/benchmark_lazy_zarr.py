"""Benchmark: Lazy vs Eager Zarr Loading with PyTorch Profiler
=============================================================

Profiles DataLoader throughput, memory, and per-op breakdown using
``torch.profiler`` for both ``preload=True`` and ``preload=False``.
Generates Chrome-compatible trace files for inspection.
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD-3

import gc
import time
from pathlib import Path

import numpy as np
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader

from braindecode.datasets import BaseConcatDataset

REPO_ID = "braindecode/mdd_mumtaz2016"
BATCH_SIZE = 64
N_WARMUP = 2
N_ACTIVE = 20
N_WORKERS_LIST = [0, 2, 4]
TRACE_DIR = Path("profiler_traces")


def load_dataset(preload):
    gc.collect()
    t0 = time.time()
    ds = BaseConcatDataset.pull_from_hub(REPO_ID, preload=preload)
    elapsed = time.time() - t0
    return ds, elapsed


def profile_dataloader(ds, label, num_workers):
    """Profile DataLoader iteration and export a Chrome trace."""
    TRACE_DIR.mkdir(exist_ok=True)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    trace_path = TRACE_DIR / f"{label}_w{num_workers}.json"

    prof_schedule = schedule(wait=1, warmup=N_WARMUP, active=N_ACTIVE, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i, batch in enumerate(loader):
            with record_function("batch_processing"):
                X, y, crop_inds = batch
            prof.step()
            if i >= N_WARMUP + N_ACTIVE + 1:
                break

    prof.export_chrome_trace(str(trace_path))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return str(trace_path)


def benchmark_random_access(ds, label, n_samples=1000):
    """Profile single __getitem__ calls."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=n_samples, replace=False)

    trace_path = TRACE_DIR / f"{label}_getitem.json"

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for idx in indices:
            with record_function("getitem"):
                _ = ds[int(idx)]

    prof.export_chrome_trace(str(trace_path))

    times_us = [e.cpu_time_total for e in prof.key_averages() if e.key == "getitem"]
    times_ms = np.array(times_us) / 1e3 if times_us else np.array([0.0])
    return {
        "trace": str(trace_path),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "mean_ms": float(np.mean(times_ms)),
    }


def main():
    TRACE_DIR.mkdir(exist_ok=True)

    print(f"Dataset: {REPO_ID}")
    print(f"Batch: {BATCH_SIZE} | Active batches: {N_ACTIVE}")
    print("=" * 70)

    # ---- Load ----
    print("\n[1] Load time")
    print("-" * 50)
    lazy_ds, t_lazy = load_dataset(preload=False)
    print(
        f"  Lazy:  {t_lazy:.2f}s  ({len(lazy_ds)} windows, "
        f"{len(lazy_ds.datasets)} recordings)"
    )
    eager_ds, t_eager = load_dataset(preload=True)
    print(f"  Eager: {t_eager:.2f}s  ({len(eager_ds)} windows)")
    print(f"  Speedup: {t_eager / t_lazy:.1f}x")

    # ---- Random access ----
    print("\n[2] Random access (__getitem__) — 1000 samples")
    print("-" * 50)
    for label, ds in [("lazy", lazy_ds), ("eager", eager_ds)]:
        stats = benchmark_random_access(ds, label)
        print(
            f"  {label:5s}: median={stats['median_ms']:.3f}ms  "
            f"p95={stats['p95_ms']:.3f}ms  "
            f"mean={stats['mean_ms']:.3f}ms  "
            f"[{stats['trace']}]"
        )

    # ---- DataLoader ----
    print(f"\n[3] DataLoader profiling ({N_ACTIVE} batches)")
    print("-" * 50)
    for nw in N_WORKERS_LIST:
        print(f"\n  ===== num_workers={nw} =====")
        for label, ds in [("lazy", lazy_ds), ("eager", eager_ds)]:
            print(f"\n  >> {label} (w={nw}):")
            trace = profile_dataloader(ds, label, nw)
            print(f"  Trace: {trace}")

    # ---- Correctness ----
    print("\n[4] Correctness")
    print("-" * 50)
    rng = np.random.default_rng(0)
    for idx in rng.choice(len(lazy_ds), size=100, replace=False):
        lX, ly, li = lazy_ds[int(idx)]
        eX, ey, ei = eager_ds[int(idx)]
        assert np.array_equal(lX, eX) and ly == ey and li == ei
    print("  100 random windows: PASS")

    print(f"\nAll traces in: {TRACE_DIR.resolve()}")


if __name__ == "__main__":
    main()

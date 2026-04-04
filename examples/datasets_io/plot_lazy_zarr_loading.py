"""Lazy Zarr Loading from Hugging Face Hub
========================================

This example shows how to load a dataset lazily from the Hugging Face Hub
using ``preload=False``, and access windows in parallel via PyTorch
DataLoader and joblib. With lazy loading, data stays on disk and is read
on-demand per window, so large datasets no longer need to fit in RAM.
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD-3

import time

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from braindecode.datasets import BaseConcatDataset


def main():
    # %%
    # Load the dataset lazily (no data in RAM)
    # -----------------------------------------
    t0 = time.time()
    lazy_ds = BaseConcatDataset.pull_from_hub(
        "braindecode/mdd_mumtaz2016", preload=False
    )
    t_lazy = time.time() - t0

    print(
        f"Lazy load:  {t_lazy:.2f}s | {len(lazy_ds)} windows | "
        f"{len(lazy_ds.datasets)} recordings"
    )
    print(f"  windows is None: {lazy_ds.datasets[0].windows is None}")
    print(f"  zarr ref set:    {lazy_ds.datasets[0]._zarr_data is not None}")
    print(f"  First window:    {lazy_ds[0][0].shape}")
    print()

    # %%
    # Parallel access with joblib
    # ----------------------------
    indices = np.random.default_rng(42).choice(len(lazy_ds), size=200, replace=False)

    def fetch_window(idx):
        X, y, crop_inds = lazy_ds[idx]
        return X.shape, y

    t0 = time.time()
    results = Parallel(n_jobs=4, backend="loky")(
        delayed(fetch_window)(int(i)) for i in indices
    )
    t_parallel = time.time() - t0
    shapes, targets = zip(*results)
    print(f"Joblib (200 windows, 4 workers): {t_parallel:.2f}s")
    print(f"  All shapes equal: {len(set(shapes)) == 1} ({shapes[0]})")
    print(f"  Unique targets:   {sorted(set(targets))}")
    print()

    # %%
    # PyTorch DataLoader with multiple workers
    # ------------------------------------------
    for n_workers in [0, 2, 4]:
        loader = DataLoader(lazy_ds, batch_size=64, shuffle=True, num_workers=n_workers)
        t0 = time.time()
        n_batches = 0
        for batch in loader:
            X, y, crop_inds = batch
            assert X.shape[1:] == (19, 1000)
            assert X.dtype == torch.float32
            n_batches += 1
            if n_batches >= 10:
                break
        t_dl = time.time() - t0
        print(
            f"DataLoader num_workers={n_workers}: "
            f"{n_batches} batches in {t_dl:.2f}s "
            f"({n_batches * 64 / t_dl:.0f} samples/s)"
        )
    print()

    # %%
    # Compare with eager (preloaded) loading
    # ----------------------------------------
    t0 = time.time()
    eager_ds = BaseConcatDataset.pull_from_hub(
        "braindecode/mdd_mumtaz2016", preload=True
    )
    t_eager = time.time() - t0
    print(f"Eager load: {t_eager:.2f}s | {len(eager_ds)} windows")

    # %%
    # Verify lazy and eager return identical data
    # ---------------------------------------------
    n_check = 20
    check_idx = np.random.default_rng(0).choice(
        len(lazy_ds), size=n_check, replace=False
    )
    for idx in check_idx:
        lX, ly, li = lazy_ds[int(idx)]
        eX, ey, ei = eager_ds[int(idx)]
        np.testing.assert_array_equal(lX, eX)
        assert ly == ey and li == ei

    print(f"Verified {n_check} random windows: lazy == eager")
    print(f"Speed-up (load time): {t_eager / t_lazy:.1f}x faster with lazy")


if __name__ == "__main__":
    main()

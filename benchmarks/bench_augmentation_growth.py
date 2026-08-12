"""Benchmark the :class:`~braindecode.augmentation.AugmentedDataLoader` collate.

Run from the repository root with::

    python -m benchmarks.bench_augmentation_growth

Two measurements:

1. **Expansion scaling** (single process). Cost of in-place augmentation
   (``n_augmentation=0``) versus fixed expansion, where each batch is grown to
   ``(1 + n_augmentation)`` times its size. The batch is collated once per step
   regardless of ``n_augmentation``; only the transform runs
   ``(1 + n_augmentation)`` times, so time-per-step grows ~linearly.

2. **Worker scaling.** ``collate_fn`` (and therefore the augmentation) runs in
   the DataLoader worker process, so ``num_workers > 0`` parallelizes it. This
   only works because the collate is a picklable callable. Pinned to one
   intra-op thread per process so ``num_workers`` maps to cores used (otherwise
   torch's intra-op threads already saturate the cores and hide the scaling).

This is a standalone script, not a pytest test: it has no extra dependencies
and is not collected by the test suite.
"""

from statistics import median
from time import perf_counter

import torch
from torch.utils.data import Dataset

from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation.transforms import SmoothTimeMask


class _LazyEEGDataset(Dataset):
    """Synthesize a random EEG window per index.

    Cheap to pickle (only stores the shape), so spawning DataLoader workers does
    not copy a large in-memory tensor — unlike a ``TensorDataset``.
    """

    def __init__(self, n_samples, n_chans, n_times, n_classes=4):
        self.n_samples = n_samples
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_classes = n_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.randn(self.n_chans, self.n_times), idx % self.n_classes


def _set_one_thread(_worker_id):
    torch.set_num_threads(1)


def _time_epoch(loader):
    """Iterate one full epoch; return (seconds, samples produced)."""
    start = perf_counter()
    n = 0
    for X, _ in loader:
        n += X.shape[0]
    return perf_counter() - start, n


def _median_epoch(loader, repeats):
    _time_epoch(loader)  # warmup (also spawns persistent workers)
    times, produced = [], 0
    for _ in range(repeats):
        elapsed, produced = _time_epoch(loader)
        times.append(elapsed)
    return median(times), produced


def expansion_scaling(dataset, batch_size, n_augmentations, repeats):
    n_samples = len(dataset)
    print("expansion scaling (num_workers=0):")
    header = f"{'n_aug':>6} {'batch_out':>10} {'ms/epoch':>10} {'ms/step':>9} {'samples/s':>11}"
    print(header)
    print("-" * len(header))
    n_steps = -(-n_samples // batch_size)  # ceil
    for n_aug in n_augmentations:
        loader = AugmentedDataLoader(
            dataset,
            transforms=SmoothTimeMask(probability=1.0),
            batch_size=batch_size,
            n_augmentation=n_aug,
        )
        t, produced = _median_epoch(loader, repeats)
        print(
            f"{n_aug:>6} {batch_size * (1 + n_aug):>10} "
            f"{t * 1e3:>10.1f} {t / n_steps * 1e3:>9.2f} {produced / t:>11.0f}"
        )


def worker_scaling(dataset, batch_size, n_aug, workers, repeats):
    torch.set_num_threads(1)  # one intra-op thread/process -> workers == cores
    print(f"\nworker scaling (n_augmentation={n_aug}, 1 intra-op thread/process):")
    header = f"{'workers':>7} {'ms/epoch':>10} {'samples/s':>11} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    base = None
    for nw in workers:
        loader = AugmentedDataLoader(
            dataset,
            transforms=SmoothTimeMask(probability=1.0),
            batch_size=batch_size,
            n_augmentation=n_aug,
            num_workers=nw,
            persistent_workers=nw > 0,
            worker_init_fn=_set_one_thread if nw > 0 else None,
        )
        t, produced = _median_epoch(loader, repeats)
        base = base or t
        print(f"{nw:>7} {t * 1e3:>10.1f} {produced / t:>11.0f} {base / t:>7.2f}x")


def benchmark(n_samples=2048, n_chans=22, n_times=1000, batch_size=64):
    dataset = _LazyEEGDataset(n_samples, n_chans, n_times)
    print(f"{n_samples} samples x {n_chans} ch x {n_times} t | batch={batch_size}\n")
    expansion_scaling(dataset, batch_size, n_augmentations=(0, 1, 5, 10), repeats=5)
    worker_scaling(dataset, batch_size, n_aug=5, workers=(0, 2, 4), repeats=3)


if __name__ == "__main__":
    torch.manual_seed(0)
    benchmark()

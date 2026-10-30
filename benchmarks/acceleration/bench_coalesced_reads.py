"""Synthetic lazy-FIF batch probe: ``coalesce_reads=True`` against default prefetch.

``--mode prefetch`` runs a frozen copy of the default batch path; ``--mode
coalesced`` opts in. Run each mode in a fresh process, as serial alternating
pairs on one node, with a prefetch-versus-prefetch control. No cache eviction
or downloads. Cells: ``overlap`` (stride 1 s, 4-s windows), ``adjacent``
(stride 4 s), ``sparse`` (shuffled), ``cross`` (32 recordings x 16 windows).
"""

import argparse
import bisect
import json
import os
import resource
import threading
import time

import mne
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

from braindecode.datasets import BaseConcatDataset, EEGWindowsDataset, base
from braindecode.datasets.base import _read_fif_window


def prefetch(self, indices):
    """3ce69fc6 BaseConcatDataset.__getitems__, unchanged."""
    if hasattr(os, "posix_fadvise"):
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                continue
            ds, i = self._locate(idx)
            if isinstance(ds, EEGWindowsDataset) and ds._zarr_data is None:
                _, start, stop = ds.crop_inds[i].tolist()
                _read_fif_window(ds.raw, start, stop, willneed=True)
    return [self[idx] for idx in indices]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=["prefetch", "coalesced"], required=True)
    parser.add_argument(
        "--cell", choices=["overlap", "adjacent", "sparse", "cross"], required=True
    )
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    mne.set_log_level("ERROR")
    started = time.perf_counter()
    path = os.path.join(args.root, "synthetic-raw.fif")
    if args.prepare:
        info = mne.create_info([f"e{i}" for i in range(22)], 250, "eeg")
        rng = np.random.RandomState(7)
        raw = mne.io.RawArray(rng.standard_normal((22, 513000)) * 1e-5, info)
        raw.save(path, buffer_size_sec=1, overwrite=False, verbose=False)
        for i in range(32):
            small = mne.io.RawArray(rng.standard_normal((22, 5000)) * 1e-5, info)
            small.save(
                os.path.join(args.root, f"cross-{i}-raw.fif"),
                buffer_size_sec=1,
                overwrite=False,
                verbose=False,
            )
        return
    if args.mode == "coalesced" and not hasattr(base, "_coalesced_fif_batch"):
        parser.error("Coalesced opt-in unavailable in this checkout")
    if args.mode == "prefetch":
        BaseConcatDataset.__getitems__ = prefetch
    stride = 250 if args.cell in ("overlap", "cross") else 1000
    recordings = []
    for i in range(32 if args.cell == "cross" else 1):
        filename = (
            os.path.join(args.root, f"cross-{i}-raw.fif")
            if args.cell == "cross"
            else path
        )
        raw = mne.io.read_raw_fif(filename, preload=False, verbose=False)
        n = 16 if args.cell == "cross" else 512
        starts = np.arange(n) * stride
        metadata = pd.DataFrame(
            {
                "i_window_in_trial": np.arange(n),
                "i_start_in_trial": starts,
                "i_stop_in_trial": starts + 1000,
                "target": np.arange(n) % 2,
            }
        )
        recordings.append(EEGWindowsDataset(raw, metadata))
    ds = BaseConcatDataset(recordings, coalesce_reads=args.mode == "coalesced")
    indices = np.arange(512)
    if args.cell == "sparse":
        np.random.RandomState(8).shuffle(indices)
    batches = [indices[i : i + 32].tolist() for i in range(0, 512, 32)]
    loader = DataLoader(
        ds,
        batch_sampler=batches,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    peak = [0]
    stop = threading.Event()
    process = psutil.Process()

    def sample():
        while not stop.wait(0.01):
            total = 0
            for p in [process, *process.children(recursive=True)]:
                try:
                    total += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak[0] = max(peak[0], total)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    first_batch = None

    def epoch():
        nonlocal first_batch
        t0 = time.perf_counter()
        checksum = 0.0
        count = 0
        for x, y, crop in loader:
            if first_batch is None:
                first_batch = time.perf_counter() - started
            checksum += x[:, 0, 0].double().sum().item() + y.sum().item()
            count += len(x)
        assert count == 512
        return time.perf_counter() - t0, checksum

    warmups = []
    stable = False
    for _ in range(8):
        warmups.append(epoch()[0])
        if len(warmups) >= 3 and max(warmups[-3:]) / min(warmups[-3:]) <= 1.10:
            stable = True
            break
    epoch_times, checksums = [], []
    for _ in range(5):
        elapsed, checksum = epoch()
        epoch_times.append(elapsed)
        checksums.append(checksum)
    stop.set()
    monitor.join()
    # Separate diagnostic pass: do not instrument the timed workers.
    calls, samples, physical_bytes = 0, 0, 0
    read = base._read_fif_window

    def counted(raw, start, end, willneed=False):
        nonlocal calls, samples, physical_bytes
        if not willneed:
            calls += 1
            samples += end - start
            extra = raw._raw_extras[0]
            bounds = extra["bounds"]
            first = bisect.bisect_right(bounds, start + raw.first_samp) - 1
            last = bisect.bisect_left(bounds, end + raw.first_samp)
            tags = extra["ent"][first:last]
            physical_bytes += tags[-1].pos + 16 + tags[-1].size - tags[0].pos
        return read(raw, start, end, willneed)

    base._read_fif_window = counted
    globals()["_read_fif_window"] = counted
    for batch in batches:
        ds.__getitems__(batch)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "cell": args.cell,
                "workers": args.workers,
                "epoch_s": epoch_times,
                "checksum": checksums,
                "first_batch_s": first_batch,
                "warmup_epoch_s": warmups,
                "warmup_stable": stable,
                "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "aggregate_sampled_rss": peak[0],
                "read_calls": calls,
                "requested_samples": samples,
                "requested_window_samples": 512000,
                "physical_buffer_bytes": physical_bytes,
                "versions": {
                    "torch": torch.__version__,
                    "mne": mne.__version__,
                    "numpy": np.__version__,
                },
            }
        )
    )


if __name__ == "__main__":
    main()

"""Window reads from a preloaded ``EEGWindowsDataset``, against the earlier path.

``BeforeWindows`` repeats the earlier item path (no transform), which called
``raw._getitem`` and, with ``targets_from="channels"``, ``raw.get_channel_types()``
for every window. Both versions read the same windows in alternating rounds,
and their items are checked equal. The epoch time is one pass of a
``DataLoader`` (batch 64, shuffled, ``num_workers=0``, one torch thread), data
only, no model.
Run from the repository root::

    python -m benchmarks.acceleration.bench_window_getitem
"""

import argparse
import time

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from braindecode.datasets import BaseConcatDataset, EEGWindowsDataset

SFREQ, N_TIMES, WINDOW, STRIDE = 250, 150_000, 1000, 100  # 10 min recording


class BeforeWindows(EEGWindowsDataset):
    def __getitem__(self, index):
        crop_inds = self.crop_inds[index].tolist()
        _, start, stop = crop_inds
        X = self.raw._getitem((slice(None), slice(start, stop)), return_times=False)
        X = np.array(X, dtype="float32")
        if self.targets_from == "metadata":
            return X, self.y[index], crop_inds
        misc_mask = np.array(self.raw.get_channel_types()) == "misc"
        return X[~misc_mask, :], X[misc_mask, -1].copy(), crop_inds


def make_recording(n_chans, targets_from):
    n_misc = 2 if targets_from == "channels" else 0
    info = mne.create_info(
        n_chans + n_misc, SFREQ, ["eeg"] * n_chans + ["misc"] * n_misc
    )
    data = np.random.default_rng(0).standard_normal((n_chans + n_misc, N_TIMES))
    starts = np.arange(0, N_TIMES - WINDOW + 1, STRIDE)
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": np.arange(len(starts)),
            "i_start_in_trial": starts,
            "i_stop_in_trial": starts + WINDOW,
            "target": starts % 2,
        }
    )
    return mne.io.RawArray(data, info), metadata


def time_items(ds, indices):
    started = time.perf_counter()
    for i in indices:
        ds[i]
    return (time.perf_counter() - started) / len(indices) * 1e6


def time_epoch(ds):
    loader = DataLoader(BaseConcatDataset([ds]), batch_size=64, shuffle=True)
    started = time.perf_counter()
    for _ in loader:
        pass
    return time.perf_counter() - started


def spread(values, fmt):
    return f"{np.median(values):{fmt}} ({min(values):{fmt}}-{max(values):{fmt}})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--items", type=int, default=1500)
    args = parser.parse_args()
    mne.set_log_level("ERROR")
    # the batch collate (torch.stack) takes most of an epoch, and with all
    # threads it varies several-fold whenever other processes load the machine
    torch.set_num_threads(1)
    print(
        "| channels | targets_from | before us/item | after us/item "
        "| before s/epoch | after s/epoch | epoch speedup |"
    )
    print("|---|---|---|---|---|---|---|")
    for n_chans, targets_from in [
        (22, "metadata"),
        (64, "metadata"),
        (128, "metadata"),
        (64, "channels"),
    ]:
        raw, metadata = make_recording(n_chans, targets_from)
        datasets = {
            "before": BeforeWindows(raw, metadata, targets_from=targets_from),
            "after": EEGWindowsDataset(raw, metadata, targets_from=targets_from),
        }
        for i in range(len(metadata)):
            for a, b in zip(datasets["before"][i], datasets["after"][i]):
                np.testing.assert_array_equal(a, b)
        rng = np.random.default_rng(1)
        item_us = {name: [] for name in datasets}
        epoch_s = {name: [] for name in datasets}
        for r in range(args.rounds):
            indices = rng.integers(len(metadata), size=args.items)
            for name in ("before", "after") if r % 2 == 0 else ("after", "before"):
                item_us[name].append(time_items(datasets[name], indices))
                epoch_s[name].append(time_epoch(datasets[name]))
        speedup = np.median(epoch_s["before"]) / np.median(epoch_s["after"])
        print(
            f"| {n_chans} | {targets_from} | {spread(item_us['before'], '.0f')} "
            f"| {spread(item_us['after'], '.0f')} "
            f"| {spread(epoch_s['before'], '.2f')} "
            f"| {spread(epoch_s['after'], '.2f')} | {speedup:.2f}x |"
        )
    print("items equal: yes")


if __name__ == "__main__":
    main()

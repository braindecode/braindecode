# License: BSD-3
"""Synthetic windows and the historical reference for trial-prediction tests."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def reference_trial_preds(preds, i_window_in_trials, i_stop_in_trials):
    """Frozen BASE (d9340316) ``trial_preds_from_window_preds``: the oracle."""
    assert len(preds) == len(i_window_in_trials) == len(i_stop_in_trials), (
        f"{len(preds)}, {len(i_window_in_trials)}, {len(i_stop_in_trials)}"
    )
    preds_per_trial = []
    cur_trial_preds = []
    i_last_stop = None
    i_last_window = -1
    for window_preds, i_window, i_stop in zip(
        preds, i_window_in_trials, i_stop_in_trials
    ):
        window_preds = np.array(window_preds)
        if i_window != (i_last_window + 1):
            assert i_window == 0, "window numbers in new trial should start from 0"
            preds_per_trial.append(np.concatenate(cur_trial_preds, axis=1))
            cur_trial_preds = []
            i_last_stop = None

        if i_last_stop is not None:
            n_needed_preds = i_stop - i_last_stop
            window_preds = window_preds[:, -n_needed_preds:]
        cur_trial_preds.append(window_preds)
        i_last_window = i_window
        i_last_stop = i_stop
    preds_per_trial.append(np.concatenate(cur_trial_preds, axis=1))
    return preds_per_trial


def window_stops(length, n_preds, stride):
    stops = list(range(n_preds, length + 1, stride))
    if stops[-1] != length:
        # last window aligned to the trial end: duplicate time overlap
        stops.append(length)
    return stops


def make_trial_windows(trial_lengths, n_preds, stride, n_classes=3, dtype=np.float64, seed=0):
    """Windows cut from known per-trial signals, in chronological order."""
    rng = np.random.default_rng(seed)
    signals, preds, i_windows, i_stops = [], [], [], []
    for length in trial_lengths:
        assert length >= n_preds
        signal = (rng.standard_normal((n_classes, length)) * 100).astype(dtype)
        signals.append(signal)
        for i_window, stop in enumerate(window_stops(length, n_preds, stride)):
            preds.append(signal[:, stop - n_preds : stop].copy())
            i_windows.append(i_window)
            i_stops.append(stop)
    return signals, preds, i_windows, i_stops


TRIAL_LAYOUTS = {
    "1_trial": {"trial_lengths": [40], "n_preds": 10, "stride": 7},
    "2_trials": {"trial_lengths": [40, 23], "n_preds": 10, "stride": 7},
    "17_trials_unequal": {
        "trial_lengths": [
            10,
            31,
            57,
            12,
            10,
            44,
            23,
            59,
            18,
            36,
            11,
            50,
            27,
            10,
            41,
            33,
            19,
        ],
        "n_preds": 10,
        "stride": 4,
    },
    "17_trials_dense_overlap": {
        "trial_lengths": [8 + 3 * i for i in range(17)],
        "n_preds": 8,
        "stride": 1,
    },
    "no_overlap": {"trial_lengths": [30, 30, 15], "n_preds": 5, "stride": 5},
}


WINDOW_CASES = {
    # same stop twice: n_needed_preds == 0 keeps the whole window ([:, -0:])
    "repeated_stop": (
        [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]],
        [0, 1, 2],
        [4, 4, 6],
    ),
    # decreasing stop: negative n_needed_preds slices from the front
    "decreasing_stop": ([[[0, 1, 2, 3]], [[4, 5, 6, 7]]], [0, 1], [8, 6]),
    # gap larger than the window: the whole window is kept
    "gap_larger_than_window": ([[[0, 1, 2, 3]], [[4, 5, 6, 7]]], [0, 1], [4, 10]),
    # extra trailing axis: overlap is removed along axis 1 only
    "three_dim_windows": (
        list(np.arange(3 * 2 * 4 * 2).reshape(3, 2, 4, 2)),
        [0, 1, 0],
        [4, 6, 4],
    ),
    "single_window_trials": (
        list(np.arange(5 * 2 * 3).reshape(5, 2, 3)),
        [0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3],
    ),
    "existing_three_windows_two_trials": (
        [[[4, 5, 6, 7]], [[6, 7, 8, 9]], [[0, 1, 2, 3]]],
        [0, 1, 0],
        [8, 10, 8],
    ),
}


class InferenceWindows(Dataset):
    """Five windows, overlapping in two unequal trials; no stored signals."""

    def __init__(self, target_kind="scalar", repeats=1, fail_at=None, ch_pos=False):
        self.target_kind = target_kind
        self.repeats = repeats
        self.reads = 0
        self.fail_at = fail_at
        self.ch_pos = ch_pos

    def __len__(self):
        return 5 * self.repeats

    def __getitem__(self, index):
        self.reads += 1
        if index == self.fail_at:
            raise RuntimeError("dataset failure")
        local = index % 5
        trial = index // 5 * 2 + (local >= 2)
        window = local if local < 2 else local - 2
        start = window * 2
        signal = np.arange(start, start + 4, dtype=np.float64)[None] + 100 * trial
        if self.target_kind == "scalar":
            target = np.int64(trial)
        elif self.target_kind == "multi":
            target = np.array([trial, trial + 10], dtype=np.int64)
        else:
            target = signal.astype(np.int64)
        row = (signal, target, [window, start, start + 4])
        return (*row, np.zeros((1, 3))) if self.ch_pos else row

    def get_metadata(self):
        return pd.DataFrame({"i_window_in_trial": [0, 1, 0, 1, 2] * self.repeats})


class InferenceModule(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.batch_sizes = []
        self.fail = fail

    def forward(self, X):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.batch_sizes.append(len(X))
        if self.fail:
            raise RuntimeError("forward failure")
        return X * self.scale

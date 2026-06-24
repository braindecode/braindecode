# Authors: The braindecode developers
#
# License: BSD (3-clause)

"""Tests for channel positions in the batch (issue #1066).

Covers the opt-in ``return_ch_pos`` accessor on windowed datasets, the
collection-level ``set_return_ch_pos`` toggle, and ``pad_channels_collate`` for
heterogeneous (variable-channel) montages.
"""

import json

import mne
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from braindecode.datasets import (
    BaseConcatDataset,
    EEGWindowsDataset,
    RawDataset,
    pad_channels_collate,
)
from braindecode.datasets.bids.hub_io import _restore_nan_from_json
from braindecode.preprocessing import create_fixed_length_windows

mne.set_log_level("ERROR")

CHS_A = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]  # 8 channels
CHS_B = ["Fz", "Cz", "Pz", "Oz", "T7", "T8", "O1", "O2", "F7", "F8"]  # 10


def _raw_ds(ch_names, label=0, sfreq=100.0, n_times=1000, montage=True):
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    data = np.random.RandomState(label).randn(len(ch_names), n_times) * 1e-6
    raw = mne.io.RawArray(data, info)
    if montage:
        raw.set_montage("standard_1020")
    return RawDataset(raw, description={"label": label})


def _windows(datasets):
    concat = BaseConcatDataset(datasets)
    return create_fixed_length_windows(
        concat,
        window_size_samples=200,
        window_stride_samples=200,
        drop_last_window=True,
        preload=True,
    )


def test_ch_pos_shape_and_values():
    win = _windows([_raw_ds(CHS_A)])
    ds = win.datasets[0]
    pos = ds.ch_pos
    assert pos.shape == (len(CHS_A), 3)
    assert pos.dtype == np.float32
    assert np.isfinite(pos).all()
    # matches the raw info loc directly
    expected = np.asarray([ch["loc"][:3] for ch in ds.raw.info["chs"]], dtype="float32")
    np.testing.assert_allclose(pos, expected)


def test_default_off_returns_three_tuple():
    win = _windows([_raw_ds(CHS_A)])
    assert len(win.datasets[0][0]) == 3  # (X, y, crop_inds)


def test_opt_in_returns_positions():
    win = _windows([_raw_ds(CHS_A)])
    win.set_return_ch_pos(True)
    sample = win.datasets[0][0]
    assert len(sample) == 4
    X, y, crop_inds, pos = sample
    assert pos.shape == (X.shape[0], 3)


def test_set_return_ch_pos_propagates_and_collection_accessor():
    win = _windows([_raw_ds(CHS_A, 0), _raw_ds(CHS_B, 1)])
    out = win.set_return_ch_pos(True)
    assert out is win  # chainable
    assert all(ds.return_ch_pos for ds in win.datasets)
    # collection-level accessor returns the first recording's positions
    np.testing.assert_allclose(win.ch_pos, win.datasets[0].ch_pos)
    win.set_return_ch_pos(False)
    assert not any(ds.return_ch_pos for ds in win.datasets)


def test_ch_pos_cached():
    win = _windows([_raw_ds(CHS_A)])
    ds = win.datasets[0]
    assert ds.ch_pos is ds.ch_pos  # same cached array object


def test_ch_pos_misc_alignment():
    # targets_from="channels": the misc channel is dropped from X, so the
    # matching position row must be dropped too.
    ch_names = CHS_A + ["target"]
    info = mne.create_info(ch_names, 100.0, ch_types=["eeg"] * len(CHS_A) + ["misc"])
    data = np.random.RandomState(0).randn(len(ch_names), 1000) * 1e-6
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    metadata = pd.DataFrame(
        {"i_window_in_trial": [0], "i_start_in_trial": [0], "i_stop_in_trial": [200]}
    )
    ds = EEGWindowsDataset(raw, metadata, targets_from="channels")
    ds.return_ch_pos = True
    X, y, crop_inds, pos = ds[0]
    assert X.shape[0] == len(CHS_A)  # misc excluded from signal
    assert pos.shape == (len(CHS_A), 3)  # ...and from positions


def test_no_montage_warns_and_non_finite():
    win = _windows([_raw_ds(CHS_A, montage=False)])
    ds = win.datasets[0]
    with pytest.warns(UserWarning, match="no montage"):
        pos = ds.ch_pos
    has_real_pos = np.isfinite(pos).all(axis=1) & np.any(pos != 0, axis=1)
    assert not has_real_pos.any()


def test_uniform_default_collate_stacks_positions():
    win = _windows([_raw_ds(CHS_A, 0), _raw_ds(CHS_A, 1)])  # both 8 channels
    win.set_return_ch_pos(True)
    loader = DataLoader(win, batch_size=4)
    X, y, crop_inds, pos = next(iter(loader))
    assert pos.shape == (4, len(CHS_A), 3)


def test_pad_channels_collate_heterogeneous():
    win = _windows([_raw_ds(CHS_A, 0), _raw_ds(CHS_B, 1)])
    win.set_return_ch_pos(True)
    loader = DataLoader(
        win, batch_size=4, shuffle=True, collate_fn=pad_channels_collate
    )
    X, y, crop_inds, pos, ch_mask = next(iter(loader))
    max_ch = len(CHS_B)
    assert X.shape == (4, max_ch, 200)
    assert pos.shape == (4, max_ch, 3)
    assert ch_mask.shape == (4, max_ch)
    assert ch_mask.dtype == torch.bool
    # mask counts equal each sample's true channel count
    assert set(ch_mask.sum(1).tolist()) <= {len(CHS_A), len(CHS_B)}
    # padded rows are zero in both signal and positions
    for i in range(X.shape[0]):
        c = int(ch_mask[i].sum())
        assert X[i, c:].abs().sum() == 0
        assert pos[i, c:].abs().sum() == 0


def test_pad_channels_collate_without_positions():
    win = _windows([_raw_ds(CHS_A, 0), _raw_ds(CHS_B, 1)])  # no return_ch_pos
    loader = DataLoader(win, batch_size=4, collate_fn=pad_channels_collate)
    batch = next(iter(loader))
    assert len(batch) == 4  # (X, y, crop_inds, ch_mask) -- no positions
    X, y, crop_inds, ch_mask = batch
    assert ch_mask.shape == (X.shape[0], X.shape[1])


def test_pad_channels_collate_nonuniform_time_raises():
    a = np.zeros((4, 100), dtype="float32")
    b = np.zeros((6, 200), dtype="float32")
    batch = [(a, 0, [0, 0, 100]), (b, 1, [0, 0, 200])]
    with pytest.raises(ValueError, match="same number of time samples"):
        pad_channels_collate(batch)


def test_info_json_roundtrip_preserves_positions():
    # Guards the Zarr/Hub lazy path: positions come from
    # ``mne.Info.from_json_dict``; ``loc`` must survive the JSON round-trip.
    info = mne.create_info(CHS_A, 100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((len(CHS_A), 10)), info)
    raw.set_montage("standard_1020")
    before = np.asarray([ch["loc"][:3] for ch in raw.info["chs"]], dtype="float32")

    json_dict = json.loads(json.dumps(raw.info.to_json_dict()))
    restored = mne.Info.from_json_dict(_restore_nan_from_json(json_dict))
    after = np.asarray([ch["loc"][:3] for ch in restored["chs"]], dtype="float32")

    np.testing.assert_allclose(before, after)
    assert np.isfinite(after).all()

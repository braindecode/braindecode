"""Exact bounded batch-read contracts, using synthetic lazy FIF only."""

import mne
import numpy as np
import pandas as pd
import pytest
import torch

from braindecode.datasets import BaseConcatDataset, EEGWindowsDataset, base


@pytest.fixture
def lazy_windows(tmp_path):
    info = mne.create_info(["a", "b", "target"], 100, ["eeg", "eeg", "misc"])
    raw = mne.io.RawArray(np.random.RandomState(4).randn(3, 4000), info, first_samp=17)
    path = tmp_path / "synthetic-raw.fif"
    raw.save(path, buffer_size_sec=0.37, verbose=False)
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    raw.crop(tmin=0.05).pick(["b", "a", "target"])
    starts = np.arange(30) * 100
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": np.arange(30),
            "i_start_in_trial": starts,
            "i_stop_in_trial": starts + 150,
            "target": np.arange(30) % 3,
        }
    )
    windows = BaseConcatDataset([EEGWindowsDataset(raw, metadata)], coalesce_reads=True)
    return windows



def _overlap(ds):
    """150-sample windows every 25 samples: 8 consecutive ones are coalesced."""
    ds.crop_inds = ds.crop_inds.copy()
    ds.crop_inds[:, 1] = np.arange(30) * 25
    ds.crop_inds[:, 2] = ds.crop_inds[:, 1] + 150


def _trace_reads(monkeypatch):
    """Record ``(start, stop)`` of every FIF read, not of the prefetch hints."""
    calls, read = [], base._read_fif_window

    def traced(raw, start, stop, willneed=False):
        if not willneed:
            calls.append((start, stop))
        return read(raw, start, stop, willneed)

    monkeypatch.setattr(base, "_read_fif_window", traced)
    return calls


def _assert_items_equal(actual, expected):
    assert len(actual) == len(expected)
    for item, ref in zip(actual, expected):
        for a, b in zip(item, ref):
            np.testing.assert_array_equal(a, b)


def _next_draws():
    return np.random.random(), torch.rand(1).item()


@pytest.mark.parametrize("coalescing", [False, True])
@pytest.mark.parametrize("eligible", [True, False])
@pytest.mark.parametrize("targets", ["metadata", "channels"])
@pytest.mark.parametrize("last", [True, False])
@pytest.mark.parametrize("positions", [True, False])
def test_order_rng_mutation_targets(
    lazy_windows, targets, last, positions, eligible, coalescing
):
    lazy_windows.coalesce_reads = coalescing
    ds = lazy_windows.datasets[0]
    ds.targets_from, ds.last_target_only = targets, last
    ds.return_ch_pos = positions
    ds._ch_pos = np.arange(9, dtype=np.float32).reshape(3, 3)

    def mutate(x):
        x += np.random.random() + torch.rand(1).item()
        return x

    ds.transform = mutate
    lazy_windows.target_transform = lambda y: y + np.random.random()
    _overlap(ds)
    indices = list(range(8)) if eligible else [3, 0, 2, 0, -29]
    np.random.seed(11)
    torch.manual_seed(13)
    expected = [lazy_windows[i] for i in indices]
    expected_draws = _next_draws()
    np.random.seed(11)
    torch.manual_seed(13)
    actual = lazy_windows.__getitems__(indices)
    assert _next_draws() == expected_draws
    _assert_items_equal(actual, expected)
    for i, item in enumerate(actual):
        for other in actual[i + 1 :]:
            assert not np.shares_memory(item[0], other[0])


def test_sequential_overlap_reads_one_union(lazy_windows, monkeypatch):
    _overlap(lazy_windows.datasets[0])
    expected = [lazy_windows[i] for i in range(8)]
    calls = _trace_reads(monkeypatch)
    actual = lazy_windows.__getitems__(list(range(8)))
    assert calls == [(0, 325)]
    _assert_items_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        "unordered",
        "cross",
        "sequence",
        "single",
        "empty",
        "preload",
        "zarr",
        "subclass",
        "bytes",
        "ratio",
        "double",
    ],
)
def test_ineligible_batches_keep_default_reads(lazy_windows, monkeypatch, case):
    """Each case is the only reason an otherwise coalesced batch is not."""
    ds = lazy_windows.datasets[0]
    _overlap(ds)
    indices = list(range(8))
    if case == "unordered":
        indices = [0, 2, 1, 3, 4, 5, 6, 7]
    elif case == "cross":
        second = EEGWindowsDataset(ds.raw.copy(), ds.metadata.copy())
        lazy_windows = BaseConcatDataset([ds, second], coalesce_reads=True)
        indices = list(range(26, 34))
    elif case == "sequence":
        indices = [(0, 1), (2, 3)]
    elif case == "single":
        indices = [0]
    elif case == "empty":
        indices = []
    elif case == "preload":
        ds.raw.load_data()
    elif case == "zarr":
        ds._zarr_data = ds.raw.get_data()
    elif case == "subclass":

        class CustomWindows(EEGWindowsDataset):
            def __getitem__(self, index):
                x, y, crop = super().__getitem__(index)
                return x + 1, y, crop

        ds.__class__ = CustomWindows
    elif case == "bytes":
        monkeypatch.setattr(base, "_COALESCED_MAX_BYTES", 100)
    elif case == "ratio":
        ds.crop_inds[1:7, 2] = ds.crop_inds[1:7, 1] + 1  # union > 1/3 of samples
    elif case == "double":
        path = ds.raw.filenames[0].parent / "double-raw.fif"
        ds.raw.save(path, fmt="double", verbose=False)
        ds.raw = mne.io.read_raw_fif(path, verbose=False)
    expected = [lazy_windows[i] for i in indices]
    calls = _trace_reads(monkeypatch)
    actual = lazy_windows.__getitems__(indices)
    n_windows = sum(len(i) if isinstance(i, tuple) else 1 for i in indices)
    # zarr reads no FIF; a double FIF fails the union read, then reads windows
    assert len(calls) == {"zarr": 0, "double": n_windows + 1}.get(case, n_windows)
    _assert_items_equal(actual, expected)


def test_lazy_metadata_keeps_default_path(lazy_windows):
    from braindecode.datasets import RawDataset
    from braindecode.preprocessing import create_fixed_length_windows

    windows = create_fixed_length_windows(
        BaseConcatDataset([RawDataset(lazy_windows.datasets[0].raw, {"subject": 1})]),
        window_size_samples=150,
        window_stride_samples=25,
        drop_last_window=True,
        lazy_metadata=True,
    )
    windows.coalesce_reads = True
    assert type(windows) is BaseConcatDataset  # so lazy crop_inds is the reason
    assert windows._get_coalesced_items(list(range(8))) is None


def test_default_flag_and_no_candidate_work(lazy_windows, monkeypatch):
    ds = lazy_windows.datasets[0]
    _overlap(ds)
    windows = BaseConcatDataset([ds])
    assert windows.coalesce_reads is False
    disabled = BaseConcatDataset([ds], coalesce_reads=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("default must never enter candidate code")

    monkeypatch.setattr(BaseConcatDataset, "_get_coalesced_items", forbidden)
    expected = [windows[i] for i in range(8)]
    for dataset in (windows, disabled):
        _assert_items_equal(dataset.__getitems__(list(range(8))), expected)
    del windows.coalesce_reads  # old pickles may lack the new instance attribute
    assert windows.coalesce_reads is False
    windows.__getitems__(list(range(8)))


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_flag_requires_bool(lazy_windows, value):
    with pytest.raises(TypeError, match="coalesce_reads must be a bool"):
        BaseConcatDataset(lazy_windows.datasets, coalesce_reads=value)


def test_constructor_optin_and_split_default(lazy_windows):
    enabled = BaseConcatDataset(lazy_windows.datasets, coalesce_reads=True)
    assert enabled.coalesce_reads is True
    split = enabled.split([[0]])
    assert all(ds.coalesce_reads is False for ds in split.values())


# Equality matrix: the opt-in path against the default path (and the scalar
# oracle) for every data backend, window layout and DataLoader worker count.
_MATRIX_BATCHES = {
    "overlap": [list(range(8)), list(range(8, 16)), list(range(16, 24))],
    "adjacent": [list(range(8)), list(range(8, 16)), list(range(16, 24))],
    # The middle batch spans both recordings.
    "cross": [list(range(4, 12)), list(range(12, 20)), list(range(20, 28))],
}


def _matrix_dataset(tmp_path, kind, layout):
    info = mne.create_info(["a", "b", "target"], 100, ["eeg", "eeg", "misc"])
    stride = 150 if layout == "adjacent" else 25
    recordings = []
    for i in range(2):
        signal = np.random.RandomState(20 + i).randn(3, 4000)
        path = tmp_path / f"matrix-{i}-raw.fif"
        mne.io.RawArray(signal, info).save(path, buffer_size_sec=0.37, verbose=False)
        raw = mne.io.read_raw_fif(path, preload=kind == "preloaded", verbose=False)
        starts = np.arange(16) * stride
        metadata = pd.DataFrame(
            {
                "i_window_in_trial": np.arange(16),
                "i_start_in_trial": starts,
                "i_stop_in_trial": starts + 150,
                "target": np.arange(16) % 3,
            }
        )
        windows = EEGWindowsDataset(raw, metadata, description={"subject": i})
        if kind == "epochs":
            windows = windows.to_epochs_dataset()
        recordings.append(windows)
    if kind == "zarr":
        pytest.importorskip("zarr")
        from braindecode.datautil.hub_formats import convert_to_zarr, load_from_zarr

        convert_to_zarr(BaseConcatDataset(recordings), tmp_path / "matrix.zarr")
        recordings = load_from_zarr(tmp_path / "matrix.zarr", preload=False).datasets
        assert all(ds._zarr_data is not None for ds in recordings)
    return recordings


def _assert_identical(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_identical(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("workers", [0, 2, 4])
@pytest.mark.parametrize("layout", ["overlap", "adjacent", "cross"])
@pytest.mark.parametrize("kind", ["lazy", "preloaded", "epochs", "zarr"])
def test_optin_equality_matrix(tmp_path, monkeypatch, kind, layout, workers):
    from torch.utils.data import DataLoader, default_collate

    recordings = _matrix_dataset(tmp_path, kind, layout)
    off = BaseConcatDataset(recordings)
    on = BaseConcatDataset(recordings, coalesce_reads=True)
    assert off.coalesce_reads is False and on.coalesce_reads is True
    batches = _MATRIX_BATCHES[layout]
    oracle = [default_collate([off[i] for i in batch]) for batch in batches]
    for dataset in (off, on):
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=workers)
        loaded = list(loader)
        assert len(loaded) == len(oracle)
        for batch, expected in zip(loaded, oracle):
            assert batch[0].dtype == torch.float32
            _assert_identical(batch, expected)

    # Which read path ran: only lazy FIF overlap batches inside one recording
    # are read as one union; every other case reads window by window.
    calls = _trace_reads(monkeypatch)
    for dataset in (off, on):
        calls.clear()
        items = [dataset.__getitems__(batch) for batch in batches]
        for batch_items, expected in zip(items, oracle):
            _assert_identical(default_collate(batch_items), expected)
        if dataset is on and kind == "lazy" and layout == "overlap":
            assert len(calls) == len(batches)
        elif dataset is on and kind == "lazy" and layout == "cross":
            # Batches inside one recording overlap and are coalesced; the
            # batch spanning both recordings falls back to window reads.
            assert len(calls) == 2 + len(batches[1])
        elif kind in ("lazy", "preloaded"):
            # Preloaded raws are not FIF-mapped: the traced reader returns
            # None and mne's in-memory read is used for each window.
            assert len(calls) == sum(map(len, batches))
        else:
            assert calls == []

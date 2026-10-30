"""Exact bounded batch-read contracts, using synthetic lazy FIF only."""

import random

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


def test_unordered_batch_retains_scalar_reads(lazy_windows, monkeypatch):
    calls = []
    read = base._read_fif_window

    def traced(raw, start, stop, willneed=False):
        if not willneed:
            calls.append((start, stop))
        return read(raw, start, stop, willneed)

    monkeypatch.setattr(base, "_read_fif_window", traced)
    indices = [3, 1, 2, 1]
    actual = lazy_windows.__getitems__(indices)
    assert calls == [(300, 450), (100, 250), (200, 350), (100, 250)]
    for item, index in zip(actual, indices):
        expected = lazy_windows[index]
        np.testing.assert_array_equal(item[0], expected[0])
        assert item[0].dtype == np.float32
        assert item[1:] == expected[1:]
    assert not np.shares_memory(actual[1][0], actual[3][0])


def _seed():
    np.random.seed(11)
    random.seed(12)
    torch.manual_seed(13)


def _next_rng():
    return np.random.random(), random.random(), torch.rand(1).item()


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
        x += np.random.random() + random.random() + torch.rand(1).item()
        return x

    def target(y):
        return y + np.random.random()

    ds.transform = mutate
    lazy_windows.target_transform = target
    ds.crop_inds = ds.crop_inds.copy()
    ds.crop_inds[:, 1] = np.arange(30) * 25
    ds.crop_inds[:, 2] = ds.crop_inds[:, 1] + 150
    indices = list(range(8)) if eligible else [3, 0, 2, 0, -29]
    _seed()
    expected = [lazy_windows[i] for i in indices]
    state_expected = (np.random.get_state(), random.getstate(), torch.get_rng_state())
    next_expected = _next_rng()
    _seed()
    actual = lazy_windows.__getitems__(indices)
    for a, b in zip(np.random.get_state(), state_expected[0]):
        np.testing.assert_array_equal(a, b)
    assert random.getstate() == state_expected[1]
    assert torch.equal(torch.get_rng_state(), state_expected[2])
    assert _next_rng() == next_expected
    for item, ref in zip(actual, expected):
        for a, b in zip(item, ref):
            np.testing.assert_array_equal(a, b)
    for i, item in enumerate(actual):
        for other in actual[i + 1 :]:
            assert not np.shares_memory(item[0], other[0])
    saved = actual[3][0].copy()
    actual[1][0][:] = 999
    np.testing.assert_array_equal(actual[3][0], saved)


@pytest.mark.parametrize(
    "case",
    [
        "sparse",
        "gap",
        "amplification",
        "oversized",
        "cross",
        "sequence",
        "preload",
        "double",
        "subclass",
        "single",
        "empty",
        "zarr",
    ],
)
def test_unsupported_batches_keep_prefetch(lazy_windows, monkeypatch, case):
    ds = lazy_windows.datasets[0]
    indices = [0, 1, 2]
    if case == "sparse":
        indices = [0, 29]
    elif case in ("gap", "amplification"):
        starts = [0, 400] if case == "gap" else [0, 230]
        ds.crop_inds = ds.crop_inds.copy()
        ds.crop_inds[:2, 1:] = [[s, s + 150] for s in starts]
        indices = [0, 1]
    elif case == "oversized":
        monkeypatch.setattr(base, "_COALESCED_MAX_BYTES", 100)
    elif case == "cross":
        lazy_windows = BaseConcatDataset(
            [ds, EEGWindowsDataset(ds.raw.copy(), ds.metadata.copy())]
        )
        lazy_windows.coalesce_reads = True
        indices = [0, 30]
    elif case == "sequence":
        indices = [(0, 1), (2, 3)]
    elif case == "preload":
        ds.raw.load_data()
    elif case == "double":
        path = ds.raw.filenames[0].parent / "double-raw.fif"
        ds.raw.save(path, fmt="double", verbose=False)
        ds.raw = mne.io.read_raw_fif(path, verbose=False)
    elif case == "subclass":

        class CustomWindows(EEGWindowsDataset):
            def __getitem__(self, index):
                x, y, crop = super().__getitem__(index)
                return x + 1, y, crop

        ds.__class__ = CustomWindows
    elif case == "single":
        indices = [0]
    elif case == "empty":
        indices = []
    elif case == "zarr":
        ds._zarr_data = ds.raw.get_data()
    expected = [lazy_windows[i] for i in indices]
    calls = []
    read = base._read_fif_window

    def traced(raw, start, stop, willneed=False):
        calls.append((start, stop, willneed))
        return read(raw, start, stop, willneed)

    monkeypatch.setattr(base, "_read_fif_window", traced)
    actual = lazy_windows.__getitems__(indices)
    n_scalar = sum(len(i) if isinstance(i, tuple) else 1 for i in indices)
    assert sum(not c[2] for c in calls) == (0 if case == "zarr" else n_scalar)
    for item, ref in zip(actual, expected):
        for a, b in zip(item, ref):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("coalescing", [False, True])
@pytest.mark.parametrize("workers", [0, 2, 4])
def test_loader_prefetch_parity(lazy_windows, workers, coalescing):
    from torch.utils.data import DataLoader

    lazy_windows.coalesce_reads = coalescing
    ds = lazy_windows.datasets[0]
    ds.crop_inds = ds.crop_inds.copy()
    ds.crop_inds[:, 1] = np.arange(30) * 25
    ds.crop_inds[:, 2] = ds.crop_inds[:, 1] + 150
    batches = [list(range(8)), [3, 1, 2, 1], [29, 28]]
    expected = [[lazy_windows[i] for i in batch] for batch in batches]
    for (x, y, crop), ref in zip(
        DataLoader(lazy_windows, batch_sampler=batches, num_workers=workers), expected
    ):
        np.testing.assert_array_equal(x.numpy(), np.stack([item[0] for item in ref]))
        np.testing.assert_array_equal(y.numpy(), [item[1] for item in ref])
        np.testing.assert_array_equal(np.stack(crop, axis=1), [item[2] for item in ref])


def test_sequential_overlap_reads_one_union(lazy_windows, monkeypatch):
    ds = lazy_windows.datasets[0]
    ds.crop_inds = ds.crop_inds.copy()
    ds.crop_inds[:, 1] = np.arange(30) * 25
    ds.crop_inds[:, 2] = ds.crop_inds[:, 1] + 150
    calls = []
    read = base._read_fif_window

    def traced(raw, start, stop, willneed=False):
        if not willneed:
            calls.append((start, stop))
        return read(raw, start, stop, willneed)

    monkeypatch.setattr(base, "_read_fif_window", traced)
    actual = lazy_windows.__getitems__(list(range(8)))
    assert calls == [(0, 325)]
    for index, item in enumerate(actual):
        np.testing.assert_array_equal(item[0], lazy_windows[index][0])


@pytest.mark.parametrize("case", ["sparse", "cross", "adjacent"])
def test_no_union_helper_or_span_allocations(lazy_windows, monkeypatch, case):
    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible batch must not enter union reader")

    monkeypatch.setattr(base, "_coalesced_fif_batch", forbidden, raising=False)
    monkeypatch.setattr(EEGWindowsDataset, "_finish_window", forbidden, raising=False)
    ds = lazy_windows.datasets[0]
    if case == "sparse":
        indices = [0, 4, 17, 12]
    elif case == "cross":
        lazy_windows = BaseConcatDataset(
            [ds, EEGWindowsDataset(ds.raw.copy(), ds.metadata.copy())]
        )
        lazy_windows.coalesce_reads = True
        indices = list(range(28, 32))
    else:
        # Adjacent spans rejected by cheap endpoint span/byte precheck.
        ds.crop_inds = ds.crop_inds.copy()
        ds.crop_inds[:, 2] = ds.crop_inds[:, 1] + 100
        indices = list(range(8))
    actual = lazy_windows.__getitems__(indices)
    for index, item in zip(indices, actual):
        np.testing.assert_array_equal(item[0], lazy_windows[index][0])


@pytest.mark.parametrize("reason", ["bytes", "ratio", "hole", "unordered", "double"])
def test_exact_budget_and_reader_fallback(lazy_windows, monkeypatch, reason):
    ds = lazy_windows.datasets[0]
    ds.crop_inds = ds.crop_inds.copy()
    ds.crop_inds[:, 1] = np.arange(30) * 25
    ds.crop_inds[:, 2] = ds.crop_inds[:, 1] + 150
    indices = list(range(8))
    if reason == "bytes":
        monkeypatch.setattr(base, "_COALESCED_MAX_BYTES", 100)
    elif reason == "ratio":
        # Endpoint screen allows this, exact sum rejects short middle windows.
        ds.crop_inds[1:7, 2] = ds.crop_inds[1:7, 1] + 1
    elif reason == "hole":
        # Generous endpoint window defeats the cheap screen; strict hole rejects.
        ds.crop_inds[0, 2] = 100
        ds.crop_inds[1:, 1] += 200
        ds.crop_inds[1:, 2] = ds.crop_inds[1:, 1] + 150
        ds.crop_inds[0, 2] = 2000
        # Hole after first very short window, ending before the last long one.
        ds.crop_inds[0, 2] = 500
        ds.crop_inds[1:7, 1] = np.arange(6) * 2 + 650
        ds.crop_inds[1:7, 2] = ds.crop_inds[1:7, 1] + 1
        ds.crop_inds[7, 1:] = [1000, 1150]
    elif reason == "unordered":
        indices = [0, 2, 1, 3, 4, 5, 6, 7]
    elif reason == "double":
        path = ds.raw.filenames[0].parent / "double-raw-2.fif"
        ds.raw.save(path, fmt="double", verbose=False)
        ds.raw = mne.io.read_raw_fif(path, verbose=False)
    expected = [lazy_windows[i] for i in indices]
    calls = []
    read = base._read_fif_window

    def traced(raw, start, stop, willneed=False):
        if not willneed:
            calls.append((start, stop))
        return read(raw, start, stop, willneed)

    monkeypatch.setattr(base, "_read_fif_window", traced)
    actual = lazy_windows.__getitems__(indices)
    assert len(calls) == len(indices) + (reason == "double")
    for item, ref in zip(actual, expected):
        for a, b in zip(item, ref):
            np.testing.assert_array_equal(a, b)


def test_lazy_metadata_retains_scalar_path(lazy_windows, monkeypatch):
    from braindecode.datasets import RawDataset
    from braindecode.preprocessing import create_fixed_length_windows

    windows = create_fixed_length_windows(
        BaseConcatDataset([RawDataset(lazy_windows.datasets[0].raw, {"subject": 1})]),
        window_size_samples=150,
        window_stride_samples=25,
        drop_last_window=True,
        lazy_metadata=True,
    )
    indices = list(range(8))
    windows.coalesce_reads = True
    expected = [windows[i] for i in indices]

    def forbidden(*args, **kwargs):
        raise AssertionError("lazy metadata must use the exact scalar path")

    monkeypatch.setattr(base, "_coalesced_fif_batch", forbidden, raising=False)
    monkeypatch.setattr(EEGWindowsDataset, "_finish_window", forbidden, raising=False)
    actual = windows.__getitems__(indices)
    for item, ref in zip(actual, expected):
        for a, b in zip(item, ref):
            np.testing.assert_array_equal(a, b)


_BASELINE_PREFETCH_TAIL = """        # ponytail: advises the whole batch at once; chunk it if a batch's
        # windows ever outgrow the page cache
        if hasattr(os, "posix_fadvise"):
            for idx in indices:
                if not isinstance(idx, (int, np.integer)):
                    continue  # e.g. a SequenceSampler tuple: read, not advised
                ds, i = self._locate(idx)
                if isinstance(ds, EEGWindowsDataset) and ds._zarr_data is None:
                    _, start, stop = ds.crop_inds[i].tolist()
                    _read_fif_window(ds.raw, start, stop, willneed=True)
        return [self[idx] for idx in indices]
"""


def test_default_source_identity():
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(BaseConcatDataset.__getitems__)
    tail = source[source.index("        # ponytail:") :]
    assert tail == _BASELINE_PREFETCH_TAIL
    body = ast.parse(textwrap.dedent(source)).body[0].body[1:]
    assert isinstance(body[0], ast.If)
    assert ast.unparse(body[0].test) == "self.coalesce_reads"
    # Only one flag test before precisely the baseline AST statements.
    original = ast.parse(textwrap.dedent(_BASELINE_PREFETCH_TAIL)).body
    assert [ast.dump(x) for x in body[1:]] == [ast.dump(x) for x in original]


def test_default_flag_and_no_candidate_work(lazy_windows, monkeypatch):
    ds = lazy_windows.datasets[0]
    windows = BaseConcatDataset([ds])
    assert windows.coalesce_reads is False
    disabled = BaseConcatDataset([ds], coalesce_reads=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("default must never enter candidate code")

    monkeypatch.setattr(BaseConcatDataset, "_get_coalesced_items", forbidden)
    monkeypatch.setattr(base, "_coalesced_fif_batch", forbidden)
    monkeypatch.setattr(EEGWindowsDataset, "_finish_window", forbidden)
    expected = [windows[i] for i in range(8)]
    for dataset in (windows, disabled):
        for item, ref in zip(dataset.__getitems__(list(range(8))), expected):
            for a, b in zip(item, ref):
                np.testing.assert_array_equal(a, b)
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
    calls = []
    read = base._read_fif_window

    def traced(raw, start, stop, willneed=False):
        if not willneed:
            calls.append((start, stop))
        return read(raw, start, stop, willneed)

    monkeypatch.setattr(base, "_read_fif_window", traced)
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

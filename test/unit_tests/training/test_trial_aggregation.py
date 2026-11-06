# License: BSD-3
"""Tests for the private streaming trial reconstruction primitive.

The oracle is a frozen copy of ``trial_preds_from_window_preds`` as it was at
d9340316 (before it delegated to ``iter_trial_preds``). Every reconstructed
trial must be bit-identical to it, and the streaming iterator must hold at
most the windows of the trial in progress plus one boundary lookahead.
"""

import gc
import tracemalloc
import weakref

import numpy as np
import pytest
import torch

from braindecode.training._trial_aggregation import iter_trial_preds
from braindecode.training.scoring import trial_preds_from_window_preds


def _reference_trial_preds(preds, i_window_in_trials, i_stop_in_trials):
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


def _window_stops(length, n_preds, stride):
    stops = list(range(n_preds, length + 1, stride))
    if stops[-1] != length:
        # last window aligned to the trial end: duplicate time overlap
        stops.append(length)
    return stops


def _make_case(trial_lengths, n_preds, stride, n_classes=3, dtype=np.float64, seed=0):
    """Windows cut from known per-trial signals, in chronological order."""
    rng = np.random.default_rng(seed)
    signals, preds, i_windows, i_stops = [], [], [], []
    for length in trial_lengths:
        assert length >= n_preds
        signal = (rng.standard_normal((n_classes, length)) * 100).astype(dtype)
        signals.append(signal)
        for i_window, stop in enumerate(_window_stops(length, n_preds, stride)):
            preds.append(signal[:, stop - n_preds : stop].copy())
            i_windows.append(i_window)
            i_stops.append(stop)
    return signals, preds, i_windows, i_stops


_LAYOUTS = {
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


def _batched_rows(preds, batch_size):
    """Yield windows as row views of batches, so trials cross batch boundaries."""
    for start in range(0, len(preds), batch_size):
        batch = np.stack(preds[start : start + batch_size])
        yield from batch


def _containers(preds, i_windows, i_stops):
    """The input shapes used by braindecode callers and by a streaming caller."""
    return {
        "list_of_arrays": (list(preds), list(i_windows), list(i_stops)),
        "nested_lists": ([p.tolist() for p in preds], i_windows, i_stops),
        "stacked_array": (np.stack(preds), np.array(i_windows), np.array(i_stops)),
        # as predict_trials: float32 rows, torch tensor indices
        "torch_indices": (
            list(np.stack(preds).astype(np.float32)),
            torch.tensor(i_windows),
            torch.tensor(i_stops),
        ),
        "batched_generator": (
            _batched_rows(preds, batch_size=7),
            iter(i_windows),
            iter(i_stops),
        ),
    }


def _assert_bit_identical(actual_trials, reference_trials):
    assert len(actual_trials) == len(reference_trials)
    for actual, reference in zip(actual_trials, reference_trials):
        assert type(actual) is np.ndarray
        np.testing.assert_array_equal(actual, reference)
        assert actual.dtype == reference.dtype
        assert actual.shape == reference.shape
        assert actual.tobytes() == reference.tobytes()
        for flag in ("C_CONTIGUOUS", "F_CONTIGUOUS", "OWNDATA", "WRITEABLE"):
            assert actual.flags[flag] == reference.flags[flag], flag


@pytest.mark.parametrize(
    "container",
    [
        "list_of_arrays",
        "nested_lists",
        "stacked_array",
        "torch_indices",
        "batched_generator",
    ],
)
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_streaming_trial_preds_matches_reference(layout, container):
    signals, preds, i_windows, i_stops = _make_case(**_LAYOUTS[layout])
    oracle_inputs = _containers(preds, i_windows, i_stops)[container]
    if container == "batched_generator":
        oracle_inputs = (list(preds), i_windows, i_stops)
    reference = _reference_trial_preds(*oracle_inputs)
    assert len(reference) == len(signals)

    actual = list(iter_trial_preds(*_containers(preds, i_windows, i_stops)[container]))
    _assert_bit_identical(actual, reference)
    for trial, signal in zip(actual, signals):
        # sanity of the fixture: overlap removal recovers each trial's signal
        np.testing.assert_array_equal(trial, signal.astype(trial.dtype))

    if container != "batched_generator":
        wrapped = trial_preds_from_window_preds(
            *_containers(preds, i_windows, i_stops)[container]
        )
        assert type(wrapped) is list
        _assert_bit_identical(wrapped, reference)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_streaming_matches_reference_dtypes_and_special_values(dtype):
    _, preds, i_windows, i_stops = _make_case(
        [23, 10, 31], n_preds=10, stride=6, dtype=dtype, seed=3
    )
    if np.issubdtype(dtype, np.floating):
        # NaN sequence targets / non-finite logits must pass through untouched
        preds[0][0, 3] = np.nan
        preds[1][1, -1] = np.inf
        preds[4][2, 0] = -0.0
        preds[-1][:, -2:] = np.nan
    reference = _reference_trial_preds(preds, i_windows, i_stops)
    _assert_bit_identical(list(iter_trial_preds(preds, i_windows, i_stops)), reference)
    _assert_bit_identical(
        trial_preds_from_window_preds(preds, i_windows, i_stops), reference
    )


_QUIRKY = {
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


@pytest.mark.parametrize("case", list(_QUIRKY))
def test_streaming_matches_reference_on_quirky_stops(case):
    preds, i_windows, i_stops = _QUIRKY[case]
    reference = _reference_trial_preds(preds, i_windows, i_stops)
    _assert_bit_identical(list(iter_trial_preds(preds, i_windows, i_stops)), reference)
    _assert_bit_identical(
        trial_preds_from_window_preds(preds, i_windows, i_stops), reference
    )


_INVALID = {
    "empty": ([], [], []),
    "first_window_not_zero": ([[[1, 2]], [[3, 4]]], [1, 2], [2, 3]),
    "window_index_gap": ([[[1, 2]], [[3, 4]], [[5, 6]]], [0, 1, 3], [2, 3, 4]),
    "one_dim_windows_with_overlap": ([np.arange(4), np.arange(4)], [0, 1], [4, 6]),
    "class_count_changes_within_trial": (
        [np.zeros((2, 4)), np.zeros((3, 4))],
        [0, 1],
        [4, 6],
    ),
}


def _raised(func, *args):
    # Only the first message line: pytest rewrites asserts in this test module
    # (the oracle) and appends introspection that production asserts lack.
    with pytest.raises(Exception) as excinfo:
        func(*args)
    return type(excinfo.value), str(excinfo.value).split("\n")[0]


@pytest.mark.parametrize("case", list(_INVALID))
def test_invalid_inputs_raise_as_before(case):
    args = _INVALID[case]
    expected = _raised(_reference_trial_preds, *args)
    assert _raised(trial_preds_from_window_preds, *args) == expected
    assert _raised(lambda *a: list(iter_trial_preds(*a)), *args) == expected


def test_wrapper_keeps_length_assertion():
    args = ([[[1, 2]], [[3, 4]], [[5, 6]]], [0, 1], [2, 3, 4])
    expected = _raised(_reference_trial_preds, *args)
    assert expected == (AssertionError, "3, 2, 3")
    assert _raised(trial_preds_from_window_preds, *args) == expected


def test_streaming_rejects_length_mismatch():
    # The iterator cannot call len(); it must not silently truncate either.
    with pytest.raises(ValueError):
        list(
            iter_trial_preds(
                iter([[[1, 2]], [[3, 4]], [[5, 6]]]), iter([0, 1]), [2, 3, 4]
            )
        )


def test_streaming_is_lazy_and_errors_after_completed_trials():
    it = iter_trial_preds([], [], [])  # no work, no error yet
    assert iter(it) is it
    with pytest.raises(ValueError, match="need at least one array"):
        next(it)

    # trials completed before an invalid window are still delivered, in order
    preds = [[[1, 2]], [[2, 3]], [[7, 8]], [[8, 9]], [[0, 1]]]
    i_windows, i_stops = [0, 1, 0, 1, 3], [2, 3, 2, 3, 5]
    expected = _reference_trial_preds(preds[:3], i_windows[:3], i_stops[:3])[:1]
    it = iter_trial_preds(preds, i_windows, i_stops)
    _assert_bit_identical([next(it)], expected)
    with pytest.raises(AssertionError, match="should start from 0"):
        next(it)


class _CountingIterable:
    """Single-pass source that records how many items were pulled."""

    def __init__(self, items):
        self._items = items
        self.n_consumed = 0

    def __iter__(self):
        for item in self._items:
            self.n_consumed += 1
            yield item


@pytest.mark.parametrize(
    "trial_lengths",
    [[40, 23, 31], [10, 40, 12], [57, 10]],
    ids=["long_first", "one_window_first", "two"],
)
def test_streaming_does_not_consume_all_trials(trial_lengths):
    _, preds, i_windows, i_stops = _make_case(trial_lengths, n_preds=10, stride=7)
    reference = _reference_trial_preds(preds, i_windows, i_stops)
    n_windows = np.bincount(np.cumsum(np.array(i_windows) == 0) - 1)
    sources = [_CountingIterable(x) for x in (preds, i_windows, i_stops)]
    it = iter_trial_preds(*sources)
    for i_trial in range(len(trial_lengths) - 1):
        _assert_bit_identical([next(it)], [reference[i_trial]])
        # windows of completed trials plus exactly one boundary lookahead
        expected = int(n_windows[: i_trial + 1].sum()) + 1
        assert [s.n_consumed for s in sources] == [expected] * 3
    _assert_bit_identical([next(it)], [reference[-1]])
    assert [s.n_consumed for s in sources] == [len(preds)] * 3
    with pytest.raises(StopIteration):
        next(it)


def test_streaming_early_close_stops_consumption():
    _, preds, i_windows, i_stops = _make_case([40, 23, 31], n_preds=10, stride=7)
    sources = [_CountingIterable(x) for x in (preds, i_windows, i_stops)]
    it = iter_trial_preds(*sources)
    next(it)
    consumed = [s.n_consumed for s in sources]
    it.close()
    with pytest.raises(StopIteration):
        next(it)
    assert [s.n_consumed for s in sources] == consumed


def _iterator_owned_arrays(it):
    """Root buffers of all arrays held in the iterator's (generator) frames.

    Follows lists/tuples/dicts and ``ndarray.base`` chains, never the caller's
    source iterators, so it counts exactly what the iterator itself retains.
    """
    frames, gen = [], it
    while gen is not None and getattr(gen, "gi_frame", None) is not None:
        frames.append(gen.gi_frame)
        gen = gen.gi_yieldfrom
    assert frames, "iter_trial_preds must return a suspended generator"
    roots, seen = {}, set()
    stack = [value for frame in frames for value in frame.f_locals.values()]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            root = obj
            while isinstance(root.base, np.ndarray):
                root = root.base
            roots[id(root)] = root
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
    return list(roots.values())


def test_streaming_retains_no_completed_trial_storage():
    n_classes, n_preds = 3, 10
    # trial 0 has 10 windows (batches 0, 1 and part of 2), then two more trials
    _, preds, i_windows, i_stops = _make_case(
        [10 + 9 * 5, 23, 31], n_preds=n_preds, stride=5, n_classes=n_classes
    )
    assert i_windows[:11] == list(range(10)) + [0]
    reference = _reference_trial_preds(preds, i_windows, i_stops)
    window_nbytes = preds[0].nbytes
    batch_refs = []

    def lazy_batches(batch_size=4):
        for start in range(0, len(preds), batch_size):
            batch = np.stack(preds[start : start + batch_size])
            batch_refs.append(weakref.ref(batch))
            yield from batch  # row views keep their batch alive

    it = iter_trial_preds(lazy_batches(), iter(i_windows), iter(i_stops))
    for i_trial, expected in enumerate(reference):
        trial = next(it)
        _assert_bit_identical([trial], [expected])
        gc.collect()
        owned = _iterator_owned_arrays(it)
        assert not any(np.shares_memory(trial, a) for a in owned)
        owned_nbytes = sum(a.nbytes for a in owned)
        # at most one copied window: the lookahead (first window of the next
        # trial) or, after the last trial, the loop variable of its last window
        assert owned_nbytes <= window_nbytes, (i_trial, owned_nbytes)
        if i_trial == 0:
            # batches holding only trial-0 windows were released
            assert batch_refs[0]() is None and batch_refs[1]() is None
        del trial


def test_streaming_completed_trial_storage_is_bounded_over_long_sequences():
    n_trials, n_classes, n_preds, stride, length = 200, 4, 64, 32, 256
    stops = _window_stops(length, n_preds, stride)
    i_windows = list(range(len(stops))) * n_trials
    i_stops = stops * n_trials
    window_nbytes = n_classes * n_preds * 8
    trial_nbytes = n_classes * length * 8
    one_trial_bound = len(stops) * window_nbytes + trial_nbytes

    def lazy_windows():
        for i_trial in range(n_trials):
            for stop in stops:
                window = np.empty((n_classes, n_preds))
                window[:] = np.arange(stop - n_preds, stop) + length * i_trial
                yield window

    def measure(consume):
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        try:
            gc.collect()
            tracemalloc.reset_peak()
            start = tracemalloc.get_traced_memory()[0]
            result = consume()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not was_tracing:
                tracemalloc.stop()
        return peak - start, result

    def stream():
        n_seen = 0
        for i_trial, trial in enumerate(
            iter_trial_preds(lazy_windows(), i_windows, i_stops)
        ):
            assert trial.shape == (n_classes, length)
            assert (
                trial[0, 0] == length * i_trial
                and trial[-1, -1] == length * (i_trial + 1) - 1
            )
            n_seen += 1
        return n_seen

    streamed_peak, n_seen = measure(stream)
    assert n_seen == n_trials
    assert streamed_peak <= 3 * one_trial_bound + 64 * 1024, streamed_peak

    materialized_input = list(lazy_windows())
    materialized_peak, trials = measure(
        lambda: trial_preds_from_window_preds(materialized_input, i_windows, i_stops)
    )
    # the public wrapper necessarily stores every reconstructed trial
    assert len(trials) == n_trials
    assert materialized_peak >= n_trials * trial_nbytes
    assert streamed_peak * 10 < materialized_peak

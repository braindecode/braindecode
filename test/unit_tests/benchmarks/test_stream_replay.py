"""Exact online window replay contract for ``bench_stream_replay``.

The offline reference is Braindecode's own windowing
(``create_fixed_length_windows`` read through ``EEGWindowsDataset``), with the
event windower's index computation as a cross-check. These are correctness
checks on tiny synthetic data, not performance claims.
"""

import importlib
import json
import tracemalloc

import mne
import numpy as np
import pytest
import torch

from braindecode.datasets.base import BaseConcatDataset, RawDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.preprocessing.windowers import _compute_window_inds

SIZE_STRIDE = [(16, 16), (16, 5), (16, 1), (16, 20), (7, 3), (1, 1)]


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Restore caller RNG, thread count and grad mode after each case."""
    yield


def _bench():
    # Imported inside each test so a missing helper is a test failure.
    return importlib.import_module("benchmarks.acceleration.bench_stream_replay")


def _offline(data, size, stride, drop_last_window):
    """Windows, starts and stops from Braindecode's offline windowing."""
    info = mne.create_info(data.shape[0], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose="error")
    windows = create_fixed_length_windows(
        BaseConcatDataset([RawDataset(raw, description={"trial": 0})]),
        window_size_samples=size,
        window_stride_samples=stride,
        on_last_window="drop" if drop_last_window else "overlap",
    )
    out = [windows[i] for i in range(len(windows))]
    return [(X, inds[1], inds[2]) for X, _, inds in out]


def _event_indices(n_times, size, stride, drop_last_window):
    _, _, starts, stops = _compute_window_inds(
        np.array([0]), np.array([n_times]), 0, 0, size, stride, drop_last_window, 0
    )
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _split(data, sizes):
    """Chronological views of ``data`` with the given chunk lengths."""
    chunks, pos = [], 0
    for n in sizes:
        chunks.append(data[:, pos : pos + n])
        pos += n
    chunks.append(data[:, pos:])
    return chunks


def _chunk_sizes(mode, n_times, size, stride, rng):
    if mode == "single":
        return [1] * n_times
    if mode == "whole":
        return []
    if mode == "stride":
        return [stride] * (n_times // stride)
    sizes, total = [], 0
    while total < n_times:
        n = int(rng.integers(0, 3 * size + 2))  # includes empty chunks
        sizes.append(n)
        total += n
    return sizes


class _Feed:
    """Chunk iterator that records how many samples were handed out."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.received = 0
        self.before_last = 0
        self.exhausted = False

    def __iter__(self):
        for chunk in self.chunks:
            self.before_last = self.received
            self.received += chunk.shape[1]
            yield chunk
        self.exhausted = True


def _lengths(size, stride):
    exact_end = size + 3 * stride  # last regular window ends exactly at trial end
    return sorted({size, size + 1, exact_end, exact_end + 1, 4 * size + 3})


@pytest.mark.parametrize("drop_last_window", [True, False])
@pytest.mark.parametrize(("size", "stride"), SIZE_STRIDE)
def test_stream_replay_matches_offline_windowing(size, stride, drop_last_window):
    bench = _bench()
    rng = np.random.default_rng([size, stride, int(drop_last_window)])
    for n_times in _lengths(size, stride):
        data = rng.standard_normal((3, n_times)).astype(np.float32)
        reference = _offline(data, size, stride, drop_last_window)
        assert [(s, e) for _, s, e in reference] == _event_indices(
            n_times, size, stride, drop_last_window
        )
        for mode in ("single", "random", "stride", "whole"):
            case = f"n={n_times} mode={mode}"
            feed = _Feed(_split(data, _chunk_sizes(mode, n_times, size, stride, rng)))
            streamed, after_flush = [], []
            for window, start, stop in bench.iter_completed_windows(
                feed,
                window_size=size,
                stride=stride,
                drop_last_window=drop_last_window,
            ):
                assert stop <= feed.received, case
                if feed.exhausted:
                    after_flush.append((start, stop))
                else:
                    # Emitted as soon as the completing chunk arrives.
                    assert stop > feed.before_last, case
                streamed.append((window, start, stop))
            assert [(s, e) for _, s, e in streamed] == [
                (s, e) for _, s, e in reference
            ], case
            for (got, _, _), (want, _, _) in zip(streamed, reference):
                assert got.dtype == want.dtype == np.float32
                np.testing.assert_array_equal(got, want, err_msg=case)
            regular_end = (n_times - size) % stride == 0
            if drop_last_window or regular_end:
                # No tail window; an exact end window is never duplicated.
                assert after_flush == [], case
            else:
                assert after_flush == [(n_times - size, n_times)], case
            assert len({(s, e) for _, s, e in streamed}) == len(streamed), case


@pytest.mark.parametrize("drop_last_window", [True, False])
def test_short_trial_fails_like_offline_after_flush(drop_last_window):
    bench = _bench()
    data = np.ones((2, 15), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds trial duration"):
        _offline(data, 16, 4, drop_last_window)
    feed = _Feed(_split(data, [1] * 15))
    stream = bench.iter_completed_windows(
        feed, window_size=16, stride=4, drop_last_window=drop_last_window
    )
    with pytest.raises(ValueError, match="shorter than window_size"):
        next(stream)
    assert feed.exhausted  # raised only at end-of-trial flush
    with pytest.raises(ValueError, match="shorter than window_size"):
        list(
            bench.iter_completed_windows(
                iter([]), window_size=4, stride=1, drop_last_window=drop_last_window
            )
        )


def test_stream_reset_and_availability():
    bench = _bench()
    rng = np.random.default_rng(0)
    first = rng.uniform(1, 2, (4, 101)).astype(np.float32)
    second = -rng.uniform(1, 2, (4, 77)).astype(np.float32)
    kwargs = {"window_size": 16, "stride": 5, "drop_last_window": False}

    # Abandon a call mid-trial; the next call must not see its samples.
    partial = bench.iter_completed_windows(_split(first, [9, 13, 30]), **kwargs)
    next(partial), next(partial)
    partial.close()
    got = list(bench.iter_completed_windows(_split(second, [3, 1, 40]), **kwargs))
    want = _offline(second, 16, 5, False)
    assert [(s, e) for _, s, e in got] == [(s, e) for _, s, e in want]
    for (window, _, _), (ref, _, _) in zip(got, want):
        np.testing.assert_array_equal(window, ref)
        assert (window < 0).all()

    # Two live calls interleaved sample-by-sample keep independent state.
    streams = [
        bench.iter_completed_windows(_split(x, [1] * x.shape[1]), **kwargs)
        for x in (first, second)
    ]
    results = [[], []]
    active = [0, 1]
    while active:
        for i in list(active):
            try:
                results[i].append(next(streams[i]))
            except StopIteration:
                active.remove(i)
    for data, result in zip((first, second), results):
        want = _offline(data, 16, 5, False)
        assert [(s, e) for _, s, e in result] == [(s, e) for _, s, e in want]
        for (window, _, _), (ref, _, _) in zip(result, want):
            np.testing.assert_array_equal(window, ref)


@pytest.mark.parametrize("drop_last_window", [True, False])
@pytest.mark.parametrize(("size", "stride"), [(16, 5), (16, 40), (8, 8)])
def test_buffered_samples_bounded_by_window(size, stride, drop_last_window):
    bench = _bench()
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, 50 * size + 3)).astype(np.float32)
    scheduler = bench.StreamWindowScheduler(
        window_size=size, stride=stride, drop_last_window=drop_last_window
    )
    sizes = _chunk_sizes("random", data.shape[1], size, stride, rng)
    received, n_windows = 0, 0
    for chunk in _split(data, sizes):
        for _, _, stop in scheduler.push(chunk):
            n_windows += 1
            assert stop <= received + chunk.shape[1]
        received += chunk.shape[1]
        assert scheduler.received_samples == received
        limit = size if not drop_last_window else size - 1
        assert scheduler.buffered_samples <= limit
        assert scheduler.buffer_capacity <= 2 * size
    n_windows += len(scheduler.flush())
    assert n_windows == len(_offline(data, size, stride, drop_last_window))
    with pytest.raises(RuntimeError, match="flushed"):
        list(scheduler.push(data[:, :1]))


def _peak_bytes(bench, n_times):
    data = np.random.default_rng(2).standard_normal((4, n_times)).astype(np.float32)
    chunks = _split(data, [7] * (n_times // 7))
    tracemalloc.start()
    try:
        for _ in bench.iter_completed_windows(
            iter(chunks), window_size=64, stride=3, drop_last_window=False
        ):
            pass
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_stream_memory_does_not_grow_with_trial_length():
    bench = _bench()
    short = _peak_bytes(bench, 64 * 100)
    long = _peak_bytes(bench, 64 * 1000)  # 1 MiB of float32 samples
    assert long < 1.5 * short + 16 * 1024
    assert long < 256 * 1024


def test_windows_are_owned_and_inputs_untouched():
    bench = _bench()
    rng = np.random.default_rng(3)
    data = rng.standard_normal((3, 90)).astype(np.float32)
    data.flags.writeable = False
    before = data.copy()
    want = _offline(before, 16, 3, False)
    got = []
    sizes = _chunk_sizes("random", 90, 16, 3, rng)
    for window, start, stop in bench.iter_completed_windows(
        _split(data, sizes), window_size=16, stride=3, drop_last_window=False
    ):
        assert window.flags.owndata and window.flags.writeable
        assert window.flags.c_contiguous and window.shape == (3, 16)
        assert not np.shares_memory(window, data)
        got.append((window.copy(), start, stop))
        window.fill(np.nan)  # consumer mutation must not reach later windows
    np.testing.assert_array_equal(data, before)
    assert [(s, e) for _, s, e in got] == [(s, e) for _, s, e in want]
    for (window, _, _), (ref, _, _) in zip(got, want):
        np.testing.assert_array_equal(window, ref)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"window_size": 0, "stride": 1}, ValueError, "window_size"),
        ({"window_size": -4, "stride": 1}, ValueError, "window_size"),
        ({"window_size": 4.0, "stride": 1}, TypeError, "window_size"),
        ({"window_size": True, "stride": 1}, TypeError, "window_size"),
        ({"window_size": None, "stride": 1}, TypeError, "window_size"),
        ({"window_size": 4, "stride": 0}, ValueError, "stride"),
        ({"window_size": 4, "stride": -1}, ValueError, "stride"),
        ({"window_size": 4, "stride": 1.5}, TypeError, "stride"),
        ({"window_size": 4, "stride": 1, "drop_last_window": None}, TypeError, "drop"),
        ({"window_size": 4, "stride": 1, "drop_last_window": 1}, TypeError, "drop"),
    ],
)
def test_invalid_window_arguments_fail_at_call(kwargs, error, match):
    bench = _bench()
    kwargs.setdefault("drop_last_window", False)
    with pytest.raises(error, match=match):
        bench.iter_completed_windows(iter([]), **kwargs)  # not iterated


INVALID_CHUNKS = {
    "one_dim": ([(8,)], np.float32, ValueError, "2-D"),
    "three_dim": ([(1, 2, 8)], np.float32, ValueError, "2-D"),
    "no_channels": ([(0, 8)], np.float32, ValueError, "channel"),
    "channel_change": ([(2, 3), (3, 8)], np.float32, ValueError, "channel"),
    "dtype_change": ([(2, 3), (2, 8)], [np.float32, np.float64], TypeError, "dtype"),
    "integer": ([(2, 8)], np.int64, TypeError, "floating"),
    "not_array": ([(2, 8)], list, TypeError, "ndarray"),
}


@pytest.mark.parametrize("case", sorted(INVALID_CHUNKS))
def test_invalid_chunks_fail_explicitly(case):
    bench = _bench()
    shapes, dtypes, error, match = INVALID_CHUNKS[case]
    if not isinstance(dtypes, list):
        dtypes = [dtypes] * len(shapes)
    chunks = [
        np.zeros(shape, np.float32).tolist()
        if dtype is list
        else np.zeros(shape, dtype)
        for shape, dtype in zip(shapes, dtypes)
    ]
    stream = bench.iter_completed_windows(
        iter(chunks), window_size=4, stride=4, drop_last_window=False
    )
    emitted = []
    with pytest.raises(error, match=match):
        for item in stream:
            emitted.append(item[1:])
    # Nothing from the offending chunk is ever emitted.
    assert all(stop <= 3 for _, stop in emitted)


def _model(n_chans, n_times):
    from braindecode.models import ShallowFBCSPNet

    torch.manual_seed(0)
    model = ShallowFBCSPNet(n_chans=n_chans, n_outputs=3, n_times=n_times, sfreq=100.0)
    return model.eval()


def test_model_replay_matches_offline_outputs():
    bench = _bench()
    rng = np.random.default_rng(4)
    size, stride = 128, 37
    data = rng.standard_normal((4, 3 * size + 11)).astype(np.float32)
    model = _model(4, size)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    sizes = _chunk_sizes("random", data.shape[1], size, stride, rng)
    replay = bench.replay_trial(
        model,
        _split(data, sizes),
        window_size=size,
        stride=stride,
        drop_last_window=False,
    )
    reference = _offline(data, size, stride, False)
    assert [(r["start"], r["stop"]) for r in replay] == [
        (s, e) for _, s, e in reference
    ]
    with torch.no_grad():
        offline = [model(torch.from_numpy(X)[None])[0] for X, _, _ in reference]
        batched = model(torch.from_numpy(np.stack([X for X, _, _ in reference])))
    for record, want, want_batched in zip(replay, offline, batched):
        assert record["available_samples"] >= record["stop"]
        assert torch.equal(record["output"], want)  # same bytes in, batch of one
        torch.testing.assert_close(record["output"], want_batched)
        assert record["model_s"] > 0 and record["latency_s"] >= record["model_s"]
    assert replay[-1]["after_flush"] and not any(r["after_flush"] for r in replay[:-1])
    assert not model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])

    with pytest.raises(ValueError, match="eval"):
        bench.replay_trial(
            model.train(),
            _split(data, sizes),
            window_size=size,
            stride=stride,
            drop_last_window=False,
        )


def test_benchmark_cli_reports_latency_separately(tmp_path):
    bench = _bench()
    output = tmp_path / "replay.json"
    bench.main(
        [
            "--n-chans",
            "4",
            "--sfreq",
            "100",
            "--window-size",
            "128",
            "--stride",
            "32",
            "--chunk-samples",
            "10",
            "--n-trials",
            "2",
            "--trial-samples",
            "300",
            "400",
            "--warmup-min-calls",
            "3",
            "--warmup-min-s",
            "0",
            "--warmup-max-s",
            "1",
            "--output",
            str(output),
        ]
    )
    record = json.loads(output.read_text())
    assert record["status"] == "ok"
    assert record["correctness"]["status"] == "passed"
    metrics = record["correctness"]["metrics"]
    assert metrics["window_mismatches"] == 0 and metrics["output_mismatches"] == 0
    assert metrics["random_chunk_window_mismatches"] == 0
    assert metrics["model_state_unchanged"] is True
    intrinsic = record["intrinsic_delay"]
    assert intrinsic["window_size_samples"] == 128
    assert intrinsic["first_output_available_samples"] >= 128
    assert intrinsic["tail_window_requires_flush"] is True
    compute = record["compute_latency"]
    for key in ("latency_s", "model_s"):
        assert set(compute[key]) == {"p50", "p95", "p99", "n"}
        assert 0 < compute[key]["p50"] <= compute[key]["p95"] <= compute[key]["p99"]
    assert compute["first_output_s"] > 0 and compute["cold_first_forward_s"] > 0
    notes = " ".join(record["notes"])
    assert "standardization" in notes and "non-causal" in notes


def _summary_record(commit="a" * 40, status="ok", host="node1", p50=1.0):
    return {
        "schema": "stream-replay-v1",
        "status": status,
        "warmup": {"steady": True},
        "source": {"commit": commit, "dirty": False},
        "environment": {"hostname": host, "physical_cores": 2,
                        "requested_threads": 1, "torch_threads": 1},
        "config": {"window_size": 1000},
        "compute_latency": {
            "raw_latency_s": [p50] * 10,
            "raw_model_s": [p50] * 10,
            "latency_s": {"p50": p50, "p95": p50, "p99": p50, "n": 10},
            "model_s": {"p50": p50, "p95": p50, "p99": p50, "n": 10},
            "first_output_s": p50,
        },
    }


def test_summary_requires_complete_consistent_runs():
    bench = _bench()
    runs = [_summary_record(p50=v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    summary = bench.summarize_runs(runs, expected_runs=5)
    assert summary["failures"] == []
    assert summary["latency_s"]["p50"]["median"] == 3.0
    for bad in (
        runs[:4],
        runs[:4] + [_summary_record(status="failed")],
        runs[:4] + [_summary_record(commit="b" * 40)],
        runs[:4] + [_summary_record(host="node2")],
    ):
        result = bench.summarize_runs(bad, expected_runs=5)
        assert result["failures"] and result["latency_s"] is None

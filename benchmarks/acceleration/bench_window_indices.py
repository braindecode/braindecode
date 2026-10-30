"""Paired local helper and complete event-window-construction measurements.

Run after tests/edits stop, from a clean checkout::

    python -m benchmarks.acceleration.bench_window_indices --output /tmp/indices

Each pair uses fresh, serial reference/candidate processes, alternating order.
Production retains the scalar default: no dispatch threshold is qualified.
The candidate here is experimental and only measured on ordinary safe indices;
its scalar overflow diagnostics are NOT qualified for general use.
The reference helper is extracted from the actual pinned git baseline (required
for this explicit integration benchmark, not ordinary unit tests). Both
implementations use identical current pipeline code and synthetic annotations.
Construction includes event extraction, index generation, metadata and dataset
construction, but not Raw creation, disk I/O, window traversal or training.
Every window's data/target/crop and the complete metadata frame are checked
outside timing. CPU allocation tracing is a separate untimed call; it is not
RSS or a bound on untraced native allocations. Process RSS includes imports,
setup, correctness and timing. OS cache and other host load are uncontrolled.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import platform
import resource
import subprocess
import sys
import tracemalloc
import warnings
from contextlib import contextmanager
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

import braindecode
from benchmarks.acceleration._common import measure_step, write_result
from braindecode.datasets.base import BaseConcatDataset, RawDataset
from braindecode.preprocessing import windowers

BASELINE = "9d48acf211744aad33dac72aaf9785e9d96f05f2"
ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "few": {"trials": 2, "duration": 17, "size": 16, "stride": 16},
    "large_overlap": {"trials": 64, "duration": 4096, "size": 256, "stride": 3},
}


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _reference():
    source = _git("show", f"{BASELINE}:braindecode/preprocessing/windowers.py")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_window_inds"
    )
    namespace = {"np": np, "warnings": warnings}
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), BASELINE, "exec"),
        namespace,
    )
    return namespace["_compute_window_inds"], hashlib.sha256(
        ast.get_source_segment(source, function).encode()
    ).hexdigest()


def _experimental_window_inds(
    starts,
    stops,
    start_offset,
    stop_offset,
    size,
    stride,
    drop_last_window,
    accepted_bads_ratio,
):
    """Benchmark-only vectorization; not eligible for production dispatch.

    Create window starts from trial onsets (shifted by start_offset) to trial
    end (shifted by stop_offset) separated by stride, as long as window size
    fits into trial.

    Parameters
    ----------
    starts: array-like
        Trial starts in samples.
    stops: array-like
        Trial stops in samples.
    start_offset: int
        Start offset from original trial onsets in samples.
    stop_offset: int
        Stop offset from original trial stop in samples.
    size: int
        Window size.
    stride: int
        Stride between windows.
    drop_last_window: bool
        Toggles of shifting last window within range or dropping last samples.
    accepted_bads_ratio: float
        Acceptable proportion of bad trials within a raw. If the number of
        trials whose length is exceeded by the window size is smaller than
        this, then only the corresponding trials are dropped, but the
        computation continues. Otherwise, an error is raised.

    Returns
    -------
    result_lists: (list, list, list, list)
        Trial, i_window_in_trial, start sample and stop sample of windows.
    """
    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts += start_offset
    stops += stop_offset
    if any(size > (stops - starts)):
        bads_mask = size > (stops - starts)
        min_duration = (stops - starts).min()
        if sum(bads_mask) <= accepted_bads_ratio * len(starts):
            starts = starts[np.logical_not(bads_mask)]
            stops = stops[np.logical_not(bads_mask)]
            warnings.warn(
                f"Trials {np.where(bads_mask)[0]} are being dropped as the "
                f"window size ({size}) exceeds their duration {min_duration}."
            )
        else:
            current_ratio = sum(bads_mask) / len(starts)
            raise ValueError(
                f"Window size {size} exceeds trial duration "
                f"({min_duration}) for too many trials "
                f"({current_ratio * 100}%). Set "
                f"accepted_bads_ratio to at least {current_ratio}"
                "and restart training to be able to continue."
            )

    i_window_in_trials, i_trials, window_starts = [], [], []
    for start_i, (start, stop) in enumerate(zip(starts, stops)):
        # Generate possible window starts, with given stride, between original
        # trial onsets and stops (shifted by start_offset and stop_offset,
        # respectively)
        possible_starts = np.arange(start, stop, stride)

        # Possible window start is actually a start, if window size fits in
        # trial start and stop
        valid = (possible_starts + size) <= stop
        valid_inds = np.flatnonzero(valid)
        # Extending from the array keeps NumPy scalar starts, as in the scalar
        # loop; tolist() would silently change their Python types.
        window_starts.extend(possible_starts[valid])
        i_window_in_trials.extend(valid_inds.tolist())
        i_trials.extend([start_i] * len(valid_inds))

        # If the last window start + window size is not the same as
        # stop + stop_offset, create another window that overlaps and stops
        # at onset + stop_offset
        if not drop_last_window:  # noqa: SIM102 -- frozen experimental candidate
            if window_starts[-1] + size != stop:
                window_starts.append(stop - size)
                i_window_in_trials.append(i_window_in_trials[-1] + 1)
                i_trials.append(start_i)

    # Set window stops to be event stops (rather than trial stops)
    window_stops = np.array(window_starts) + size
    if not (len(i_window_in_trials) == len(window_starts) == len(window_stops)):
        raise ValueError(
            f"{len(i_window_in_trials)} == {len(window_starts)} == {len(window_stops)}"
        )

    return i_trials, i_window_in_trials, window_starts, window_stops


@contextmanager
def _using(function):
    original = windowers._compute_window_inds
    windowers._compute_window_inds = function
    try:
        yield
    finally:
        windowers._compute_window_inds = original


def _fixture(config):
    starts = 100 + np.arange(config["trials"]) * (config["duration"] + 100)
    stops = starts + config["duration"]
    data = np.arange(2 * (int(stops[-1]) + 100), dtype=np.float64).reshape(2, -1)
    data /= 100000
    raw = mne.io.RawArray(
        data, mne.create_info(["C3", "C4"], 100, ch_types="eeg"), verbose=False
    )
    raw.set_annotations(
        mne.Annotations(
            starts / 100,
            np.full(len(starts), config["duration"] / 100),
            ["A" if i % 2 == 0 else "B" for i in range(len(starts))],
            extras=[
                {"trial_uid": f"trial-{i}", "phase": "train"}
                for i in range(len(starts))
            ],
        )
    )
    concat = BaseConcatDataset([RawDataset(raw, description={"subject": 1})])
    return starts, stops, raw, concat


def _construct(concat, config):
    return windowers.create_windows_from_events(
        concat,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=config["size"],
        window_stride_samples=config["stride"],
        on_last_window="overlap",
        mapping={"A": 0, "B": 1},
        use_mne_epochs=False,
        drop_bad_windows=None,
        n_jobs=1,
        verbose=False,
    )


def _verify(reference, candidate, starts, stops, concat, config):
    args = (0, 0, config["size"], config["stride"], False, 0)
    expected = reference(starts.copy(), stops.copy(), *args)
    actual = candidate(starts.copy(), stops.copy(), *args)
    for want, got in zip(expected, actual):
        assert type(got) is type(want)
        np.testing.assert_array_equal(got, want)
        if isinstance(want, list):
            assert [type(value) for value in got] == [type(value) for value in want]
        else:
            assert got.dtype == want.dtype
    with _using(reference):
        old = _construct(concat, config)
    with _using(candidate):
        new = _construct(concat, config)
    pd.testing.assert_frame_equal(old.get_metadata(), new.get_metadata())
    assert len(old) == len(new)
    for i in range(len(old)):
        for want, got in zip(old[i], new[i]):
            np.testing.assert_array_equal(got, want)
    metadata = new.get_metadata()
    for trial, group in metadata.groupby("i_trial_in_dataset", sort=False):
        assert set(group["trial_uid"]) == {f"trial-{trial}"}
        assert set(group["target"]) == {trial % 2}
    return {
        "windows_checked": len(new),
        "metadata_rows_checked": len(metadata),
        "helper_arrays_exact": True,
        "window_data_targets_crops_exact": True,
        "annotation_identity_exact": True,
    }


def _input(name, value):
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.strides),
        "stride_unit": "bytes",
    }


def _allocations(run):
    gc.collect()
    tracemalloc.start()
    try:
        output = run()
        retained, peak = tracemalloc.get_traced_memory()
        del output
    finally:
        tracemalloc.stop()
    return {"traced_retained_bytes": retained, "traced_peak_bytes": peak}


def _worker(args):
    """Keep disposable-worker runtime policy local even when invoked by a test."""
    threads, level = torch.get_num_threads(), mne.utils.logger.level
    try:
        return _run_worker(args)
    finally:
        torch.set_num_threads(threads)
        mne.set_log_level(level)


def _run_worker(args):
    if _git("status", "--porcelain"):
        raise RuntimeError("Measurements require a clean checkout; commit edits first")
    assert Path(braindecode.__file__).resolve().is_relative_to(ROOT)
    torch.set_num_threads(1)
    mne.set_log_level("ERROR")
    reference, reference_hash = _reference()
    candidate = _experimental_window_inds
    function = reference if args.implementation == "reference" else candidate
    config = CASES[args.case]
    starts, stops, raw, concat = _fixture(config)
    metrics = _verify(reference, candidate, starts, stops, concat, config)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "mne": mne.__version__,
            "pandas": pd.__version__,
            "torch": str(torch.__version__),
            "braindecode": braindecode.__version__,
        },
        "device": "cpu",
        "device_name": args.device_name,
    }
    source = {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": False,
        "dirty_source_hashes": {},
    }

    def helper():
        # Copies preserve the baseline helper's input-mutation contract.
        return function(
            starts.copy(),
            stops.copy(),
            0,
            0,
            config["size"],
            config["stride"],
            False,
            0,
        )

    with _using(function):
        for stage, run in (
            ("helper", helper),
            ("construction", lambda: _construct(concat, config)),
        ):
            measurement = measure_step(
                run,
                device=torch.device("cpu"),
                warmup=3,
                min_run_time=args.min_run_time,
            )
            allocations = _allocations(run)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            allocations["process_high_water_rss_bytes"] = int(
                rss if sys.platform == "darwin" else rss * 1024
            )
            record = {
                "schema_version": 1,
                "case_id": f"window_indices/{args.case}/{stage}",
                "status": "ok",
                "implementation": args.implementation,
                "config": {
                    "inputs": [_input("starts", starts), _input("stops", stops)]
                    + ([_input("raw", raw._data)] if stage == "construction" else []),
                    "seed": None,
                    "threads": 1,
                    "workers": 0,
                    "cache_policy": "fresh process; prior exact reference/candidate verification; warmup/calibration; OS cache uncontrolled",
                    "autograd": "not_applicable",
                    "precision": "integer indices; float64 raw; float32 dataset retrieval",
                    "parameters": {
                        **config,
                        "pair": args.pair,
                        "sfreq": 100,
                        "channels": 2,
                        "stage": stage,
                        "on_last_window": "overlap",
                        "annotation_extras": ["trial_uid", "phase"],
                        "use_mne_epochs": False,
                        "baseline": BASELINE,
                        "baseline_function_sha256": reference_hash,
                    },
                },
                "environment": environment,
                "source": source,
                "correctness": {
                    "status": "passed",
                    "method": "exact helper arrays/types, full metadata frame, all windows data/target/crop; annotation identity and target literals",
                    "metrics": metrics,
                },
                **measurement,
                "notes": [
                    "Production uses scalar default; candidate is benchmark-only, no qualified dispatch threshold. Deterministic arange input; no RNG. Both implementations use the same current construction pipeline; only helper differs.",
                    "Construction excludes Raw setup, disk I/O, subsequent traversal and training. No helper-only pipeline claim.",
                    "Serial paired processes alternate order; other host load/OS cache uncontrolled, not exclusive hardware.",
                    "Separate untimed tracemalloc run: includes tracked Python/NumPy allocations, not all native memory. RSS is cumulative including imports, both correctness paths and prior stage; not stage-isolated.",
                    "CPU allocation evidence: "
                    + json.dumps(allocations, sort_keys=True),
                ],
            }
            write_result(
                record,
                args.output
                / f"{args.case}-pair{args.pair}-{args.implementation}-{stage}.json",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--device-name", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=["reference", "candidate"])
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--pair", type=int)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify both synthetic pipelines without timing; allows dirty source",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error("Output must be outside the checkout")
    if args.check_only:
        reference, _ = _reference()
        for name, config in CASES.items():
            starts, stops, _, concat = _fixture(config)
            print(
                name,
                _verify(
                    reference,
                    _experimental_window_inds,
                    starts,
                    stops,
                    concat,
                    config,
                ),
            )
        return
    if args.device_name is None:
        parser.error("Supply actual --device-name hardware identity")
    if args.worker:
        _worker(args)
        return
    if args.pairs < 5:
        parser.error("At least five independent pairs are required")
    args.output.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        for pair in range(args.pairs):
            order = (
                ("reference", "candidate")
                if pair % 2 == 0
                else ("candidate", "reference")
            )
            for implementation in order:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.acceleration.bench_window_indices",
                        "--worker",
                        "--output",
                        str(args.output),
                        "--case",
                        case,
                        "--pair",
                        str(pair),
                        "--implementation",
                        implementation,
                        "--min-run-time",
                        str(args.min_run_time),
                        "--device-name",
                        args.device_name,
                    ],
                    cwd=ROOT,
                    check=True,
                    timeout=180,
                )


if __name__ == "__main__":
    main()

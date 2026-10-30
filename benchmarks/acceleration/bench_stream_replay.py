"""Exact online window replay through an unchanged eval model.

Run from a clean checkout, on whole otherwise idle cores::

    python -m benchmarks.acceleration.bench_stream_replay --output /tmp/run1.json
    python -m benchmarks.acceleration.bench_stream_replay \
        --summarize /tmp/run*.json --expected-runs 5

Chunks are already-preprocessed samples of exactly one trial in chronological
``(channels, time)`` layout. ``iter_completed_windows`` yields each window as
soon as the chunk that completes it arrives; with ``drop_last_window=False``
the tail-aligned last window appears only after the input ends (explicit
flush). A new call starts from empty state. The offline reference is
Braindecode's ``create_fixed_length_windows``. This is a benchmark-local
scheduler, not a public streaming runtime.

Equivalence covers windowing of identical samples only. It does not cover
per-chunk (reset) standardization, non-causal filters or any other
preprocessing whose online form differs from the offline one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmarks.acceleration._qualification import historical_only

SCHEMA = "stream-replay-v1"
ROOT = Path(__file__).resolve().parents[2]
QUANTILES = {"p50": 50, "p95": 95, "p99": 99}
NOTES = [
    (
        "Exact equivalence is claimed for windowing of identical, already "
        "preprocessed samples only. It is not claimed for per-chunk (reset) "
        "standardization, non-causal filters (e.g. zero-phase filtering) or any "
        "preprocessing whose online form differs from the offline one."
    ),
    (
        "Replay runs as fast as possible, not paced in real time. Compute latency "
        "excludes acquisition time and the intrinsic window delay, which is "
        "reported separately in samples and signal seconds."
    ),
    (
        "Streamed outputs are compared bit-exactly with the offline windows "
        "evaluated one at a time; batched offline evaluation is reported only as "
        "a maximum absolute difference, since other kernels may round differently."
    ),
    (
        "The model runs unchanged in eval mode under torch.no_grad; replay never "
        "switches its mode and checks parameters and buffers are unchanged."
    ),
    (
        "Benchmark-local scheduler, one call per trial, no cross-trial state; not "
        "a public streaming runtime."
    ),
    (
        "Timing claims need five independent serial invocations on one otherwise "
        "idle node; one invocation is not evidence of a latency level."
    ),
]


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


class StreamWindowScheduler:
    """Bounded-buffer scheduler of fixed-length windows for one trial.

    ``push(chunk)`` validates the chunk immediately and returns an iterator
    over the ``(window, start, stop)`` tuples it completes (``stop`` is
    exclusive); exhaust it before the next push. ``flush()`` ends the trial and
    returns the tail-aligned window, if any. Windows are new C-contiguous
    arrays owned by the caller; inputs are never modified or retained.

    Between pushes at most ``window_size`` samples per channel are held
    (``window_size - 1`` with ``drop_last_window=True``), in a buffer of
    ``2 * window_size`` samples per channel, whatever the trial duration.
    """

    def __init__(self, *, window_size, stride, drop_last_window):
        self.window_size = _positive_int("window_size", window_size)
        self.stride = _positive_int("stride", stride)
        if not isinstance(drop_last_window, bool):
            raise TypeError(
                "drop_last_window must be a bool, got "
                f"{type(drop_last_window).__name__}."
            )
        self.drop_last_window = drop_last_window
        self._buf = None
        self._lo = self._hi = 0
        self._received = 0
        self._next_start = 0
        self._last_stop = None
        self._busy = False
        self._flushed = False

    @property
    def received_samples(self) -> int:
        return self._received

    @property
    def buffered_samples(self) -> int:
        return self._hi - self._lo

    @property
    def buffer_capacity(self) -> int:
        return 0 if self._buf is None else self._buf.shape[1]

    def _check_open(self):
        if self._flushed:
            raise RuntimeError("Scheduler already flushed; use one per trial.")
        if self._busy:
            raise RuntimeError("Exhaust the previous push before continuing.")

    def _validate(self, chunk):
        if not isinstance(chunk, np.ndarray):
            raise TypeError(
                f"Chunks must be numpy.ndarray, got {type(chunk).__name__}."
            )
        if chunk.ndim != 2:
            raise ValueError(
                f"Chunks must be 2-D (channels, time), got shape {chunk.shape}."
            )
        if not np.issubdtype(chunk.dtype, np.floating):
            raise TypeError(f"Chunks must have a floating dtype, got {chunk.dtype}.")
        if self._buf is None:
            if chunk.shape[0] == 0:
                raise ValueError("Chunks must have at least one channel.")
            self._buf = np.empty(
                (chunk.shape[0], 2 * self.window_size), dtype=chunk.dtype
            )
        elif chunk.shape[0] != self._buf.shape[0]:
            raise ValueError(
                f"Chunk channel count changed from {self._buf.shape[0]} to "
                f"{chunk.shape[0]} within a trial."
            )
        elif chunk.dtype != self._buf.dtype:
            raise TypeError(
                f"Chunk dtype changed from {self._buf.dtype} to {chunk.dtype} "
                "within a trial."
            )

    def push(self, chunk) -> Iterator[tuple[np.ndarray, int, int]]:
        self._check_open()
        self._validate(chunk)
        self._busy = True  # stays set if the returned iterator is abandoned
        return self._complete(chunk)

    def _complete(self, chunk):
        size, received = self.window_size, self._received
        end = received + chunk.shape[1]
        while self._next_start + size <= end:
            start = self._next_start
            stop = start + size
            n_old = received - start  # samples that arrived before this chunk
            if n_old <= 0:
                window = chunk[:, start - received : stop - received].copy()
            else:
                window = np.empty((chunk.shape[0], size), dtype=chunk.dtype)
                window[:, :n_old] = self._buf[:, self._hi - n_old : self._hi]
                window[:, n_old:] = chunk[:, : size - n_old]
            self._next_start += self.stride
            self._last_stop = stop
            yield window, start, stop
        self._retain(chunk)
        self._busy = False

    def _retain(self, chunk):
        """Keep only the samples a later window or the tail window can use."""
        n_new = chunk.shape[1]
        total = self._received + n_new
        if self.drop_last_window:
            keep = min(self.window_size, max(0, total - self._next_start))
        else:
            keep = min(self.window_size, total)
        from_old = keep - min(keep, n_new)
        buf = self._buf
        if from_old == 0:
            buf[:, :keep] = chunk[:, n_new - keep :]
            self._lo, self._hi = 0, keep
        else:  # n_new < keep <= window_size
            lo, hi = self._hi - from_old, self._hi
            if hi + n_new > buf.shape[1]:
                buf[:, :from_old] = buf[:, lo:hi]
                lo, hi = 0, from_old
            buf[:, hi : hi + n_new] = chunk
            self._lo, self._hi = lo, hi + n_new
        self._received = total

    def flush(self) -> list[tuple[np.ndarray, int, int]]:
        self._check_open()
        self._flushed = True
        n_times, size = self._received, self.window_size
        if n_times < size:
            self._buf = None
            raise ValueError(
                f"Trial of {n_times} samples is shorter than window_size={size}; "
                "offline windowing rejects it too."
            )
        tail = []
        if not self.drop_last_window and self._last_stop != n_times:
            window = self._buf[:, self._hi - size : self._hi].copy()
            tail.append((window, n_times - size, n_times))
        self._buf = None
        self._lo = self._hi = 0
        return tail


def iter_completed_windows(
    chunks: Iterable[np.ndarray],
    *,
    window_size: int,
    stride: int,
    drop_last_window: bool,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield ``(window, start, stop)`` for one trial as chunks arrive.

    Arguments are validated at call time; chunks when they are pulled. The end
    of ``chunks`` is the explicit flush. Each call has its own state.
    """
    scheduler = StreamWindowScheduler(
        window_size=window_size, stride=stride, drop_last_window=drop_last_window
    )
    return _iterate(scheduler, chunks)


def _iterate(scheduler, chunks):
    for chunk in chunks:
        yield from scheduler.push(chunk)
    yield from scheduler.flush()


def replay_trial(
    model,
    chunks: Iterable[np.ndarray],
    *,
    window_size: int,
    stride: int,
    drop_last_window: bool,
    clock=time.perf_counter,
) -> list[dict]:
    """Stream one trial through ``model`` and time each output.

    ``latency_s`` runs from handing the completing chunk (or the flush) to the
    scheduler until the output exists; ``model_s`` is the forward alone;
    ``ready_s`` is measured from handing in the first chunk.
    """
    import torch

    if model.training:
        raise ValueError(
            "Model must be in eval mode; replay never changes the model mode."
        )
    scheduler = StreamWindowScheduler(
        window_size=window_size, stride=stride, drop_last_window=drop_last_window
    )
    records: list[dict] = []
    origin = None

    def run(items, t_in, after_flush):
        first = len(records)
        for window, start, stop in items:
            t0 = clock()
            output = model(torch.from_numpy(window)[None])[0]
            t1 = clock()
            records.append(
                {
                    "start": start,
                    "stop": stop,
                    "after_flush": after_flush,
                    "output": output,
                    "model_s": t1 - t0,
                    "latency_s": t1 - t_in,
                    "ready_s": t1 - origin,
                }
            )
        for record in records[first:]:
            record["available_samples"] = scheduler.received_samples

    with torch.no_grad():
        for chunk in chunks:
            t_in = clock()
            origin = t_in if origin is None else origin
            run(scheduler.push(chunk), t_in, False)
        t_in = clock()
        origin = t_in if origin is None else origin
        run(scheduler.flush(), t_in, True)
    return records


def offline_windows(data, *, window_size, stride, drop_last_window, sfreq):
    """Windows of one trial from Braindecode's offline fixed-length windowing."""
    import mne

    from braindecode.datasets.base import BaseConcatDataset, RawDataset
    from braindecode.preprocessing import create_fixed_length_windows

    info = mne.create_info(data.shape[0], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose="error")
    windows = create_fixed_length_windows(
        BaseConcatDataset([RawDataset(raw, description={"trial": 0})]),
        window_size_samples=window_size,
        window_stride_samples=stride,
        on_last_window="drop" if drop_last_window else "overlap",
    )
    items = [windows[i] for i in range(len(windows))]
    return [(X, int(inds[1]), int(inds[2])) for X, _, inds in items]


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    q = np.percentile(values, list(QUANTILES.values()), method="linear")
    return {**{k: float(v) for k, v in zip(QUANTILES, q)}, "n": int(values.size)}


def _chunks(data, chunk_samples):
    return [
        data[:, i : i + chunk_samples] for i in range(0, data.shape[1], chunk_samples)
    ]


def _warmup(model, example, *, min_calls, min_s, max_s, clock):
    import torch

    timings, steady, start = [], False, clock()
    with torch.no_grad():
        while True:
            t0 = clock()
            model(example)
            t1 = clock()
            timings.append(t1 - t0)
            if len(timings) >= max(min_calls, 5) and t1 - start >= min_s:
                last = np.asarray(timings[-5:])
                if last.max() - last.min() <= 0.03 * np.median(last):
                    steady = True
                    break
            if t1 - start >= max_s:
                break
    return {
        "policy": (
            f"at least {max(min_calls, 5)} forwards and {min_s} s, then until "
            f"the last 5 agree within 3% of their median; cap {max_s} s"
        ),
        "timings_s": timings,
        "steady": steady,
    }


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _physical_cores(cpus):
    cores = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            package, core = "cpu", str(cpu)
        cores.add((package, core))
    return len(cores)


def _environment(threads):
    import mne
    import torch

    import braindecode

    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    else:
        affinity = list(range(os.cpu_count() or 1))
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_affinity": affinity,
        "physical_cores": _physical_cores(affinity),
        "torch_threads": torch.get_num_threads(),
        "requested_threads": threads,
        "dependencies": {
            "braindecode": braindecode.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mne": mne.__version__,
        },
    }


def run_benchmark(args) -> dict:
    import torch

    from braindecode import models

    started = datetime.now(UTC).isoformat()
    torch.set_num_threads(args.threads)
    environment = _environment(args.threads)
    if environment["physical_cores"] < args.threads:
        raise SystemExit(
            f"Requested {args.threads} threads but the affinity has "
            f"{environment['physical_cores']} physical cores."
        )
    size, stride, drop = args.window_size, args.stride, args.drop_last_window
    rng = np.random.default_rng(args.seed)
    if args.trial_samples is None:
        lengths = rng.integers(
            int(30 * args.sfreq), int(40 * args.sfreq), args.n_trials
        ).tolist()
    elif len(args.trial_samples) != args.n_trials:
        raise SystemExit("--trial-samples needs one length per trial.")
    else:
        lengths = list(args.trial_samples)
    trials = [
        rng.standard_normal((args.n_chans, n)).astype(np.float32) for n in lengths
    ]

    torch.manual_seed(args.seed)
    model = getattr(models, args.model)(
        n_chans=args.n_chans, n_outputs=args.n_outputs, n_times=size, sfreq=args.sfreq
    ).eval()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    clock = time.perf_counter
    example = torch.zeros((1, args.n_chans, size))
    with torch.no_grad():
        t0 = clock()
        model(example)
        cold_first_forward_s = clock() - t0
    warmup = _warmup(
        model,
        example,
        min_calls=args.warmup_min_calls,
        min_s=args.warmup_min_s,
        max_s=args.warmup_max_s,
        clock=clock,
    )

    # Timed replay; every check below runs outside it.
    replays = [
        replay_trial(
            model,
            _chunks(data, args.chunk_samples),
            window_size=size,
            stride=stride,
            drop_last_window=drop,
            clock=clock,
        )
        for data in trials
    ]

    metrics = {
        "windows_total": 0,
        "window_mismatches": 0,
        "output_mismatches": 0,
        "max_abs_diff_vs_batched_offline": 0.0,
        "random_chunk_window_mismatches": 0,
        "max_buffered_samples": 0,
        "buffer_capacity_samples": 0,
        "tail_windows": 0,
    }
    check_rng = np.random.default_rng([args.seed, 1])
    for data, replay in zip(trials, replays):
        reference = offline_windows(
            data,
            window_size=size,
            stride=stride,
            drop_last_window=drop,
            sfreq=args.sfreq,
        )
        metrics["windows_total"] += len(reference)
        metrics["tail_windows"] += sum(r["after_flush"] for r in replay)
        metrics["window_mismatches"] += abs(len(replay) - len(reference))
        with torch.no_grad():
            for record, (X, start, stop) in zip(replay, reference):
                same = (record["start"], record["stop"]) == (start, stop)
                metrics["window_mismatches"] += not same
                offline = model(torch.from_numpy(X)[None])[0]
                metrics["output_mismatches"] += not torch.equal(
                    record["output"], offline
                )
            batched = model(torch.from_numpy(np.stack([X for X, _, _ in reference])))
        streamed = torch.stack([r["output"] for r in replay])
        if streamed.shape == batched.shape:
            diff = float((streamed - batched).abs().max())
            metrics["max_abs_diff_vs_batched_offline"] = max(
                metrics["max_abs_diff_vs_batched_offline"], diff
            )
        # Window data from random chunking (including single samples).
        sizes = check_rng.integers(0, 3 * size + 2, data.shape[1]).tolist()
        scheduler = StreamWindowScheduler(
            window_size=size, stride=stride, drop_last_window=drop
        )
        got, pos = [], 0  # type: ignore[var-annotated]
        for n in sizes:
            if pos >= data.shape[1]:
                break
            got.extend(scheduler.push(data[:, pos : pos + n]))
            pos += n
            metrics["max_buffered_samples"] = max(
                metrics["max_buffered_samples"], scheduler.buffered_samples
            )
            metrics["buffer_capacity_samples"] = max(
                metrics["buffer_capacity_samples"], scheduler.buffer_capacity
            )
        got.extend(scheduler.flush())
        metrics["random_chunk_window_mismatches"] += abs(len(got) - len(reference))
        for (window, start, stop), (X, ref_start, ref_stop) in zip(got, reference):
            same = (start, stop) == (ref_start, ref_stop)
            metrics["random_chunk_window_mismatches"] += not (
                same and np.array_equal(window, X)
            )

    unchanged = all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    metrics["model_state_unchanged"] = unchanged
    metrics["model_eval_mode"] = not model.training
    passed = (
        metrics["window_mismatches"] == 0
        and metrics["output_mismatches"] == 0
        and metrics["random_chunk_window_mismatches"] == 0
        and unchanged
        and not model.training
    )

    records = [r for replay in replays for r in replay]
    regular = [r for r in records if not r["after_flush"]]
    waits = [r["available_samples"] - r["stop"] for r in regular]
    first = replays[0][0]
    intrinsic = {
        "definition": (
            "Signal-time delay of windowing, independent of compute. A window "
            "ending at sample stop needs stop samples and is produced at the "
            "end of the chunk holding sample stop - 1 (chunk wait). The tail "
            "window additionally needs the end-of-trial flush."
        ),
        "sfreq": args.sfreq,
        "window_size_samples": size,
        "window_span_s": size / args.sfreq,
        "stride_samples": stride,
        "output_period_s": stride / args.sfreq,
        "chunk_samples": args.chunk_samples,
        "chunk_period_s": args.chunk_samples / args.sfreq,
        "first_output_stop_samples": first["stop"],
        "first_output_available_samples": first["available_samples"],
        "first_output_available_s": first["available_samples"] / args.sfreq,
        "chunk_wait_samples": {
            "max": int(max(waits)) if waits else None,
            "median": float(np.median(waits)) if waits else None,
        },
        "lookahead_samples_beyond_window": 0,
        "tail_window_requires_flush": metrics["tail_windows"] > 0,
    }
    compute = {
        "definition": (
            "latency_s: from handing the completing chunk (or the flush) to "
            "the scheduler until the model output exists, including scheduler "
            "work and earlier windows of the same chunk; model_s: forward "
            "alone; first_output_s: from the first chunk of a trial to its "
            "first output. Warm process; cold_first_forward_s is the first "
            "forward of the process, before warmup."
        ),
        "quantile_method": "numpy.percentile, linear interpolation",
        "raw_latency_s": [r["latency_s"] for r in records],
        "raw_model_s": [r["model_s"] for r in records],
        "latency_s": _quantiles([r["latency_s"] for r in records]),
        "model_s": _quantiles([r["model_s"] for r in records]),
        "first_output_s": replays[0][0]["ready_s"],
        "first_output_s_per_trial": [replay[0]["ready_s"] for replay in replays],
        "cold_first_forward_s": cold_first_forward_s,
        "pacing": "as fast as possible",
    }
    status = _git("status", "--porcelain")
    return {
        "schema": SCHEMA,
        "status": "ok" if passed else "failed",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source": {"commit": _git("rev-parse", "HEAD"), "dirty": status != ""},
        "config": {
            "model": args.model,
            "n_chans": args.n_chans,
            "n_outputs": args.n_outputs,
            "sfreq": args.sfreq,
            "window_size": size,
            "stride": stride,
            "drop_last_window": drop,
            "chunk_samples": args.chunk_samples,
            "n_trials": args.n_trials,
            "trial_samples": lengths,
            "seed": args.seed,
            "threads": args.threads,
            "dtype": "float32",
            "autograd": "torch.no_grad",
            "batch_size": 1,
        },
        "environment": environment,
        "correctness": {
            "status": "passed" if passed else "failed",
            "method": (
                "Streamed (start, stop) equal to create_fixed_length_windows; "
                "streamed outputs torch.equal to the offline windows evaluated "
                "one at a time; random chunk sizes (0 to 3 * window_size + 1 "
                "samples) give np.array_equal windows; model state and mode "
                "unchanged."
            ),
            "metrics": metrics,
        },
        "intrinsic_delay": intrinsic,
        "compute_latency": compute,
        "warmup": warmup,
        "notes": NOTES,
    }


def summarize_runs(records: list[dict], *, expected_runs: int) -> dict:
    """Pool independent invocations only when they are complete and consistent."""
    failures = []
    for r in records:
        if r.get("schema") != SCHEMA:
            failures.append("missing or unsupported replay schema")
        if r.get("warmup", {}).get("steady") is not True:
            failures.append("warmup was not steady")
        env = r.get("environment", {})
        threads = env.get("requested_threads", 0)
        if (type(threads) is not int or threads <= 0
                or env.get("torch_threads") != threads
                or env.get("physical_cores", 0) < threads):
            failures.append("unverified or oversubscribed physical cores/threads")
        compute = r.get("compute_latency", {})
        for key in ("latency_s", "model_s"):
            raw = compute.get("raw_" + key, [])
            if (not raw or not np.isfinite(raw).all() or min(raw) <= 0
                    or len(raw) != compute.get(key, {}).get("n")):
                failures.append(f"invalid or missing raw {key}")
            elif any(not np.isclose(compute[key].get(q, np.nan), v)
                     for q, v in _quantiles(raw).items()):
                failures.append(f"quantiles disagree with raw {key}")
        first = compute.get("first_output_s", np.nan)
        if not np.isfinite(first) or first <= 0:
            failures.append("invalid first-output latency")
    if len(records) != expected_runs:
        failures.append(f"expected {expected_runs} runs, got {len(records)}")
    if any(r.get("status") != "ok" for r in records):
        failures.append("a run did not end with status ok")
    commits = {r["source"]["commit"] for r in records}
    if len(commits) != 1 or None in commits:
        failures.append(f"runs do not share one commit: {sorted(map(str, commits))}")
    if any(r["source"]["dirty"] for r in records):
        failures.append("a run used a dirty checkout")
    hosts = {r["environment"]["hostname"] for r in records}
    if len(hosts) != 1:
        failures.append(f"runs span several hosts: {sorted(hosts)}")
    configs = {json.dumps(r["config"], sort_keys=True) for r in records}
    if len(configs) != 1:
        failures.append("runs differ in configuration")
    summary = historical_only(
        {"runs": len(records), "failures": failures},
        "Pooling checks are diagnostic, not an attestation of launch/source/policy receipts.",
    )
    if failures:
        return {**summary, "latency_s": None, "model_s": None, "first_output_s": None}

    def stats(values):
        values = [float(v) for v in values]
        return {
            "median": float(np.median(values)),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    for key in ("latency_s", "model_s"):
        summary[key] = {
            q: stats(r["compute_latency"][key][q] for r in records) for q in QUANTILES
        }
    summary["first_output_s"] = stats(
        r["compute_latency"]["first_output_s"] for r in records
    )
    summary["commit"] = commits.pop()
    summary["hostname"] = hosts.pop()
    return summary


def _write_json(payload, path: Path):
    text = json.dumps(payload, indent=2, allow_nan=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", choices=("ShallowFBCSPNet", "EEGNet"), default="ShallowFBCSPNet"
    )
    parser.add_argument("--n-chans", type=int, default=22)
    parser.add_argument("--n-outputs", type=int, default=4)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=125)
    parser.add_argument(
        "--drop-last-window", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--chunk-samples", type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--trial-samples", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-min-calls", type=int, default=20)
    parser.add_argument("--warmup-min-s", type=float, default=2.0)
    parser.add_argument("--warmup-max-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summarize", type=Path, nargs="+", default=None)
    parser.add_argument("--expected-runs", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.summarize is not None:
        records = [json.loads(p.read_text()) for p in args.summarize]
        result = summarize_runs(records, expected_runs=args.expected_runs)
        ok = not result["failures"]
    else:
        result = historical_only(run_benchmark(args),
                                 "Launch/source/policy receipts are externally audited only.")
        ok = result["status"] == "ok"
    if args.output is not None:
        _write_json(result, args.output)
    print(json.dumps(result if args.summarize else _brief(result), indent=2))
    return 0 if ok else 1


def _brief(record):
    return {
        "status": record["status"],
        "correctness": record["correctness"]["metrics"],
        "intrinsic_delay": {
            k: v for k, v in record["intrinsic_delay"].items() if k != "definition"
        },
        "compute_latency": {
            k: record["compute_latency"][k]
            for k in ("latency_s", "model_s", "first_output_s", "cold_first_forward_s")
        },
    }


if __name__ == "__main__":
    sys.exit(main())

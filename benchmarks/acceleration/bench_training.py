"""Training and train-metric scoring phase accounting for synthetic EEGClassifier.

Run from a clean commit with --output outside Git. Each metric case runs in a
fresh process; cases rotate order across repeats. Nothing in braindecode or
skorch is patched: the measured net is a stock EEGClassifier with the
braindecode training-iterator defaults (shuffle, drop_last=True), string
training metrics expanded to PostEpochTrainScoring, and train_split=None (no
validation pass; set explicitly, unlike the skorch default).

Headline: an uninstrumented net timed by ``measure_step`` over
``partial_fit(epochs=1)`` calls; its counts come only from skorch history and
optimizer state after timing. Profile: a separate fresh net with the same
configuration observed through public module/criterion/optimizer hooks and
skorch callbacks. Its sequential exclusive segments plus an unattributed
remainder add up to each epoch's wall time; inclusive spans overlap them and
are reported separately, never summed.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import sklearn
import skorch
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skorch.callbacks import Callback

import braindecode
from benchmarks.acceleration._common import measure_step, write_result
from braindecode.classifier import EEGClassifier
from braindecode.util import set_random_seeds

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261026
METRICS = ("accuracy", "balanced_accuracy")
SCORERS = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}
EXCLUSIVE_PHASES = (
    "train_loading",
    "train_step_setup",
    "train_zero_grad_input_transfer",
    "train_forward",
    "train_target_transfer",
    "train_loss",
    "train_backward",
    "train_optimizer_update",
    "train_batch_bookkeeping_callbacks",
    "epoch_end_default_callbacks",
    "train_scoring_forward",
    "train_scoring_other",
)
_TRAIN_BATCH = (
    "batch_begin",
    "step0",
    "train_fwd0",
    "train_fwd1",
    "loss0",
    "loss1",
    "grad",
    "step1",
    "batch_end",
)
# Each per-batch exclusive phase ends at the matching _TRAIN_BATCH event.
_BATCH_PHASES = EXCLUSIVE_PHASES[: len(_TRAIN_BATCH)]
assert _BATCH_PHASES[-1] == "train_batch_bookkeeping_callbacks"
_STEP0, _STEP1 = _TRAIN_BATCH.index("step0"), _TRAIN_BATCH.index("step1")
_TRAIN_FWD0 = _TRAIN_BATCH.index("train_fwd0")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def _source():
    hashes = {}
    status = _git("status", "--porcelain", "--untracked-files=all")
    for line in status.splitlines():
        name = line[3:].split(" -> ")[-1]
        path = ROOT / name
        content = path.read_bytes() if path.is_file() else b"<deleted>"
        hashes[name] = hashlib.sha256(content).hexdigest()
    return {
        "commit": _git("rev-parse", "HEAD").strip(),
        "dirty": bool(hashes),
        "dirty_source_hashes": hashes,
    }


def _synchronizer(device):
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "mps":
        return torch.mps.synchronize
    return None


def _data(n_samples, n_chans, n_times):
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, n_chans, n_times), dtype=np.float32)
    y = (np.arange(n_samples) % 2).astype(np.int64)
    return X, y


def _classifier(
    model_name,
    *,
    n_chans,
    n_times,
    batch_size,
    valid_batch_size,
    n_metrics,
    device,
    before=(),
    after=(),
):
    set_random_seeds(SEED, cuda=device.type == "cuda")
    return EEGClassifier(
        model_name,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        train_split=None,
        batch_size=batch_size,
        iterator_valid__batch_size=valid_batch_size,
        iterator_train__shuffle=True,
        iterator_train__drop_last=True,
        classes=[0, 1],
        max_epochs=1,
        device=str(device),
        verbose=0,
        callbacks=[*before, *METRICS[:n_metrics], *after],
    )


class _ScoringStart(Callback):
    """First user callback: marks where on_epoch_end user callbacks begin."""

    def on_epoch_end(self, net, **kwargs):
        dict(net.callbacks_)["profile_recorder"].mark("scoring_begin")


class _PhaseRecorder(Callback):
    """Last user callback plus public hooks; timestamps boundaries only."""

    def __init__(self, synchronize=None):
        self.synchronize = synchronize

    def initialize(self):
        self.remove_hooks()
        self.events_ = []
        self.hooked_ = None
        return self

    def remove_hooks(self):
        for handle in getattr(self, "handles_", []):
            handle.remove()
        self.handles_ = []

    def mark(self, kind, value=0):
        if self.synchronize is not None:
            self.synchronize()
        self.events_.append((kind, perf_counter(), value))

    def _forward_begin(self, module, args):
        self.mark("train_fwd0" if module.training else "eval_fwd0", len(args[0]))

    def _forward_end(self, module, args, output):
        self.mark("train_fwd1" if module.training else "eval_fwd1")

    def on_train_begin(self, net, **kwargs):
        key = (id(net.module_), id(net.criterion_), id(net.optimizer_))
        if self.hooked_ == key:
            return
        self.remove_hooks()
        module, criterion, optimizer = net.module_, net.criterion_, net.optimizer_
        self.handles_ = [
            module.register_forward_pre_hook(self._forward_begin),
            module.register_forward_hook(self._forward_end),
            criterion.register_forward_pre_hook(lambda m, a: self.mark("loss0")),
            criterion.register_forward_hook(lambda m, a, o: self.mark("loss1")),
            optimizer.register_step_pre_hook(lambda o, a, k: self.mark("step0")),
            optimizer.register_step_post_hook(lambda o, a, k: self.mark("step1")),
        ]
        self.hooked_ = key

    def on_epoch_begin(self, net, **kwargs):
        self.mark("epoch_begin")

    def on_batch_begin(self, net, training=False, **kwargs):
        if training:
            self.mark("batch_begin")

    def on_grad_computed(self, net, named_parameters, **kwargs):
        self.mark("grad")

    def on_batch_end(self, net, training=False, **kwargs):
        if training:
            self.mark("batch_end")

    def on_epoch_end(self, net, **kwargs):
        self.mark("scoring_end")


def _expect(events, i, kind):
    if i >= len(events) or events[i][0] != kind:
        found = events[i][0] if i < len(events) else "<end>"
        raise RuntimeError(
            f"Unexpected profile event {found!r} at {i}; expected {kind!r}"
        )
    return events[i][1]


def _segments(events, wall_s, n_samples):
    """Split one epoch's boundary events into sequential exclusive segments."""
    exclusive = dict.fromkeys(EXCLUSIVE_PHASES, 0.0)
    counts = dict.fromkeys(
        (
            "train_batches",
            "train_forwards",
            "optimizer_steps",
            "train_samples",
            "dropped_train_samples",
            "eval_forwards",
            "eval_samples",
        ),
        0,
    )
    optimizer_span = 0.0
    previous = _expect(events, 0, "epoch_begin")
    i = 1
    while i < len(events) and events[i][0] == "batch_begin":
        t = [_expect(events, i + j, kind) for j, kind in enumerate(_TRAIN_BATCH)]
        for phase, start, stop in zip(_BATCH_PHASES, [previous, *t[:-1]], t):
            exclusive[phase] += stop - start
        optimizer_span += t[_STEP1] - t[_STEP0]
        counts["train_batches"] += 1
        counts["train_forwards"] += 1
        counts["optimizer_steps"] += 1
        counts["train_samples"] += events[i + _TRAIN_FWD0][2]
        previous = t[-1]
        i += len(_TRAIN_BATCH)
    scoring_begin = _expect(events, i, "scoring_begin")
    exclusive["epoch_end_default_callbacks"] = scoring_begin - previous
    i += 1
    while i < len(events) and events[i][0] == "eval_fwd0":
        start, stop = events[i][1], _expect(events, i + 1, "eval_fwd1")
        exclusive["train_scoring_forward"] += stop - start
        counts["eval_forwards"] += 1
        counts["eval_samples"] += events[i][2]
        i += 2
    scoring_end = _expect(events, i, "scoring_end")
    if i + 1 != len(events):
        raise RuntimeError(f"Unexpected trailing profile events: {events[i + 1 :]}")
    scoring = scoring_end - scoring_begin
    exclusive["train_scoring_other"] = scoring - exclusive["train_scoring_forward"]
    counts["dropped_train_samples"] = n_samples - counts["train_samples"]
    return {
        "wall_s": wall_s,
        "exclusive_s": exclusive,
        "unattributed_s": wall_s - sum(exclusive.values()),
        "inclusive_s": {
            "optimizer_step_with_closure": optimizer_span,
            "train_scoring_callbacks": scoring,
        },
        "counts": counts,
    }


def _rss():
    return int(psutil.Process().memory_info().rss)


def _ru_maxrss():
    scale = 1 if sys.platform == "darwin" else 1024  # Linux reports KiB
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _input(name, value):
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.strides),
        "stride_unit": "bytes",
    }


def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _cpu_placement():
    """Logical CPUs this process may run on and the physical cores they span."""
    affinity = (sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                else list(range(os.cpu_count() or 1)))
    cores = set()
    for cpu in affinity:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:  # no topology: count every logical CPU as a core
            package, core = "unknown", str(cpu)
        cores.add((package, core))
    return {
        "affinity": affinity,
        "physical_cores": len(cores),
        "torch_threads": torch.get_num_threads(),
    }


def _warm_up(run, synchronize, *, min_s, max_s, window, rel_tol):
    """Run discarded calls for at least min_s, then until the last ``window``
    calls agree within ``rel_tol`` (max-min over min) or max_s is exceeded."""
    calls = []
    while True:
        if synchronize is not None:
            synchronize()
        start = perf_counter()
        run()
        if synchronize is not None:
            synchronize()
        calls.append(perf_counter() - start)
        total = sum(calls)
        last = calls[-window:]
        steady = len(calls) >= window and (max(last) - min(last)) / min(last) <= rel_tol
        if total >= min_s and len(calls) >= window and (steady or total >= max_s):
            return {"calls_s": calls, "total_s": total, "steady": steady}


def _headline_counts(net):
    history = net.history
    optimizer_steps = {int(state["step"]) for state in net.optimizer_.state.values()}
    return {
        "epochs": len(history),
        "train_batches_per_epoch": [int(n) for n in history[:, "train_batch_count"]],
        "train_samples_per_epoch": [
            int(sum(sizes)) for sizes in history[:, "batches", :, "train_batch_size"]
        ],
        "optimizer_steps": optimizer_steps.pop() if len(optimizer_steps) == 1 else -1,
        "skorch_epoch_dur_s": [float(d) for d in history[:, "dur"]],
    }


def run_case(
    *,
    model_name: str,
    n_samples: int,
    batch_size: int,
    n_metrics: int,
    device: str,
    n_chans: int = 22,
    n_times: int = 1000,
    valid_batch_size: int | None = None,
    min_run_time: float = 10.0,
    warmup_min_s: float = 20.0,
    warmup_max_s: float = 120.0,
    warmup_window: int = 5,
    warmup_rel_tol: float = 0.03,
    profile_epochs: int = 5,
    threads: int | None = None,
    device_name: str | None = None,
) -> dict:
    """Time and account one synthetic training configuration.

    Returns ``{"record": strict v1 record, "profile": separate phase profile}``.
    """
    if n_metrics not in range(len(METRICS) + 1):
        raise ValueError(f"n_metrics must be in 0..{len(METRICS)}")
    if profile_epochs < 1:
        raise ValueError("profile_epochs must be positive")
    if not (0 <= warmup_min_s <= warmup_max_s and warmup_window >= 2):
        raise ValueError("Need 0 <= warmup_min_s <= warmup_max_s, warmup_window >= 2")
    if not warmup_rel_tol > 0:
        raise ValueError("warmup_rel_tol must be positive")
    if threads is not None:
        torch.set_num_threads(threads)
    device = torch.device(device)
    valid_batch_size = valid_batch_size or batch_size
    metric_names = [f"train_{name}" for name in METRICS[:n_metrics]]
    X, y = _data(n_samples, n_chans, n_times)
    kwargs = {
        "n_chans": n_chans,
        "n_times": n_times,
        "batch_size": batch_size,
        "valid_batch_size": valid_batch_size,
        "n_metrics": n_metrics,
        "device": device,
    }
    train_batches = n_samples // batch_size  # drop_last=True
    expected = {
        "train_batches": train_batches,
        "train_forwards": train_batches,
        "optimizer_steps": train_batches,
        "train_samples": train_batches * batch_size,
        "dropped_train_samples": n_samples - train_batches * batch_size,
        "eval_forwards": math.ceil(n_samples / valid_batch_size) if n_metrics else 0,
        "eval_samples": n_samples if n_metrics else 0,
    }

    # Headline: stock net, no hooks or recorder callbacks; device syncs only at
    # call boundaries (warmup and measure_step), never inside an epoch.
    synchronize = _synchronizer(device)
    headline_net = _classifier(model_name, **kwargs)
    timed_call = functools.partial(headline_net.partial_fit, X, y, epochs=1)
    warmup = _warm_up(
        timed_call,
        synchronize,
        min_s=warmup_min_s,
        max_s=warmup_max_s,
        window=warmup_window,
        rel_tol=warmup_rel_tol,
    )
    measurement = measure_step(
        timed_call, device=device, warmup=0, min_run_time=min_run_time
    )
    del timed_call
    headline = _headline_counts(headline_net)
    history = headline_net.history
    headline_metrics_logged = all(
        name in row for row in history for name in metric_names
    ) and not any(f"train_{m}" in history[-1] for m in METRICS[n_metrics:])
    del headline_net, history
    gc.collect()
    rss_after_headline = _rss()

    # Profile: separate fresh net, same seed/config, boundary markers only.
    if device.type == "cuda":  # type: ignore[attr-defined]
        torch.cuda.reset_peak_memory_stats(device)
    net = _classifier(
        model_name,
        **kwargs,
        before=[("profile_scoring_start", _ScoringStart())],
        after=[("profile_recorder", _PhaseRecorder(synchronize))],
    )
    recorder = None
    epochs = []
    for _ in range(profile_epochs):
        if synchronize is not None:
            synchronize()
        start = perf_counter()
        net.partial_fit(X, y, epochs=1)
        if synchronize is not None:
            synchronize()
        wall_s = perf_counter() - start
        recorder = dict(net.callbacks_)["profile_recorder"]
        epoch = _segments(recorder.events_, wall_s, n_samples)
        epoch["skorch_dur_s"] = float(net.history[-1, "dur"])
        epochs.append(epoch)
        recorder.events_ = []
    recorder.remove_hooks()  # type: ignore[union-attr]
    profile_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None  # type: ignore[attr-defined]
    )
    with torch.no_grad():
        predictions = net.forward(X, training=False).argmax(1).numpy()
    final_weight_check = {
        f"train_{name}": {
            "history": float(net.history[-1, f"train_{name}"]),
            "recomputed": float(SCORERS[name](y, predictions)),
        }
        for name in METRICS[:n_metrics]
    }
    del net
    memory = {
        "rss_after_headline_bytes": rss_after_headline,
        "rss_after_profile_bytes": _rss(),
        "ru_maxrss_bytes": _ru_maxrss(),
        "profile_peak_cuda_allocated_bytes": profile_peak,
    }

    profiled = [epoch["counts"] for epoch in epochs]
    passed = (
        headline["train_batches_per_epoch"] == [train_batches] * headline["epochs"]
        and headline["train_samples_per_epoch"]
        == [expected["train_samples"]] * headline["epochs"]
        and headline["optimizer_steps"] == train_batches * headline["epochs"]
        and headline_metrics_logged
        and profiled == [expected] * profile_epochs
        and all(v["history"] == v["recomputed"] for v in final_weight_check.values())
    )
    case_id = (
        f"training/{model_name}/n{n_samples}-b{batch_size}-vb{valid_batch_size}"
        f"/metrics{n_metrics}"
    )
    profile = {
        "case_id": case_id,
        "separate_from_headline": True,
        "synchronized_markers": synchronize is not None,
        "exclusive_phase_order": list(EXCLUSIVE_PHASES),
        "epochs": epochs,
        "memory": memory,
        "notes": [
            "Exclusive segments are consecutive boundary intervals; with unattributed_s they sum to wall_s. Inclusive spans overlap them and must not be added. Markers synchronize the device only on CUDA/MPS and only in this profile run.",
            "Labels name the dominant work: other callbacks' on_batch_begin falls in train_loading, and other on_grad_computed callbacks plus skorch store_step fall in train_backward (the recorder is the last callback). The default callbacks used here add no such hooks.",
            "unattributed_s: partial_fit entry/exit (on_train_begin/end), dataset and iterator construction, default on_epoch_begin callbacks, PrintLog.",
            "train_loading: DataLoader iteration/collation up to on_batch_begin (first batch includes iterator creation). train_step_setup: train_step closure setup until the optimizer step pre-hook. train_zero_grad_input_transfer: zero_grad plus input to_tensor/device copy. train_target_transfer: target to_tensor/device copy before the criterion. train_batch_bookkeeping_callbacks: skorch loss.item(), history and on_batch_end callbacks.",
            "epoch_end_default_callbacks: loader exhaustion, batch-count record, EpochTimer/BatchScoring/valid_acc on_epoch_end. train_scoring_*: PostEpochTrainScoring reevaluation (eval-mode forwards vs loading, CPU copy, target extraction and metric computation).",
            "skorch_dur_s is EpochTimer's value, recorded by a default callback before training-metric callbacks run, so it excludes train_scoring_*.",
        ],
    }
    record = {
        "schema_version": 1,
        "case_id": case_id,
        "status": "ok" if passed else "failed",
        "implementation": "baseline",
        "config": {
            "inputs": [_input("X", X), _input("y", y)],
            "seed": SEED,
            "threads": torch.get_num_threads(),
            "workers": 0,
            "cache_policy": "fresh process per case; steady-state warmup calls (see parameters.warmup), measure_step first call and calibration excluded from median; OS cache uncontrolled",
            "autograd": "training steps with autograd; reevaluation under skorch evaluation_step (grad disabled)",
            "precision": "float32 inputs, parameters and outputs; int64 targets",
            "parameters": {
                "model_name": model_name,
                "n_samples": n_samples,
                "batch_size": batch_size,
                "valid_batch_size": valid_batch_size,
                "n_metrics": n_metrics,
                "metrics": metric_names,
                "n_chans": n_chans,
                "n_times": n_times,
                "n_outputs": 2,
                "drop_last": True,
                "shuffle": True,
                "train_split": None,
                "optimizer": "AdamW",
                "lr": 1e-3,
                "timed_call": "EEGClassifier.partial_fit(X, y, epochs=1)",
                "warmup": {
                    "policy": "time floor then stability window",
                    "min_s": warmup_min_s,
                    "max_s": warmup_max_s,
                    "window": warmup_window,
                    "rel_tol": warmup_rel_tol,
                },
                "cpu_placement": _cpu_placement(),
                "profile_epochs": profile_epochs,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "torch": str(torch.__version__),
                "braindecode": str(braindecode.__version__),
                "skorch": str(skorch.__version__),
                "scikit-learn": str(sklearn.__version__),
                "numpy": str(np.__version__),
                "psutil": str(psutil.__version__),
            },
            "device": str(device),
            "device_name": device_name or _device_name(device),
        },
        "source": _source(),
        "correctness": {
            "status": "passed" if passed else "failed",
            "method": "headline counts from skorch history/optimizer state; profiled counts from public hooks; training metrics equal to a final-weight eval-mode recomputation",
            "metrics": {
                "expected_per_epoch": expected,
                "profiled_per_epoch": profiled,
                "headline": headline,
                "headline_metrics_logged": headline_metrics_logged,
                "final_weight_check": final_weight_check,
                "warmup": warmup,
            },
        },
        **measurement,
        "notes": [
            "Headline median_s is one partial_fit(epochs=1) call: training epoch, training-metric reevaluation and per-call fit overhead; no hooks, recorder callbacks or added device syncs. Phase split comes from a separate profiled net (profile file).",
            "Training metrics are stock PostEpochTrainScoring: one shared eval-mode pass over all training samples with final post-epoch weights per epoch when n_metrics >= 1; none when n_metrics == 0. drop_last=True drops dropped_train_samples per epoch from training only.",
            "Warmup: discarded partial_fit calls on the timed net for at least min_s, then until the last window calls agree within rel_tol (max-min over min) or max_s elapses; correctness.metrics.warmup lists every call (calls_s[0] is the cold first call) and whether it ended steady. timing.first_call_s is therefore the first call after warmup, not a cold start.",
            "Synthetic float32 trials, 2 balanced classes, train_split=None, 0 DataLoader workers. Slurm CPU allocation; the node is not reserved exclusively. CPU placement (affinity, physical cores, torch threads) is in parameters.cpu_placement.",
            "Memory evidence: psutil current RSS after the headline (headline net released) and after the profile, and getrusage ru_maxrss (cumulative, includes imports and both nets). Linux updates these counters lazily, so ru_maxrss may trail a recent current-RSS sample; neither isolates per-phase allocation. profile_peak_cuda_allocated_bytes is the CUDA peak over the profile run (null off CUDA): "
            + json.dumps(memory, sort_keys=True),
        ]
        + (
            []
            if passed
            else ["Count or final-weight check failed; see correctness.metrics."]
        ),
    }
    return {"record": record, "profile": profile}


def _write_json(payload, path):
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(text)
    os.replace(stream.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--model", default="ShallowFBCSPNet")
    parser.add_argument("--n-samples", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-batch-size", type=int)
    parser.add_argument("--n-chans", type=int, default=22)
    parser.add_argument("--n-times", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-run-time", type=float, default=10.0)
    parser.add_argument("--profile-epochs", type=int, default=5)
    parser.add_argument("--warmup-min-s", type=float, default=20.0)
    parser.add_argument("--warmup-max-s", type=float, default=120.0)
    parser.add_argument("--warmup-window", type=int, default=5)
    parser.add_argument("--warmup-rel-tol", type=float, default=0.03)
    parser.add_argument("--n-metrics", type=int, choices=range(len(METRICS) + 1))
    args = parser.parse_args()
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Finish edits and commit before measuring")
    if not Path(braindecode.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"braindecode imported from {braindecode.__file__}")
    if args.n_metrics is None:  # parent: one fresh process per case, rotated
        args.output.mkdir(parents=True, exist_ok=True)
        cases = list(range(len(METRICS) + 1))
        shift = args.repeat % len(cases)
        for n_metrics in cases[shift:] + cases[:shift]:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.acceleration.bench_training",
                    *sys.argv[1:],
                    "--n-metrics",
                    str(n_metrics),
                ],
                check=True,
            )
        return
    result = run_case(
        model_name=args.model,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_metrics=args.n_metrics,
        device=args.device,
        n_chans=args.n_chans,
        n_times=args.n_times,
        valid_batch_size=args.valid_batch_size,
        min_run_time=args.min_run_time,
        warmup_min_s=args.warmup_min_s,
        warmup_max_s=args.warmup_max_s,
        warmup_window=args.warmup_window,
        warmup_rel_tol=args.warmup_rel_tol,
        profile_epochs=args.profile_epochs,
        threads=args.threads,
        device_name=args.device_name,
    )
    stem = args.output / f"metrics{args.n_metrics}-rep{args.repeat}"
    write_result(result["record"], stem.with_suffix(".json"))
    result["profile"]["repeat"] = args.repeat
    _write_json(result["profile"], stem.with_suffix(".profile.json"))
    if result["record"]["status"] != "ok":
        raise SystemExit(f"{result['record']['case_id']}: correctness failed")


if __name__ == "__main__":
    main()

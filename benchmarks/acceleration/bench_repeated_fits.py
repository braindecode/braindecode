"""Fold-safe repeated-fit (cross-validation) reuse recipe, private lab only.

Example (generated sources/results stay outside the checkout)::

    python -m benchmarks.acceleration.bench_repeated_fits --mode reuse \
        --n-jobs 2 --n-folds 4 --seed 0 --work-dir /tmp/rf --output /tmp/rf/r.json

The recipe prepares a synthetic EEG source once (filter, unit scaling,
windowing), persists it with ``BaseConcatDataset.save`` and then runs each
cross-validation fold as an independent fit on the persisted data:

- splits are by subject and fixed by ``seed`` (``split_schedule``);
- the only fitted preprocessing artifact (per-channel standardization) is fit on
  training windows only and is keyed by ``artifact_key`` over the explicit
  source revision, preprocessing, training IDs and versions;
- every fold uses ``sklearn.base.clone`` of an unfitted template (no
  ``warm_start``), reseeds torch, and gets its own shallow copies of the
  child datasets, so a fold-local transform never reaches another fold through
  the children that ``BaseConcatDataset.split`` would share; the shared signal
  arrays are made read-only;
- folds are spread over at most ``n_jobs`` explicit joblib (loky) workers;
  each worker loads the persisted source once and fits its folds serially with
  one torch thread. ``n_jobs`` must be a positive integer (no ``-1``); the CLI
  also refuses more workers than physical cores in its CPU affinity.

``artifact_key`` identifies this benchmark's explicit immutable source; it is
not a transparent library cache, and nothing here changes Braindecode defaults.
``mode=cold`` is the no-reuse baseline: it re-prepares the source for every
fold, serially. Study makespans exclude removal of temporary sources in both
modes. Results are one-shot study timings, so they use this recipe's own JSON
record rather than the repeated-callable v1 envelope of ``_common``. Each
record carries a ``status`` and a within-run ``correctness`` check
(``check_run``), and ``summarize_runs`` reports paired ratios only when every
run exited 0 with status ``ok``, every configuration has the expected repeats
and folds, and per-fold digests agree across all runs (parity). See
``benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import sklearn
import skorch
import torch
from joblib import Parallel, delayed
from sklearn.base import clone

import braindecode
from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "repeated-fits-v1"
GENERATOR = "synthetic-lateralized-10hz-v1"
GENERATOR_KEYS = ("n_subjects", "n_trials")
MANIFEST = "manifest.json"
WINDOWS = "windows"
FIT_THREADS = 1
UNIT_FACTORS = {"V": 1.0, "uV": 1e6}
DEFAULT_CONFIG = {
    "n_subjects": 8,
    "n_trials": 24,
    "ch_names": (  # noqa: SIM905 (22 names; one per line otherwise)
        "Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz"
    ).split(),
    "units": "uV",
    "sfreq": 250.0,
    "n_times": 1000,
    "l_freq": 4.0,
    "h_freq": 38.0,
    "window_size_samples": 500,
    "window_stride_samples": 250,
    "target_mapping": {"left": 0, "right": 1},
    "model": "ShallowFBCSPNet",
    "max_epochs": 2,
    "batch_size": 32,
    "lr": 6.25e-4,
}
PREPROCESSING_KEYS = (
    "ch_names",
    "units",
    "sfreq",
    "n_times",
    "l_freq",
    "h_freq",
    "window_size_samples",
    "window_stride_samples",
    "target_mapping",
)
TRAINING_KEYS = ("model", "max_epochs", "batch_size", "lr")


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def artifact_key(
    source_revision: str,
    preprocessing: dict,
    split_ids: tuple[str, ...],
    versions: dict,
) -> str:
    """SHA-256 of canonical JSON over every input that defines an artifact.

    ``split_ids`` stays ordered (training order changes batch composition), so
    a reordering is a different key: conservative, never a false reuse.
    """
    payload = {
        "source_revision": str(source_revision),
        "preprocessing": preprocessing,
        "split_ids": [str(i) for i in split_ids],
        "versions": versions,
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _versions():
    return {
        "python": platform.python_version(),
        "braindecode": braindecode.__version__,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "skorch": skorch.__version__,
        "torch": str(torch.__version__),
    }


def _config(config):
    merged = copy.deepcopy(DEFAULT_CONFIG)
    unknown = set(config or {}) - set(merged)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    merged.update(copy.deepcopy(config or {}))
    if merged["units"] not in UNIT_FACTORS:
        raise ValueError(f"units must be one of {sorted(UNIT_FACTORS)}")
    if len(set(merged["target_mapping"])) != 2:
        raise ValueError("The synthetic source has exactly two classes.")
    return merged


def _subjects(n_subjects):
    return [f"sub-{i:02d}" for i in range(n_subjects)]


def _scale(data, factor):
    return data * factor


def _synthetic_raws(config, seed):
    """Gaussian noise plus a class-lateralized 10 Hz rhythm, in volts."""
    rng = np.random.default_rng(seed)
    ch_names = list(config["ch_names"])
    n_chans, n_times, sfreq = len(ch_names), config["n_times"], config["sfreq"]
    labels = sorted(config["target_mapping"])
    left = ch_names.index("C3") if "C3" in ch_names else 0
    right = ch_names.index("C4") if "C4" in ch_names else n_chans - 1
    t = np.arange(n_times) / sfreq
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    datasets = []
    for subject in _subjects(config["n_subjects"]):
        offset = rng.normal(0.0, 2.0, size=(n_chans, 1))
        targets = rng.permutation([labels[i % 2] for i in range(config["n_trials"])])
        for trial, target in enumerate(targets):
            x = rng.normal(0.0, 10.0, size=(n_chans, n_times)) + offset
            phase = rng.uniform(0, 2 * np.pi)
            channel = left if target == labels[0] else right
            x[channel] += 8.0 * np.sin(2 * np.pi * 10.0 * t + phase)
            raw = mne.io.RawArray(x * 1e-6, info, verbose="error")
            description = {"subject": subject, "trial": trial, "target": target}
            datasets.append(RawDataset(raw, description, target_name="target"))
    return BaseConcatDataset(datasets)


def _source_revision(seed, config):
    """Every generator input; preprocessing is keyed separately."""
    generator = {key: config[key] for key in GENERATOR_KEYS}
    return f"{GENERATOR}:" + _canonical_json({"seed": int(seed), **generator})


def _tree_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare_source(source_dir, *, seed: int, config: dict | None = None) -> dict:
    """Cold preparation: synthesize, preprocess, window and persist once.

    ``source_dir`` must be empty or absent; an existing source is never
    overwritten (it is immutable once its manifest exists).
    """
    source_dir = Path(source_dir)
    if source_dir.exists() and any(source_dir.iterdir()):
        raise FileExistsError(f"{source_dir} is not empty; sources are immutable.")
    source_dir.mkdir(parents=True, exist_ok=True)
    config = _config(config)
    preprocessing = {key: config[key] for key in PREPROCESSING_KEYS}
    concat = _synthetic_raws(config, seed)
    preprocess(
        concat,
        [
            Preprocessor(
                "reorder_channels",
                apply_on_array=False,
                ch_names=list(config["ch_names"]),
            ),
            Preprocessor(
                "filter",
                apply_on_array=False,
                l_freq=config["l_freq"],
                h_freq=config["h_freq"],
            ),
            Preprocessor(_scale, factor=UNIT_FACTORS[config["units"]]),
        ],
        n_jobs=1,
    )
    windows = create_fixed_length_windows(
        concat,
        window_size_samples=config["window_size_samples"],
        window_stride_samples=config["window_stride_samples"],
        on_last_window="drop",
        mapping=dict(config["target_mapping"]),
        preload=True,
    )
    windows.save(str(source_dir / WINDOWS), overwrite=False)
    # load_concat_dataset writes a header pickle next to each fif on first load.
    # Create it here, before the checksum, so later reuse never writes.
    load_concat_dataset(source_dir / WINDOWS, preload=False, n_jobs=1)
    source_revision = _source_revision(seed, config)
    versions = _versions()
    manifest = {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "seed": int(seed),
        "preprocessing": preprocessing,
        "training": {key: config[key] for key in TRAINING_KEYS},
        "subjects": _subjects(config["n_subjects"]),
        "n_recordings": len(windows.datasets),
        "n_windows": len(windows),
        "versions": versions,
        "source_key": artifact_key(source_revision, preprocessing, (), versions),
        "files_sha256": _tree_sha256(source_dir / WINDOWS),
    }
    (source_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _read_manifest(source_dir):
    manifest = json.loads((Path(source_dir) / MANIFEST).read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{source_dir} is not a {SCHEMA} source.")
    expected = artifact_key(
        manifest["source_revision"], manifest["preprocessing"], (), manifest["versions"]
    )
    if manifest["source_key"] != expected:
        raise ValueError("Manifest source_key does not match its inputs.")
    return manifest


def load_source(source_dir):
    """Load the persisted source; verify it is unchanged; freeze signals."""
    source_dir = Path(source_dir)
    manifest = _read_manifest(source_dir)
    if _tree_sha256(source_dir / WINDOWS) != manifest["files_sha256"]:
        raise ValueError(f"{source_dir} changed after preparation; refusing reuse.")
    dataset = load_concat_dataset(source_dir / WINDOWS, preload=True, n_jobs=1)
    subjects = sorted({ds.description["subject"] for ds in dataset.datasets})
    if subjects != sorted(manifest["subjects"]):
        raise ValueError("Persisted subjects differ from the manifest.")
    for ds in dataset.datasets:
        # Shared by every fold view: any in-place write must fail loudly.
        ds.raw._data.flags.writeable = False
    return dataset, manifest


def split_schedule(subjects, *, n_folds: int, seed: int) -> list[dict]:
    """Subject-disjoint folds; every subject is held out exactly once."""
    subjects = sorted(subjects)
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2.")
    if n_folds > len(subjects):
        raise ValueError("n_folds cannot exceed the number of subjects.")
    order = np.random.default_rng(seed).permutation(len(subjects))
    schedule = []
    for fold, held_out in enumerate(np.array_split(order, n_folds)):
        test_ids = tuple(sorted(subjects[i] for i in held_out))
        train_ids = tuple(s for s in subjects if s not in test_ids)
        schedule.append({"fold": fold, "train_ids": train_ids, "test_ids": test_ids})
    return schedule


class ChannelStandardize:
    """Per-channel ``(x - mean) / std`` with frozen, training-fitted values."""

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1)
        self.mean.flags.writeable = False
        self.std.flags.writeable = False

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32, copy=False)


def _view(dataset, ids, transform):
    wanted = set(ids)
    # Shallow copies: a transform set on this view stays on these copies, while
    # the (read-only) signals and metadata are shared, not duplicated.
    children = [
        copy.copy(ds) for ds in dataset.datasets if ds.description["subject"] in wanted
    ]
    found = {ds.description["subject"] for ds in children}
    if found != wanted:
        raise ValueError(f"Subjects not in source: {sorted(wanted - found)}")
    view = BaseConcatDataset(children)
    view.transform = transform
    return view


def fold_datasets(dataset, train_ids, test_ids, transform=None):
    """Fold-private training/held-out views over a shared, immutable source."""
    if set(train_ids) & set(test_ids):
        raise ValueError("Training and held-out subjects overlap.")
    return _view(dataset, train_ids, transform), _view(dataset, test_ids, transform)


def fit_training_artifact(dataset, train_ids, manifest) -> dict:
    """Per-channel mean/std over training windows only (float64)."""
    train_view = _view(dataset, train_ids, None)
    x = np.stack([train_view[i][0] for i in range(len(train_view))]).astype(np.float64)
    mean = x.mean(axis=(0, 2))
    std = np.maximum(x.std(axis=(0, 2)), 1e-12)
    key = artifact_key(
        manifest["source_revision"],
        manifest["preprocessing"],
        tuple(train_ids),
        _versions(),
    )
    return {
        "key": key,
        "train_ids": list(train_ids),
        "n_windows": int(x.shape[0]),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }


def make_estimator(manifest) -> EEGClassifier:
    """Unfitted template; each fold fits ``sklearn.base.clone`` of it."""
    pre, training = manifest["preprocessing"], manifest["training"]
    return EEGClassifier(
        training["model"],
        module__n_chans=len(pre["ch_names"]),
        module__n_outputs=len(set(pre["target_mapping"].values())),
        module__n_times=pre["window_size_samples"],
        optimizer=torch.optim.AdamW,
        optimizer__lr=training["lr"],
        batch_size=training["batch_size"],
        max_epochs=training["max_epochs"],
        train_split=None,
        warm_start=False,
        device="cpu",
        verbose=0,
    )


def _fold_seed(seed, fold):
    return int(np.random.SeedSequence([int(seed), int(fold)]).generate_state(1)[0])


def _param_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def fit_fold(dataset, manifest, *, fold, train_ids, test_ids, seed, template) -> dict:
    """One independent fit: training-only artifact, fresh clone, own views."""
    torch.set_num_threads(FIT_THREADS)
    start = time.perf_counter()
    artifact = fit_training_artifact(dataset, train_ids, manifest)
    transform = ChannelStandardize(artifact["mean"], artifact["std"])
    train_set, test_set = fold_datasets(dataset, train_ids, test_ids, transform)
    estimator = clone(template)
    if estimator.warm_start:
        raise ValueError("Repeated fits must not warm-start.")
    fold_seed = _fold_seed(seed, fold)
    torch.manual_seed(fold_seed)
    fit_start = time.perf_counter()
    estimator.fit(train_set, y=None)
    fit_s = time.perf_counter() - fit_start
    batches = [b for epoch in estimator.history[:, "batches"] for b in epoch]
    train_batches = [b["train_batch_size"] for b in batches if "train_batch_size" in b]
    predictions = estimator.predict(test_set)
    targets = np.concatenate([np.asarray(ds.y) for ds in test_set.datasets])
    test_digest = hashlib.sha256()
    for i in range(len(test_set)):
        test_digest.update(test_set[i][0].tobytes())
    return {
        "fold": int(fold),
        "fold_seed": fold_seed,
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "n_train_windows": len(train_set),
        "n_test_windows": len(test_set),
        "artifact": artifact,
        "effective_samples": int(sum(train_batches)),
        "optimizer_updates": len(train_batches),
        "epochs": len(estimator.history),
        "fit_s": fit_s,
        "fold_total_s": time.perf_counter() - start,
        "param_sha256": _param_sha256(estimator.module_),
        "test_input_sha256": test_digest.hexdigest(),
        "test_predictions": [int(p) for p in predictions],
        "test_accuracy": float(np.mean(predictions == targets)),
        "pid": os.getpid(),
        "estimator": estimator,
    }


def _run_group(source_dir, folds, seed):
    """Worker body: load the persisted source once, fit its folds serially."""
    load_start = time.perf_counter()
    dataset, manifest = load_source(source_dir)
    load_s = time.perf_counter() - load_start
    template = make_estimator(manifest)
    fits = [
        fit_fold(
            dataset,
            manifest,
            fold=fold["fold"],
            train_ids=fold["train_ids"],
            test_ids=fold["test_ids"],
            seed=seed,
            template=template,
        )
        for fold in folds
    ]
    group = {
        "pid": os.getpid(),
        "folds": [fold["fold"] for fold in folds],
        "load_s": load_s,
        "peak_rss_bytes": _peak_rss(),
    }
    return group, fits


def _check_n_jobs(n_jobs):
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError(
            f"n_jobs must be a positive integer (explicit worker count), got {n_jobs!r}"
        )


def _summarize(schedule, outputs, parent_pid):
    groups = [group for group, _ in outputs]
    fits = sorted(
        (fit for _, batch in outputs for fit in batch), key=lambda f: f["fold"]
    )
    parent_peak = _peak_rss()
    worker_peaks = {}
    for group in groups:
        if group["pid"] != parent_pid:
            worker_peaks[group["pid"]] = max(
                worker_peaks.get(group["pid"], 0), group["peak_rss_bytes"]
            )
    return {
        "schedule": schedule,
        "groups": groups,
        "fits": fits,
        "parent_peak_rss_bytes": parent_peak,
        # Upper bound: sum of per-process high-water marks (parent + workers).
        "aggregate_peak_rss_bytes": parent_peak + sum(worker_peaks.values()),
        "effective_samples": sum(f["effective_samples"] for f in fits),
        "optimizer_updates": sum(f["optimizer_updates"] for f in fits),
    }


def run_repeated_fits(
    *, source_dir: Path, n_folds: int, n_jobs: int, seed: int
) -> dict:
    """Independent fits of every fold on one persisted source.

    Folds are assigned round-robin to at most ``n_jobs`` loky workers (the
    schedule, seeds and results do not depend on ``n_jobs``). With
    ``n_jobs=1`` joblib runs the single group in this process.
    """
    _check_n_jobs(n_jobs)
    manifest = _read_manifest(source_dir)
    schedule = split_schedule(manifest["subjects"], n_folds=n_folds, seed=seed)
    groups = [schedule[i::n_jobs] for i in range(min(n_jobs, len(schedule)))]
    start = time.perf_counter()
    outputs = Parallel(n_jobs=len(groups), backend="loky")(
        delayed(_run_group)(str(source_dir), group, seed) for group in groups
    )
    fits_wall_s = time.perf_counter() - start
    result = {
        "source_dir": str(source_dir),
        "source_key": manifest["source_key"],
        "n_folds": n_folds,
        "n_jobs": n_jobs,
        "seed": seed,
        "threads_per_fit": FIT_THREADS,
        "fits_wall_s": fits_wall_s,
    }
    result.update(_summarize(schedule, outputs, os.getpid()))
    return result


def run_cold_study(*, work_dir: Path, n_folds: int, seed: int, config=None) -> dict:
    """No-reuse baseline: prepare a fresh source for every fold, serially."""
    config = _config(config)
    schedule = split_schedule(
        _subjects(config["n_subjects"]), n_folds=n_folds, seed=seed
    )
    outputs, preparation_s, cleanup_s = [], 0.0, 0.0
    start = time.perf_counter()
    for fold in schedule:
        fold_dir = Path(work_dir) / f"cold-fold-{fold['fold']}"
        prep_start = time.perf_counter()
        manifest = prepare_source(fold_dir, seed=seed, config=config)
        preparation_s += time.perf_counter() - prep_start
        outputs.append(_run_group(fold_dir, [fold], seed))
        clean_start = time.perf_counter()
        shutil.rmtree(fold_dir)
        cleanup_s += time.perf_counter() - clean_start
    # Cleanup is excluded, as in reuse mode (its temporary directory is
    # removed after study_makespan_s is taken).
    total = time.perf_counter() - start - cleanup_s
    result = {
        "source_key": manifest["source_key"],
        "n_folds": n_folds,
        "n_jobs": 1,
        "seed": seed,
        "threads_per_fit": FIT_THREADS,
        "preparation_s": preparation_s,
        "cleanup_s": cleanup_s,
        "fits_wall_s": total - preparation_s,
        "study_makespan_s": total,
    }
    result.update(_summarize(schedule, outputs, os.getpid()))
    return result


PARITY_FIELDS = (
    "n_train_windows",
    "n_test_windows",
    "param_sha256",
    "test_predictions",
    "test_input_sha256",
    "artifact",
    "effective_samples",
    "optimizer_updates",
)


def check_run(result, *, n_folds: int) -> dict:
    """Within-run correctness: every scheduled fold fitted once, held out once.

    Cross-run parity (same seed and splits give identical fold digests) is
    checked by ``summarize_runs``.
    """
    failures = []
    folds = [fit["fold"] for fit in result["fits"]]
    if folds != list(range(n_folds)):
        failures.append(f"fits for folds {folds}, expected 0..{n_folds - 1}")
    if len(result["schedule"]) != n_folds:
        failures.append(f"{len(result['schedule'])} scheduled folds, not {n_folds}")
    held_out = [s for fold in result["schedule"] for s in fold["test_ids"]]
    if len(held_out) != len(set(held_out)):
        failures.append("a subject is held out more than once")
    schedule = {fold["fold"]: fold for fold in result["schedule"]}
    for fit in result["fits"]:
        planned = schedule.get(fit["fold"], {})
        if any(tuple(fit.get(key, ())) != tuple(planned.get(key, ()))
               for key in ("train_ids", "test_ids")):
            failures.append(f"fold {fit['fold']}: fit does not match schedule")
        artifact = fit.get("artifact", {})
        key = artifact.get("key", "")
        mean, std = artifact.get("mean", []), artifact.get("std", [])
        if (not isinstance(key, str) or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
                or not mean or len(mean) != len(std)
                or not np.isfinite(mean).all() or not np.isfinite(std).all()
                or any(value <= 0 for value in std)):
            failures.append(f"fold {fit['fold']}: invalid fitted artifact")
        if (type(fit.get("n_train_windows")) is not int
                or fit["n_train_windows"] <= 0
                or artifact.get("n_windows") != fit["n_train_windows"]
                or type(fit.get("n_test_windows")) is not int
                or fit["n_test_windows"] <= 0):
            failures.append(f"fold {fit['fold']}: invalid artifact/window count")
        if set(fit["train_ids"]) & set(fit["test_ids"]):
            failures.append(f"fold {fit['fold']}: training/held-out overlap")
        if fit["artifact"]["train_ids"] != list(fit["train_ids"]):
            failures.append(f"fold {fit['fold']}: artifact not fit on training IDs")
        if fit["effective_samples"] <= 0 or fit["optimizer_updates"] <= 0:
            failures.append(f"fold {fit['fold']}: no training updates")
    return {
        "status": "failed" if failures else "passed",
        "method": (
            "within-run: folds 0..n-1 fitted once, subject-disjoint, each subject "
            "held out once, artifact fit on training IDs, nonzero updates; "
            "cross-run digest parity is checked by summarize_runs"
        ),
        "failures": failures,
    }


def _stats(values):
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "values": list(values),
    }


def summarize_runs(runs, *, expected_runs: int, n_folds: int, ratio_pairs=()) -> dict:
    """Gate a study's runs; paired makespan ratios only when it qualifies.

    ``runs``: dicts with ``label``, ``rep``, ``exit_code`` and ``record`` (the
    CLI JSON, or None if missing). The study qualifies only if every label has
    exactly ``expected_runs`` distinct repeats (labels named in ``ratio_pairs``
    must be present), every run exited 0 with a record of status ``ok`` and
    ``n_folds`` fits, and all runs have identical schedules and per-fold
    digests (parity). Otherwise ``ratios`` is None.
    """
    failures, by_label = [], {}  # type: ignore[var-annotated]
    for run in runs:
        by_label.setdefault(run["label"], []).append(run)
        name = f"{run['label']} rep {run['rep']}"
        record = run["record"]
        if run["exit_code"] != 0:
            failures.append(f"{name}: exit code {run['exit_code']}")
        if record is None:
            failures.append(f"{name}: no record")
            continue
        if record.get("status") != "ok":
            failures.append(f"{name}: status {record.get('status')}")
        failures.extend(f"{name}: {failure}" for failure in
                        check_run(record, n_folds=n_folds)["failures"])
        folds = [fit["fold"] for fit in record["fits"]]
        if folds != list(range(n_folds)):
            failures.append(f"{name}: fits for folds {folds}")
    labels = set(by_label) | {label for pair in ratio_pairs for label in pair}
    for label in sorted(labels):
        reps = sorted((run["rep"] for run in by_label.get(label, [])), key=repr)
        if len(reps) != expected_runs or len(set(reps)) != expected_runs:
            failures.append(f"{label}: repeats {reps}, expected {expected_runs}")

    repeat_sets = [{run["rep"] for run in group} for group in by_label.values()]
    if repeat_sets and any(reps != repeat_sets[0] for reps in repeat_sets):
        failures.append("repeat labels differ between configurations")
    if any(type(run["rep"]) is not int or run["rep"] < 0 for run in runs):
        failures.append("repeat labels must be nonnegative integers")

    records = [run for run in runs if run["record"] is not None]
    mismatches = []
    if records:
        reference = records[0]["record"]
        for run in records:
            record = run["record"]
            if record["schedule"] != reference["schedule"]:
                mismatches.append(f"{run['label']} rep {run['rep']}: schedule")
            for a, b in zip(record["fits"], reference["fits"]):
                for field in PARITY_FIELDS:
                    if a[field] != b[field]:
                        mismatches.append(
                            f"{run['label']} rep {run['rep']} fold {a['fold']}: {field}"
                        )
    parity = {
        "status": "passed" if records and not mismatches else "failed",
        "fields": list(PARITY_FIELDS),
        "mismatches": mismatches,
    }
    if parity["status"] != "passed":
        failures.append("parity failed")

    configs = {}
    for label, group in sorted(by_label.items()):
        done = [run["record"] for run in group if run["record"] is not None]
        if done:
            configs[label] = {
                name: _stats([record[name] for record in done])
                for name in (
                    "study_makespan_s",
                    "preparation_s",
                    "fits_wall_s",
                    "aggregate_peak_rss_bytes",
                )
            }
    ratios = None
    if not failures:
        ratios = {}
        for num, den in ratio_pairs:
            a = {r["rep"]: r["record"]["study_makespan_s"] for r in by_label[num]}
            b = {r["rep"]: r["record"]["study_makespan_s"] for r in by_label[den]}
            ratios[f"{num} / {den}"] = _stats([a[k] / b[k] for k in sorted(a)])
    from benchmarks.acceleration._qualification import historical_only

    return historical_only({
        "status": "failed" if failures else "passed",
        "failures": failures,
        "parity": parity,
        "configs": configs,
        "ratios": ratios,
    }, "Study ratios are diagnostic; launch/source/core receipts need external audit.")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _provenance():
    paths = (
        "benchmarks/acceleration/bench_repeated_fits.py",
        "test/unit_tests/benchmarks/test_repeated_fit_contract.py",
    )
    changed = set(_git("diff", "HEAD", "--name-only").splitlines())
    changed.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in paths
        if name in changed
    }
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(changed),
        "dirty_source_hashes": hashes,
    }


def _affinity():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def physical_cores(cpus) -> int:
    """Distinct physical cores among logical CPU IDs (SMT siblings count once).

    Reads Linux sysfs topology; where it is unavailable each CPU counts as one
    core.
    """
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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("cold", "reuse"), required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="existing private directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="reuse an existing persisted source (mode=reuse; preparation skipped)",
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON overrides")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    _check_n_jobs(args.n_jobs)
    affinity = _affinity()
    cores = physical_cores(affinity)
    if args.n_jobs * FIT_THREADS > cores:
        raise SystemExit(
            f"n_jobs={args.n_jobs} oversubscribes {cores} physical cores "
            f"(CPU affinity {affinity})"
        )
    if args.mode == "cold" and (args.n_jobs != 1 or args.source_dir is not None):
        raise SystemExit("mode=cold is the serial no-reuse baseline: --n-jobs 1 only")
    if not args.work_dir.is_dir() or not args.output.parent.is_dir():
        raise SystemExit("--work-dir and the output parent must already exist")
    config = json.loads(args.config.read_text()) if args.config else None
    notes = [
        (
            "study_makespan_s excludes interpreter start, imports and removal of "
            "temporary sources; one-shot timing"
        ),
        (
            "aggregate_peak_rss_bytes sums per-process ru_maxrss high-water marks "
            "(parent + loky workers): an upper bound, not a concurrent sample"
        ),
        (
            f"each fit uses torch.set_num_threads({FIT_THREADS}); fold seeds from "
            "SeedSequence([seed, fold])"
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="repeated-fits-", dir=args.work_dir) as tmp:
        start = time.perf_counter()
        if args.mode == "cold":
            result = run_cold_study(
                work_dir=Path(tmp), n_folds=args.n_folds, seed=args.seed, config=config
            )
            notes.append(
                "cold: synthesize+preprocess+window+save once per fold, then load+fit"
            )
        else:
            preparation_s, source_dir = 0.0, args.source_dir
            if source_dir is None:
                source_dir = Path(tmp) / "source"
                prepare_start = time.perf_counter()
                prepare_source(source_dir, seed=args.seed, config=config)
                preparation_s = time.perf_counter() - prepare_start
            else:
                notes.append("reuse of an existing persisted source: preparation_s=0")
            result = run_repeated_fits(
                source_dir=source_dir,
                n_folds=args.n_folds,
                n_jobs=args.n_jobs,
                seed=args.seed,
            )
            result["preparation_s"] = preparation_s
            result["study_makespan_s"] = time.perf_counter() - start
    for fit in result["fits"]:
        del fit["estimator"]
    correctness = check_run(result, n_folds=args.n_folds)
    record = {
        "schema": SCHEMA,
        "mode": args.mode,
        "status": "ok" if correctness["status"] == "passed" else "failed",
        "correctness": correctness,
        "cpu_affinity": affinity,
        "physical_cores": cores,
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "config": _config(config),
        "environment": {"platform": platform.platform(), "versions": _versions()},
        "source": _provenance(),
        "notes": notes,
        **result,
    }
    from benchmarks.acceleration._qualification import historical_only

    historical_only(record, "No automatic launch/source/policy attestation.")
    partial = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    os.replace(partial, args.output)
    summary = {k: record[k] for k in ("mode", "status", "n_jobs", "study_makespan_s")}
    summary.update(
        preparation_s=record["preparation_s"],
        aggregate_peak_rss_bytes=record["aggregate_peak_rss_bytes"],
        effective_samples=record["effective_samples"],
        optimizer_updates=record["optimizer_updates"],
    )
    print(json.dumps(summary))
    if record["status"] != "ok":
        raise SystemExit(f"correctness failed: {correctness['failures']}")


if __name__ == "__main__":
    main()

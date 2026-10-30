"""Leakage and fresh-state contract for the repeated-fit benchmark recipe.

These are correctness checks on tiny synthetic data, not performance claims.
"""

import copy
import itertools
import os

import numpy as np
import pytest

from benchmarks.acceleration import bench_repeated_fits as rf
from braindecode.datasets import BaseConcatDataset

TINY = {
    "n_subjects": 4,
    "n_trials": 2,
    "ch_names": ["C3", "Cz", "C4", "Pz"],
    "units": "uV",
    "sfreq": 128.0,
    "n_times": 384,
    "l_freq": 4.0,
    "h_freq": 30.0,
    "window_size_samples": 128,
    "window_stride_samples": 128,
    "target_mapping": {"left": 0, "right": 1},
    "model": "ShallowFBCSPNet",
    "max_epochs": 2,
    "batch_size": 4,
    "lr": 1e-3,
}


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Restore caller RNG, thread count and grad mode after each case."""
    yield


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory):
    source_dir = tmp_path_factory.mktemp("tiny-source")
    manifest = rf.prepare_source(source_dir, seed=0, config=TINY)
    return source_dir, manifest


@pytest.fixture(scope="module")
def other_source(tmp_path_factory):
    # Same layout and subject IDs, different signal values.
    source_dir = tmp_path_factory.mktemp("other-source")
    manifest = rf.prepare_source(source_dir, seed=1, config=TINY)
    return source_dir, manifest


def _key_inputs(manifest):
    return {
        "source_revision": manifest["source_revision"],
        "preprocessing": copy.deepcopy(manifest["preprocessing"]),
        "split_ids": ("sub-00", "sub-01"),
        "versions": dict(manifest["versions"]),
    }


def _mix(base, other, replace_ids):
    """Base children, except the subjects in replace_ids taken from other."""
    children = []
    for mine, theirs in zip(base.datasets, other.datasets):
        assert mine.description["subject"] == theirs.description["subject"]
        use_other = mine.description["subject"] in replace_ids
        children.append(theirs if use_other else mine)
    return BaseConcatDataset(children)


def _state_arrays(estimator):
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in estimator.module_.state_dict().items()
    }


@pytest.mark.slow
def test_artifact_key_tracks_scientific_inputs(tiny_source, other_source):
    source_dir, manifest = tiny_source
    base = _key_inputs(manifest)
    key = rf.artifact_key(**base)
    assert isinstance(key, str) and len(key) == 64
    # Canonical: equal content in a different dict insertion order, same key.
    reordered = dict(base)
    reordered["preprocessing"] = dict(reversed(list(base["preprocessing"].items())))
    reordered["versions"] = dict(reversed(list(base["versions"].items())))
    assert rf.artifact_key(**reordered) == key

    def changed(**updates):
        inputs = copy.deepcopy(base)
        for name, update in updates.items():
            if callable(update):
                update(inputs[name])
            else:
                inputs[name] = update
        return rf.artifact_key(**inputs)

    def set_item(key_name, value):
        def apply(mapping):
            mapping[key_name] = value

        return apply

    preprocessing = base["preprocessing"]
    variants = {
        "source revision": changed(source_revision=base["source_revision"] + "-x"),
        "channel order": changed(
            preprocessing=set_item("ch_names", preprocessing["ch_names"][::-1])
        ),
        "units": changed(preprocessing=set_item("units", "V")),
        "sfreq": changed(preprocessing=set_item("sfreq", preprocessing["sfreq"] * 2)),
        "filter": changed(preprocessing=set_item("h_freq", 20.0)),
        "window size": changed(
            preprocessing=set_item(
                "window_size_samples", preprocessing["window_size_samples"] // 2
            )
        ),
        "window stride": changed(
            preprocessing=set_item(
                "window_stride_samples", preprocessing["window_stride_samples"] // 2
            )
        ),
        "target mapping": changed(
            preprocessing=set_item("target_mapping", {"left": 1, "right": 0})
        ),
        "version": changed(versions=set_item("braindecode", "0.0.0+other")),
        "training split": changed(split_ids=("sub-00", "sub-02")),
    }
    for name, other_key in variants.items():
        assert other_key != key, f"{name} change did not change the key"
    assert len(set(variants.values())) == len(variants)

    # Held-out-only perturbation: same training IDs, different held-out data.
    train_ids = ("sub-00", "sub-01")
    test_ids = ("sub-02", "sub-03")
    dataset, loaded_manifest = rf.load_source(source_dir)
    other, _ = rf.load_source(other_source[0])
    held_out_perturbed = _mix(dataset, other, set(test_ids))
    fitted_training_artifact_before = rf.fit_training_artifact(
        dataset, train_ids, loaded_manifest
    )
    fitted_training_artifact_after = rf.fit_training_artifact(
        held_out_perturbed, train_ids, loaded_manifest
    )
    assert fitted_training_artifact_before == fitted_training_artifact_after
    # Non-vacuous: the same perturbation on a training subject is detected.
    train_perturbed = _mix(dataset, other, {"sub-01"})
    assert (
        rf.fit_training_artifact(train_perturbed, train_ids, loaded_manifest)
        != fitted_training_artifact_before
    )

    # The fitted model must not depend on held-out data either.
    template = rf.make_estimator(loaded_manifest)
    common = {"fold": 0, "train_ids": train_ids, "test_ids": test_ids, "seed": 0}
    before = rf.fit_fold(dataset, loaded_manifest, template=template, **common)
    after = rf.fit_fold(
        held_out_perturbed, loaded_manifest, template=template, **common
    )
    assert before["artifact"] == after["artifact"]
    assert before["param_sha256"] == after["param_sha256"]
    before_state = _state_arrays(before["estimator"])
    after_state = _state_arrays(after["estimator"])
    assert before_state.keys() == after_state.keys()
    for name in before_state:
        np.testing.assert_array_equal(before_state[name], after_state[name])
    # The held-out inputs really differed between the two fits.
    assert before["test_input_sha256"] != after["test_input_sha256"]
    # The template itself is never fitted.
    assert not hasattr(template, "module_")


@pytest.mark.slow
def test_repeated_fits_use_fresh_estimators(tiny_source):
    source_dir, manifest = tiny_source
    first = rf.run_repeated_fits(source_dir=source_dir, n_folds=2, n_jobs=1, seed=0)
    second = rf.run_repeated_fits(source_dir=source_dir, n_folds=2, n_jobs=1, seed=0)

    # Fixed split schedule: deterministic, subject-disjoint, each held out once.
    expected = rf.split_schedule(manifest["subjects"], n_folds=2, seed=0)
    assert first["schedule"] == expected == second["schedule"]
    held_out = [s for fold in expected for s in fold["test_ids"]]
    assert sorted(held_out) == sorted(manifest["subjects"])
    for fold in expected:
        assert not set(fold["train_ids"]) & set(fold["test_ids"])
        assert set(fold["train_ids"]) | set(fold["test_ids"]) == set(
            manifest["subjects"]
        )
    assert [f["fold"] for f in first["fits"]] == [0, 1]

    estimators = [fit["estimator"] for fit in first["fits"]]
    estimators += [fit["estimator"] for fit in second["fits"]]
    for estimator in estimators:
        assert estimator.warm_start is False
        own = {id(p) for p in estimator.module_.parameters()}
        for group in estimator.optimizer_.param_groups:
            assert {id(p) for p in group["params"]} <= own
    for a, b in itertools.combinations(estimators, 2):
        assert a is not b
        assert a.module_ is not b.module_
        assert a.optimizer_ is not b.optimizer_
        pointers_a = {p.data_ptr() for p in a.module_.parameters()}
        pointers_b = {p.data_ptr() for p in b.module_.parameters()}
        assert not pointers_a & pointers_b
    # Repeating the same seeds/splits gives the same fitted parameters.
    assert [f["param_sha256"] for f in first["fits"]] == [
        f["param_sha256"] for f in second["fits"]
    ]
    # Different folds really are different fits.
    assert first["fits"][0]["param_sha256"] != first["fits"][1]["param_sha256"]
    # Each run carries an explicit correctness status.
    assert rf.check_run(first, n_folds=2)["status"] == "passed"
    assert rf.check_run(first, n_folds=3)["status"] == "failed"
    truncated = dict(first, fits=first["fits"][:1])
    assert rf.check_run(truncated, n_folds=2)["status"] == "failed"

    # BaseConcatDataset.split views share child objects: setting a transform
    # on one view mutates every view (this is why the recipe copies children).
    dataset, _ = rf.load_source(source_dir)
    naive = dataset.split(by="subject")
    assert naive["sub-00"].datasets[0] is dataset.datasets[0]

    fold_a, fold_b = expected
    train_a, test_a = rf.fold_datasets(dataset, fold_a["train_ids"], fold_a["test_ids"])
    train_b, test_b = rf.fold_datasets(dataset, fold_b["train_ids"], fold_b["test_ids"])
    marker_a = rf.ChannelStandardize(np.zeros(4), np.ones(4))
    marker_b = rf.ChannelStandardize(np.ones(4), np.full(4, 2.0))
    train_a.transform = marker_a
    test_a.transform = marker_a
    train_b.transform = marker_b
    test_b.transform = marker_b
    assert all(t is marker_a for t in train_a.transform + test_a.transform)
    assert all(t is marker_b for t in train_b.transform + test_b.transform)
    assert all(t is None for t in dataset.transform)
    # Shared signal arrays are immutable, so no fold can write into another.
    originals = {id(ds) for ds in dataset.datasets}
    views = train_a.datasets + test_a.datasets + train_b.datasets + test_b.datasets
    assert not originals & {id(ds) for ds in views}
    with pytest.raises(ValueError):
        train_a.datasets[0].raw._data[0, 0] = 1.0
    # Fold B sees its own transform over the same shared signal.
    view_b = test_b.datasets[0]
    shared = next(ds for ds in dataset.datasets if ds.raw is view_b.raw)
    np.testing.assert_allclose(view_b[0][0], (shared[0][0] - 1.0) / 2.0, rtol=1e-6)


def test_repeated_fits_reject_unbounded_workers(tiny_source):
    source_dir, _ = tiny_source
    for bad in (-1, 0, 1.5, None):
        with pytest.raises(ValueError, match="n_jobs"):
            rf.run_repeated_fits(source_dir=source_dir, n_folds=2, n_jobs=bad, seed=0)
    with pytest.raises(ValueError, match="n_folds"):
        rf.run_repeated_fits(source_dir=source_dir, n_folds=1, n_jobs=1, seed=0)


@pytest.mark.slow
def test_repeated_fits_parity_across_worker_counts(tiny_source):
    source_dir, _ = tiny_source
    serial = rf.run_repeated_fits(source_dir=source_dir, n_folds=2, n_jobs=1, seed=0)
    parallel = rf.run_repeated_fits(source_dir=source_dir, n_folds=2, n_jobs=2, seed=0)
    assert serial["schedule"] == parallel["schedule"]
    for a, b in zip(serial["fits"], parallel["fits"]):
        assert a["fold"] == b["fold"]
        assert a["artifact"] == b["artifact"]
        assert a["param_sha256"] == b["param_sha256"]
        assert a["test_predictions"] == b["test_predictions"]
        assert a["effective_samples"] == b["effective_samples"] > 0
        assert a["optimizer_updates"] == b["optimizer_updates"] > 0
    assert all(fit["pid"] == os.getpid() for fit in serial["fits"])
    assert all(fit["pid"] != os.getpid() for fit in parallel["fits"])
    assert parallel["aggregate_peak_rss_bytes"] > parallel["parent_peak_rss_bytes"]


def test_source_key_tracks_generator_counts(tiny_source, tmp_path):
    _, manifest = tiny_source
    fewer = rf.prepare_source(
        tmp_path / "fewer-trials", seed=0, config={**TINY, "n_trials": 3}
    )
    assert fewer["preprocessing"] == manifest["preprocessing"]
    assert fewer["source_revision"] != manifest["source_revision"]
    assert fewer["source_key"] != manifest["source_key"]
    train_ids = ("sub-00", "sub-01")

    def key(m):
        return rf.artifact_key(
            m["source_revision"], m["preprocessing"], train_ids, m["versions"]
        )

    assert key(fewer) != key(manifest)
    more = rf.prepare_source(
        tmp_path / "more-subjects", seed=0, config={**TINY, "n_subjects": 5}
    )
    assert more["source_key"] not in {fewer["source_key"], manifest["source_key"]}


def test_physical_core_count_is_bounded(monkeypatch):
    # Four logical CPUs in two packages; siblings share one physical core.
    from pathlib import Path

    topology = {0: (0, 0), 1: (0, 0), 2: (1, 0), 3: (1, 1)}

    def read_text(path):
        cpu = int(path.parent.parent.name.removeprefix("cpu"))
        package, core = topology[cpu]
        return str(package if path.name == "physical_package_id" else core)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert rf.physical_cores([0, 1, 2, 3]) == 3
    assert rf.physical_cores([0, 1]) == 1
    assert rf.physical_cores([2, 3]) == 2


def test_cpu_topology_fallback_without_linux_apis(monkeypatch):
    from pathlib import Path

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", missing)
    assert rf._affinity() == [0, 1, 2]
    assert rf.physical_cores(rf._affinity()) == 3
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert rf._affinity() == [0]


def _fake_record(digest="a", folds=2, status="ok", makespan=10.0):
    fits = [
        {
            "fold": fold,
            "param_sha256": f"{digest}{fold}",
            "test_predictions": [0, 1],
            "test_input_sha256": f"x{fold}",
            "train_ids": [f"train{fold}"],
            "test_ids": [f"test{fold}"],
            "n_train_windows": 4,
            "n_test_windows": 2,
            "artifact": {"key": "a" * 64, "train_ids": [f"train{fold}"],
                         "n_windows": 4, "mean": [0.0], "std": [1.0]},
            "effective_samples": 8,
            "optimizer_updates": 2,
        }
        for fold in range(folds)
    ]
    return {
        "status": status,
        "schedule": [{"fold": fold, "train_ids": [f"train{fold}"],
                      "test_ids": [f"test{fold}"]} for fold in range(2)],
        "fits": fits,
        "study_makespan_s": makespan,
        "preparation_s": 1.0,
        "fits_wall_s": makespan - 1.0,
        "aggregate_peak_rss_bytes": 1000,
        "effective_samples": 8 * folds,
        "optimizer_updates": 2 * folds,
    }


def _fake_runs(n_runs=3):
    return [
        {
            "label": label,
            "rep": rep,
            "exit_code": 0,
            "record": _fake_record(makespan=makespan),
        }
        for label, makespan in (("cold-j1", 20.0), ("reuse-j1", 10.0))
        for rep in range(1, n_runs + 1)
    ]


def test_study_summary_withholds_ratios_unless_qualified():
    pairs = (("reuse-j1", "cold-j1"),)
    kwargs = {"expected_runs": 3, "n_folds": 2, "ratio_pairs": pairs}
    good = rf.summarize_runs(_fake_runs(), **kwargs)
    assert good["status"] == "passed" and good["parity"]["status"] == "passed"
    assert good["ratios"]["reuse-j1 / cold-j1"]["median"] == pytest.approx(0.5)

    def broken(mutate):
        runs = copy.deepcopy(_fake_runs())
        mutate(runs)
        summary = rf.summarize_runs(runs, **kwargs)
        assert summary["status"] == "failed"
        assert summary["ratios"] is None
        assert summary["failures"]
        return summary

    def mismatch(runs):
        runs[4]["record"]["fits"][1]["param_sha256"] = "other"

    assert broken(mismatch)["parity"]["status"] == "failed"
    broken(lambda runs: runs.pop())  # only 2 runs of one config
    broken(lambda runs: runs[0].update(exit_code=1))
    broken(lambda runs: runs[0].update(record=None))
    broken(lambda runs: runs[1]["record"].update(status="failed"))
    broken(lambda runs: runs[2]["record"].update(fits=runs[2]["record"]["fits"][:1]))
    broken(lambda runs: runs[4].update(rep=1))  # duplicate rep, none for rep 2
    missing = broken(lambda runs: runs.__delitem__(slice(3, 6)))  # no reuse-j1
    assert any("reuse-j1" in failure for failure in missing["failures"])

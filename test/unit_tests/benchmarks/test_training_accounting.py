"""Contract checks for the training accounting runner; tiny timings are not claims."""

import functools
import json
import math
import os

import pytest
import torch

from benchmarks.acceleration._common import write_result

pytestmark = pytest.mark.slow

N_SAMPLES, BATCH, VALID_BATCH = 10, 4, 3
CASE = {"model_name": "ShallowFBCSPNet", "n_samples": N_SAMPLES, "batch_size": BATCH}
SMALL = {"n_chans": 2, "n_times": 1000, "valid_batch_size": VALID_BATCH}
WARMUP = {
    "warmup_min_s": 0.2,
    "warmup_max_s": 3.0,
    "warmup_window": 3,
    "warmup_rel_tol": 0.5,
}
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Restore caller RNG, thread count and grad mode after each case."""
    yield


@functools.cache
def _run(n_metrics, device="cpu"):
    from benchmarks.acceleration.bench_training import run_case

    return run_case(
        n_metrics=n_metrics,
        device=device,
        min_run_time=0.05,
        profile_epochs=2,
        **CASE,
        **SMALL,
        **WARMUP,
    )


def _expected(n_metrics):
    # drop_last=True (braindecode default): 2 full batches, 2 samples dropped;
    # reevaluation covers all 10 samples in ceil(10 / 3) eval batches.
    return {
        "train_batches": 2,
        "train_forwards": 2,
        "optimizer_steps": 2,
        "train_samples": 8,
        "dropped_train_samples": 2,
        "eval_forwards": math.ceil(N_SAMPLES / VALID_BATCH) if n_metrics else 0,
        "eval_samples": N_SAMPLES if n_metrics else 0,
    }


@pytest.mark.parametrize("n_metrics", [0, 1, 2])
def test_record_is_strict_v1_with_exact_counts(n_metrics, tmp_path):
    record = _run(n_metrics)["record"]
    write_result(record, tmp_path / "record.json")  # strict schema validation
    assert record["status"] == "ok"
    assert record["correctness"]["status"] == "passed"
    parameters = record["config"]["parameters"]
    assert parameters["drop_last"] is True
    assert parameters["n_metrics"] == n_metrics
    assert parameters["valid_batch_size"] == VALID_BATCH
    metrics = record["correctness"]["metrics"]
    assert metrics["expected_per_epoch"] == _expected(n_metrics)
    assert metrics["profiled_per_epoch"] == [_expected(n_metrics)] * 2
    headline = metrics["headline"]
    epochs = headline["epochs"]
    assert epochs >= 1
    assert headline["train_batches_per_epoch"] == [2] * epochs
    assert headline["train_samples_per_epoch"] == [8] * epochs
    assert headline["optimizer_steps"] == 2 * epochs
    assert len(headline["skorch_epoch_dur_s"]) == epochs


@pytest.mark.parametrize("n_metrics", [0, 1, 2])
def test_train_metrics_match_final_weight_reevaluation(n_metrics):
    check = _run(n_metrics)["record"]["correctness"]["metrics"]["final_weight_check"]
    names = ["train_accuracy", "train_balanced_accuracy"][:n_metrics]
    assert sorted(check) == sorted(names)
    for values in check.values():
        assert values["history"] == values["recomputed"]


@pytest.mark.parametrize("n_metrics", [0, 1, 2])
def test_profile_is_separate_exclusive_and_bounded(n_metrics):
    from benchmarks.acceleration.bench_training import EXCLUSIVE_PHASES

    profile = _run(n_metrics)["profile"]
    assert profile["separate_from_headline"] is True
    assert profile["synchronized_markers"] is False  # CPU: no device sync needed
    assert len(profile["epochs"]) == 2
    for epoch in profile["epochs"]:
        exclusive = epoch["exclusive_s"]
        assert tuple(exclusive) == EXCLUSIVE_PHASES
        assert all(value >= 0 for value in exclusive.values())
        attributed = sum(exclusive.values())
        assert attributed <= epoch["wall_s"]
        assert epoch["unattributed_s"] == pytest.approx(
            epoch["wall_s"] - attributed, abs=1e-12
        )
        # Inclusive spans overlap exclusive ones and are never summed with them.
        inclusive = epoch["inclusive_s"]
        assert inclusive["optimizer_step_with_closure"] >= (
            exclusive["train_forward"] + exclusive["train_backward"]
        )
        assert inclusive["train_scoring_callbacks"] == pytest.approx(
            exclusive["train_scoring_forward"] + exclusive["train_scoring_other"],
            abs=1e-12,
        )
        assert (exclusive["train_scoring_forward"] > 0) == (n_metrics > 0)
        assert epoch["counts"] == _expected(n_metrics)


def test_memory_and_raw_samples_are_reported():
    result = _run(1)
    record, profile = result["record"], result["profile"]
    timing = record["timing"]
    assert len(timing["raw_block_s"]) == timing["measurement_blocks"] >= 1
    assert record["memory"]["peak_cuda_allocated_bytes"] is None
    memory = dict(profile["memory"])
    assert any(json.dumps(memory, sort_keys=True) in note for note in record["notes"])
    # Raw counters only: Linux ru_maxrss is updated lazily and may trail a
    # current-RSS sample, so no ordering between them is asserted.
    assert set(memory) == {
        "rss_after_headline_bytes",
        "rss_after_profile_bytes",
        "ru_maxrss_bytes",
        "profile_peak_cuda_allocated_bytes",
    }
    assert memory.pop("profile_peak_cuda_allocated_bytes") is None  # CPU
    assert all(type(value) is int and value > 0 for value in memory.values())
    assert all(epoch["wall_s"] > 0 for epoch in profile["epochs"])


def test_warmup_policy_is_steady_state_and_recorded():
    record = _run(1)["record"]
    policy = record["config"]["parameters"]["warmup"]
    assert policy == {
        "policy": "time floor then stability window",
        "min_s": WARMUP["warmup_min_s"],
        "max_s": WARMUP["warmup_max_s"],
        "window": WARMUP["warmup_window"],
        "rel_tol": WARMUP["warmup_rel_tol"],
    }
    observed = record["correctness"]["metrics"]["warmup"]
    calls = observed["calls_s"]
    assert len(calls) >= WARMUP["warmup_window"]
    assert observed["total_s"] == pytest.approx(sum(calls), rel=1e-9)
    assert observed["total_s"] >= WARMUP["warmup_min_s"]
    last = calls[-WARMUP["warmup_window"] :]
    stable = (max(last) - min(last)) / min(last) <= WARMUP["warmup_rel_tol"]
    assert observed["steady"] is stable
    if not stable:  # only a timeout may end an unstable warmup
        assert observed["total_s"] >= WARMUP["warmup_max_s"]
    # Warmup calls are discarded but run on the timed net: epochs add up.
    headline = record["correctness"]["metrics"]["headline"]
    assert headline["epochs"] > len(calls)
    assert any("warmup" in note.lower() for note in record["notes"])


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity"), reason="Linux CPU affinity API"
)
def test_cpu_placement_is_recorded():
    record = _run(0)["record"]
    placement = record["config"]["parameters"]["cpu_placement"]
    expected = (sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                else list(range(os.cpu_count() or 1)))
    assert placement["affinity"] == expected
    assert 1 <= placement["physical_cores"] <= len(placement["affinity"])
    assert placement["torch_threads"] == record["config"]["threads"]
    assert set(placement) == {"affinity", "physical_cores", "torch_threads"}


@CUDA
@pytest.mark.parametrize("n_metrics", [0, 1, 2])
def test_cuda_case_runs_with_synchronized_profile(n_metrics, tmp_path):
    result = _run(n_metrics, "cuda")
    record, profile = result["record"], result["profile"]
    write_result(record, tmp_path / "record.json")
    assert record["status"] == "ok"
    assert record["environment"]["device"].startswith("cuda")
    assert record["memory"]["peak_cuda_allocated_bytes"] > 0
    assert profile["synchronized_markers"] is True
    assert profile["memory"]["profile_peak_cuda_allocated_bytes"] > 0
    for epoch in profile["epochs"]:
        assert epoch["counts"] == _expected(n_metrics)
        assert epoch["unattributed_s"] >= 0

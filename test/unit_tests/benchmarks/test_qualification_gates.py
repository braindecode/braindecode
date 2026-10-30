"""Fail-closed benchmark gates; tiny fixtures are not qualification evidence."""
import pytest
import torch

from benchmarks.acceleration import bench_execution as execution
from benchmarks.acceleration import bench_repeated_fits as fits
from benchmarks.acceleration import bench_stream_replay as replay


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Use the combined suite's shared caller-state isolation convention."""
    yield


def fold_record():
    return {
        "schedule": [{"fold": 0, "train_ids": ["a"], "test_ids": ["b"]}],
        "fits": [{"fold": 0, "train_ids": ["a"], "test_ids": ["b"],
                  "artifact": {"train_ids": ["a"], "key": "a" * 64,
                               "n_windows": 2, "mean": [0.0], "std": [1.0]},
                  "n_train_windows": 2, "n_test_windows": 1,
                  "effective_samples": 2, "optimizer_updates": 1}],
    }


@pytest.mark.parametrize("mutation", ["schedule", "windows", "mean", "std", "key"])
def test_fit_gate_rejects_invalid_scientific_record(mutation):
    record = fold_record()
    assert fits.check_run(record, n_folds=1)["status"] == "passed"
    fit = record["fits"][0]
    if mutation == "schedule":
        record["schedule"][0]["train_ids"] = ["other"]
    elif mutation == "windows":
        fit["artifact"]["n_windows"] = 3
    elif mutation == "mean":
        fit["artifact"]["mean"] = [float("nan")]
    elif mutation == "std":
        fit["artifact"]["std"] = [0.0]
    else:
        fit["artifact"]["key"] = ""
    assert fits.check_run(record, n_folds=1)["status"] == "failed"


def test_repeat_labels_fail_closed_without_keyerror():
    from test.unit_tests.benchmarks.test_repeated_fit_contract import _fake_runs
    runs = _fake_runs()
    for run in runs:
        if run["label"] == "reuse-j1":
            run["rep"] += 3
    result = fits.summarize_runs(runs, expected_runs=3, n_folds=2,
                                 ratio_pairs=(("reuse-j1", "cold-j1"),))
    assert result["status"] == "failed" and result["ratios"] is None


@pytest.mark.parametrize("field", ["warmup", "cores", "raw", "finite"])
def test_replay_pooling_gates(field):
    from test.unit_tests.benchmarks.test_stream_replay import _summary_record
    records = [_summary_record() for _ in range(5)]
    for r in records:
        r["warmup"] = {"steady": True}
        r["environment"].update(physical_cores=2, requested_threads=1, torch_threads=1)
        for key in ("latency_s", "model_s"):
            r["compute_latency"]["raw_" + key] = [1.0] * 10
    if field == "warmup":
        records[0]["warmup"]["steady"] = False
    elif field == "cores":
        records[0]["environment"]["requested_threads"] = 4
    elif field == "raw":
        records[0]["compute_latency"].pop("raw_latency_s")
    else:
        records[0]["compute_latency"]["raw_model_s"][0] = float("nan")
    result = replay.summarize_runs(records, expected_runs=5)
    assert result["failures"] and result["latency_s"] is None


@pytest.mark.parametrize("label", [True, "2", -1])
def test_invalid_repeat_labels_are_failures(label):
    from test.unit_tests.benchmarks.test_repeated_fit_contract import _fake_runs
    runs = _fake_runs()
    runs[0]["rep"] = label
    result = fits.summarize_runs(runs, expected_runs=3, n_folds=2)
    assert result["status"] == "failed" and result["ratios"] is None


def test_compile_timeout_sentinel():
    assert execution._timed_out(execution.CompileTimeout("deadline"))
    assert not execution._timed_out(TimeoutError("dataset timeout"))


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_compile_timeout_entry_validation(timeout):
    with pytest.raises(ValueError, match="compile_timeout_s"):
        execution.run_case("ShallowFBCSPNet", mode="compiled", device="cpu",
                           batch_size=2, n_times=32, compile_timeout_s=timeout)


def test_compiled_output_perturbation_rejected(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda model, **kw: lambda x: model(x) + 1)
    record = execution.run_case("ShallowFBCSPNet", mode="compiled", device="cpu",
                                batch_size=2, n_times=32, min_run_time=0.001)
    assert record["status"] == "failed"
    assert record["correctness"]["metrics"]["failed_check"] == "dropout_off_parity"


def test_equal_work_executes_and_rejects_skipped_update():
    model = torch.nn.Linear(2, 2)
    x, y = torch.ones(3, 2), torch.zeros(3, dtype=torch.long)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    check = execution._check_equal_work(model, model, x, y, optimizer, 1)
    assert check["effective_samples"] == 3 and check["optimizer_updates"] == 1
    with pytest.raises((AssertionError, ValueError)):
        execution._check_equal_work(model, lambda x: model(x[:1]), x, y, optimizer, 1)


def test_timing_recompiles_recorded(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda model, **kw: model)
    count = [0]
    monkeypatch.setattr(execution, "_graph_count", lambda: count[0])
    def timing(record, run, model, duration):
        count[0] += 1
        record["status"] = "ok"
        return record
    monkeypatch.setattr(execution, "_finish_timing", timing)
    record = execution.run_case("ShallowFBCSPNet", mode="compiled", device="cpu",
                                batch_size=2, n_times=32, min_run_time=0.001)
    diag = record["config"]["parameters"]["timing_compile_diagnostics"]
    assert diag["recompiled"] and diag["unique_graphs_after"] > diag["unique_graphs_before"]


def test_linux_only_rss_guard(monkeypatch):
    from benchmarks.acceleration import _qualification
    monkeypatch.setattr(_qualification.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        _qualification.require_linux_rss()


def test_missing_mixup_history_has_actionable_error(monkeypatch):
    import subprocess

    from benchmarks.acceleration import bench_mixup
    def missing(*args):
        raise subprocess.CalledProcessError(128, args)
    monkeypatch.setattr(bench_mixup, "_git", missing)
    with pytest.raises(RuntimeError, match="exact Git object.*Never substitute"):
        bench_mixup._historical_mixup(bench_mixup.EXPERIMENTAL)


def test_portable_topology_with_smt_and_absent_affinity(monkeypatch):
    monkeypatch.delattr(fits.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(fits.os, "cpu_count", lambda: 4)
    assert fits._affinity() == [0, 1, 2, 3]
    def read(path, *args, **kwargs):
        cpu = int(str(path).split("/cpu/cpu")[1].split("/")[0])
        return "0" if path.name == "physical_package_id" else str(cpu // 2)
    monkeypatch.setattr(fits.Path, "read_text", read)
    assert fits.physical_cores(fits._affinity()) == 2


def test_historical_records_cannot_qualify_new_results():
    from benchmarks.acceleration._qualification import historical_only
    record = historical_only({"status": "ok"}, "external launch/policy evidence")
    assert record["qualification"]["status"] == "historical_only"
    assert record["qualification"]["new_claim_eligible"] is False

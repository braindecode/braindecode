"""Contract checks; synthetic timings are not performance claims."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.acceleration import _common


@pytest.fixture
def result():
    return {
        "schema_version": 1,
        "case_id": "numpy-add",
        "status": "ok",
        "implementation": "reference",
        "config": {
            "inputs": [
                {
                    "name": "x",
                    "shape": [2, 3],
                    "dtype": "float32",
                    "strides": [12, 4],
                    "stride_unit": "bytes",
                }
            ],
            "seed": 0,
            "threads": 1,
            "workers": 0,
            "cache_policy": "prior reference; cache state uncontrolled",
            "autograd": "not_applicable",
            "precision": "float32",
            "parameters": {},
        },
        "environment": {
            "python": "3.11",
            "platform": "synthetic",
            "dependencies": {"numpy": "2"},
            "device": "cpu",
            "device_name": "synthetic CPU",
        },
        "source": {"commit": "4" * 40, "dirty": False, "dirty_source_hashes": {}},
        "correctness": {"status": "passed", "method": "array equality", "metrics": {}},
        "timing": {
            "first_call_s": 0.2,
            "median_s": 0.1,
            "iqr_s": 0.0,
            "raw_block_s": [0.2, 0.2],
            "measurement_blocks": 2,
            "calls_per_block": 2,
            "warmup": 5,
            "min_run_time_s": 0.1,
        },
        "memory": {"peak_cuda_allocated_bytes": None},
        "notes": [],
    }


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -float("inf"), 0])
def test_measure_step_rejects_nonfinite_duration(duration):
    with pytest.raises(ValueError, match="positive and finite"):
        _common.measure_step(
            lambda: None, device=torch.device("cpu"), min_run_time=duration
        )


@pytest.mark.parametrize("warmup", [-1, 1.2, True])
def test_invalid_warmup(warmup):
    with pytest.raises(ValueError, match="warmup"):
        _common.measure_step(lambda: None, device=torch.device("cpu"), warmup=warmup)


@pytest.fixture
def timer_spy(monkeypatch):
    events = []

    def timer(*, stmt, globals, timer, num_threads):
        assert stmt == "run()"
        assert num_threads == torch.get_num_threads()
        assert timer is _common.perf_counter  # no default CUDA timer on CPU

        def blocked_autorange(*, min_run_time):
            events.append("blocks")
            globals["run"]()
            return SimpleNamespace(
                median=0.1, iqr=0.02, raw_times=[0.2, 0.24], number_per_run=2
            )

        return SimpleNamespace(blocked_autorange=blocked_autorange)

    monkeypatch.setattr(_common, "Timer", timer)
    return events


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_measure_step_sync_dispatch(monkeypatch, timer_spy, device):
    events = timer_spy
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: events.append(str(d)))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("mps"))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda d: events.append("reset:" + str(d)),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 123)
    measured = _common.measure_step(
        lambda: events.append("run"), device=torch.device(device), warmup=1
    )
    expected = ["run", "run", "blocks", "run"]
    if device != "cpu":
        expected = [device, "run", device, "run", device, "blocks", "run", device]
    if device.startswith("cuda"):
        expected.insert(1, "reset:cuda:1")
    assert events == expected
    assert measured["timing"]["raw_block_s"] == [0.2, 0.24]
    assert measured["timing"]["calls_per_block"] == 2
    assert measured["memory"]["peak_cuda_allocated_bytes"] == (
        123 if device.startswith("cuda") else None
    )


def test_output_conversion_and_checking_outside_timing(monkeypatch, timer_spy):
    events = timer_spy

    class Output:
        def __array__(self):
            events.append("convert")
            return np.array([2])

    output = Output()
    monkeypatch.setattr(
        _common, "perf_counter", lambda: events.append("clock") or len(events)
    )
    _common.measure_step(
        lambda: events.append("run") or output, device=torch.device("cpu"), warmup=0
    )
    assert events == ["clock", "run", "clock", "blocks", "run"]
    assert np.array_equal(np.asarray(output), [2])
    events.append("check")
    assert events[-2:] == ["convert", "check"]


@pytest.mark.parametrize("device", ["hpu", "meta", "cuda:0", "mps"])
def test_unavailable_device_never_runs(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    calls = []
    with pytest.raises(NotImplementedError):
        _common.measure_step(lambda: calls.append(1), device=torch.device(device))
    assert not calls


def test_real_cpu_numpy_and_caller_policy():
    x = np.ones((2, 3), dtype=np.float32)
    before = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    grad_enabled = []

    def run():
        grad_enabled.append(torch.is_grad_enabled())
        return x + x

    measured = _common.measure_step(
        run, device=torch.device("cpu"), warmup=1, min_run_time=0.01
    )
    assert all(grad_enabled)
    assert torch.equal(before, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert measured["timing"]["first_call_s"] > 0
    assert measured["timing"]["median_s"] > 0
    assert measured["timing"]["measurement_blocks"] == len(
        measured["timing"]["raw_block_s"]
    )
    assert measured["memory"]["peak_cuda_allocated_bytes"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("existing", [False, True])
def test_write_result_rejects_nonfinite_json(tmp_path, result, value, existing):
    path = tmp_path / "result.json"
    if existing:
        path.write_text("previous completed result")
    result["correctness"]["metrics"]["error"] = value
    with pytest.raises(ValueError):
        _common.write_result(result, path)
    assert (
        path.read_text() == "previous completed result"
        if existing
        else not path.exists()
    )
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


def test_result_roundtrip(tmp_path, result):
    original = copy.deepcopy(result)
    path = tmp_path / "result.json"
    _common.write_result(result, path)
    assert json.loads(path.read_text()) == original
    assert result == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("status", ["failed", "unsupported"])
def test_non_ok_roundtrip(tmp_path, result, status):
    result.update(status=status, timing=None)
    result["correctness"]["status"] = "not_checked"
    result["config"]["inputs"] = None
    result["environment"]["device_name"] = None
    result["notes"] = ["Could not allocate input: requested backend unavailable"]
    _common.write_result(result, tmp_path / "result.json")
    assert json.loads((tmp_path / "result.json").read_text())["status"] == status


@pytest.mark.parametrize("status", ["failed", "not_checked"])
def test_failed_case_has_no_speedup(tmp_path, result, status):
    result["status"] = "failed"
    result["correctness"]["status"] = status
    result["timing"]["speedup"] = 2.0
    with pytest.raises(ValueError):
        _common.write_result(result, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "schema_version", True),
        (None, "status", "success"),
        (None, "case_id", ""),
        (None, "extra", 1),
        (
            "config",
            "inputs",
            [
                {
                    "name": "x",
                    "shape": [2],
                    "dtype": "float32",
                    "strides": [4, 4],
                    "stride_unit": "bytes",
                }
            ],
        ),
        ("config", "threads", 0),
        ("environment", "environ", {"SECRET": "not allowed"}),
        ("source", "commit", "unknown"),
        ("source", "dirty", True),
        ("timing", "measurement_blocks", 3),
        ("timing", "calls_per_block", 0),
        ("timing", "median_s", -1),
        ("correctness", "status", "failed"),
        ("memory", "peak_cuda_allocated_bytes", 12),
    ],
)
def test_invalid_contract(tmp_path, result, section, key, value):
    target = result if section is None else result[section]
    target[key] = value
    with pytest.raises(ValueError):
        _common.write_result(result, tmp_path / "result.json")
    assert not list(tmp_path.iterdir())


def test_atomic_failure_cleans_only_own_temporary(monkeypatch, tmp_path, result):
    path = tmp_path / "result.json"
    path.write_text("completed")
    other = tmp_path / ".someone-else.tmp"
    other.write_text("unrelated")

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(_common.os, "replace", fail)
    with pytest.raises(OSError):
        _common.write_result(result, path)
    assert path.read_text() == "completed"
    assert other.read_text() == "unrelated"
    assert set(tmp_path.iterdir()) == {path, other}


@pytest.mark.parametrize("section", ["parameters", "metrics"])
def test_speedup_cannot_hide_in_extension_fields(tmp_path, result, section):
    result["status"] = "failed"
    result["correctness"]["status"] = "failed"
    result["notes"] = ["Numerical check failed"]
    target = (
        result["config"][section]
        if section == "parameters"
        else result["correctness"][section]
    )
    target["nested"] = {"speedup": 2.0}
    with pytest.raises(ValueError, match="speedup"):
        _common.write_result(result, tmp_path / "result.json")
    assert not list(tmp_path.iterdir())


def test_non_ok_timing_without_speedup_is_diagnostic(tmp_path, result):
    result["status"] = "failed"
    result["correctness"]["status"] = "failed"
    result["notes"] = ["Numerical check failed after timing"]
    _common.write_result(result, tmp_path / "result.json")
    assert (
        json.loads((tmp_path / "result.json").read_text())["timing"]["median_s"] == 0.1
    )


def test_cuda_ok_requires_peak(tmp_path, result):
    result["environment"]["device"] = "cuda:0"
    with pytest.raises(ValueError, match="peak"):
        _common.write_result(result, tmp_path / "result.json")


def test_dirty_source_with_hash_roundtrip(tmp_path, result):
    result["source"].update(dirty=True, dirty_source_hashes={"experiment.py": "a" * 64})
    _common.write_result(result, tmp_path / "result.json")
    assert json.loads((tmp_path / "result.json").read_text())["source"]["dirty"] is True


def test_json_rejects_nonstring_key(tmp_path, result):
    result["config"]["parameters"][1] = "lossy"
    with pytest.raises(ValueError, match="keys"):
        _common.write_result(result, tmp_path / "result.json")


def test_invalid_cuda_index_never_runs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    calls = []
    with pytest.raises(NotImplementedError):
        _common.measure_step(lambda: calls.append(1), device=torch.device("cuda:1"))
    assert calls == []


def test_callable_failure_propagates():
    def run():
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        _common.measure_step(run, device=torch.device("cpu"))


def test_failed_case_has_no_speedup_in_valid_failed_record(tmp_path, result):
    result["status"] = "failed"
    result["correctness"]["status"] = "failed"
    result["notes"] = ["Mismatch"]
    result["timing"]["speedup"] = 2.0
    with pytest.raises(ValueError, match="speedup"):
        _common.write_result(result, tmp_path / "result.json")


def test_device_unknown_metadata_roundtrip(tmp_path, result):
    result["status"] = "unsupported"
    result["environment"].update(device="hpu", device_name=None)
    result["environment"]["dependencies"]["habana_frameworks"] = None
    result["correctness"].update(status="not_checked", method="backend unavailable")
    result["config"].update(inputs=None, seed=None)
    result["timing"] = None
    result["notes"] = ["HPU is not qualified; hardware and inputs not discovered"]
    _common.write_result(result, tmp_path / "result.json")
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "unsupported"


@pytest.mark.parametrize("field,value", [("median_s", 0.000001), ("iqr_s", 999.0)])
@pytest.mark.parametrize("existing", [False, True])
def test_inconsistent_summary_never_published(tmp_path, result, field, value, existing):
    path = tmp_path / "result.json"
    if existing:
        path.write_text("previous completed result")
    result["timing"].update(
        raw_block_s=[1.0, 1.0], calls_per_block=1, median_s=1.0, iqr_s=0.0
    )
    result["timing"][field] = value
    with pytest.raises(ValueError, match=field):
        _common.write_result(result, path)
    if existing:
        assert path.read_text() == "previous completed result"
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


@pytest.mark.parametrize(
    "raw,calls,median,iqr",
    [
        ([8.0], 4, 2.0, 0.0),
        ([12.0, 4.0, 8.0], 4, 2.0, 1.0),
        ([16.0, 4.0, 12.0, 8.0], 4, 2.5, 1.5),
        ([1.0, 2.0, 3.0, 4.0], 1, 2.5, 1.5),
    ],
)
def test_summary_uses_normalized_linear_quantiles(
    tmp_path, result, raw, calls, median, iqr
):
    result["timing"].update(
        raw_block_s=raw,
        calls_per_block=calls,
        measurement_blocks=len(raw),
        median_s=median,
        iqr_s=iqr,
    )
    path = tmp_path / "result.json"
    _common.write_result(result, path)
    assert json.loads(path.read_text())["timing"] == result["timing"]
    # A raw-block rather than per-call summary must not be accepted.
    result["timing"]["median_s"] = median * (calls if calls > 1 else 2)
    with pytest.raises(ValueError, match="median_s"):
        _common.write_result(result, path)
    assert json.loads(path.read_text())["timing"]["median_s"] == median


@pytest.mark.parametrize("field", ["median_s", "iqr_s"])
def test_summary_tolerance(tmp_path, result, field):
    result["timing"].update(
        raw_block_s=[1.0, 2.0, 3.0],
        calls_per_block=1,
        measurement_blocks=3,
        median_s=2.0,
        iqr_s=1.0,
    )
    expected = result["timing"][field]
    result["timing"][field] = expected * (1 + 5e-10)
    path = tmp_path / "result.json"
    _common.write_result(result, path)
    completed = path.read_bytes()
    result["timing"][field] = expected * (1 + 2e-9)
    with pytest.raises(ValueError, match=field):
        _common.write_result(result, path)
    assert path.read_bytes() == completed


def test_genuine_timer_output_roundtrip(tmp_path, result):
    x = np.ones((2, 3), dtype=np.float32)
    result.update(
        _common.measure_step(
            lambda: x + x, device=torch.device("cpu"), warmup=1, min_run_time=0.01
        )
    )
    path = tmp_path / "result.json"
    _common.write_result(result, path)
    assert json.loads(path.read_text()) == result

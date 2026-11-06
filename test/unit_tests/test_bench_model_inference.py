"""Correctness smoke tests for the standalone model-inference benchmark."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from benchmarks import bench_model_inference as bench
from benchmarks.bench_model_inference import benchmark, parser


@pytest.fixture(autouse=True)
def restore_torch_settings():
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("model", ["EEGNet", "ShallowFBCSPNet", "EEGConformer"])
def test_model_inference_benchmark_cpu(model):
    args = parser().parse_args(
        [
            "--model",
            model,
            "--batch-size",
            "2",
            "--warmup",
            "1",
            "--min-run-time",
            "0.01",
        ]
    )
    result = benchmark(args)
    assert result["config"]["model"] == model
    assert result["environment"]["device"] == "cpu"
    assert result["median_batch_s"] > 0
    assert result["first_candidate_after_reference_s"] > 0
    assert math.isfinite(result["windows_per_s"])
    assert result["windows_per_s"] == pytest.approx(2 / result["median_batch_s"])
    assert result["measurement_blocks"] >= 1
    assert result["calls_per_block"] >= 1
    assert result["peak_cuda_allocated_bytes"] is None
    assert result["numerical_check"]["max_abs_error"] < 1e-4


@pytest.mark.parametrize(
    "option", ["--batch-size", "--n-chans", "--n-times", "--threads", "--warmup"]
)
def test_positive_integer_options(option):
    with pytest.raises(SystemExit):
        parser().parse_args([option, "0"])


@pytest.mark.parametrize(
    "options, message",
    [
        (["--precision", "float16"], "only on CUDA"),
        (["--device", "mps", "--precision", "bfloat16"], "MPS"),
        (["--min-run-time", "0"], "must be positive"),
        (["--min-run-time=nan"], "positive and finite"),
        (["--min-run-time=-inf"], "positive and finite"),
        (["--device", "meta"], "Supported devices"),
    ],
)
def test_unsupported_settings(options, message):
    with pytest.raises(ValueError, match=message):
        benchmark(parser().parse_args(options))


@pytest.fixture
def tiny_benchmark(monkeypatch):
    events = []

    class TinyModel(torch.nn.Module):
        def __init__(self, n_chans, n_outputs, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(n_chans, n_outputs)

        def forward(self, x):
            events.append(("forward", x.dtype, torch.is_autocast_enabled("cpu")))
            return self.linear(x.mean(-1))

    monkeypatch.setattr(bench.models, "EEGNet", TinyModel)
    monkeypatch.setattr(
        bench,
        "Timer",
        lambda **kwargs: SimpleNamespace(
            blocked_autorange=lambda **kw: SimpleNamespace(
                median=0.01, iqr=0.001, raw_times=[0.01, 0.011], number_per_run=1
            )
        ),
    )
    return events


def test_reference_and_first_candidate_timing_order(monkeypatch, tiny_benchmark):
    events = tiny_benchmark
    original_float = torch.Tensor.float

    def cast(tensor, *args, **kwargs):
        events.append("cast")
        return original_float(tensor, *args, **kwargs)

    def clock():
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(torch.Tensor, "float", cast)
    monkeypatch.setattr(bench, "perf_counter", clock)
    benchmark(parser().parse_args(["--warmup", "1"]))
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names[:6] == ["forward", "cast", "clock", "forward", "clock", "cast"]


def test_explicit_fp32_with_outer_autocast_and_float64_default(tiny_benchmark):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = benchmark(parser().parse_args(["--warmup", "1"]))
    finally:
        torch.set_default_dtype(old_dtype)
    assert all(dtype == torch.float32 and not amp for _, dtype, amp in tiny_benchmark)
    json.dumps(result, allow_nan=False)


def test_compilation_and_amp_context(monkeypatch, tiny_benchmark):
    calls = []
    monkeypatch.setattr(torch, "compile", lambda model, **kw: calls.append(kw) or model)
    benchmark(
        parser().parse_args(["--compile", "--precision", "bfloat16", "--warmup", "1"])
    )
    assert calls == [{"mode": "default"}]
    assert tiny_benchmark[0][2] is False  # FP32 reference
    assert all(event[2] for event in tiny_benchmark[1:])


def test_numerical_mismatch_fails(monkeypatch, tiny_benchmark):
    monkeypatch.setattr(torch, "compile", lambda model, **kw: lambda x: model(x) + 100)
    with pytest.raises(AssertionError):
        benchmark(parser().parse_args(["--compile"]))


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_synchronization_dispatch(monkeypatch, device):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append(str(device))
    )
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    bench.synchronize(torch.device(device))
    assert calls == ([] if device == "cpu" else [device])


def test_cpu_benchmark_preserves_rng(tiny_benchmark):
    before = torch.random.get_rng_state()
    mps_before = (
        torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    )
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    benchmark(parser().parse_args(["--warmup", "1"]))
    assert torch.equal(before, torch.random.get_rng_state())
    if mps_before is not None:
        assert torch.equal(mps_before, torch.mps.get_rng_state())
    for before, after in zip(
        cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []
    ):
        assert torch.equal(before, after)

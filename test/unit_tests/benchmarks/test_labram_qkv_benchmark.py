"""Pure bookkeeping contracts for the bounded Labram benchmark."""
import importlib.util
from pathlib import Path


def _benchmark():
    path = Path(__file__).resolve().parents[3] / "benchmarks/acceleration/bench_labram_qkv.py"
    spec = importlib.util.spec_from_file_location("labram_qkv_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labram_benchmark_manifest():
    module = _benchmark()
    assert module.FIXTURES["small"]["n_chans"] == 4
    assert module.FIXTURES["representative"]["n_chans"] == 128
    assert module._summary([1., 2., 3., 4., 5.]) == {
        "raw_s": [1., 2., 3., 4., 5.], "median_s": 3., "iqr_s": 2.,
        "max_min_ratio": 5.}

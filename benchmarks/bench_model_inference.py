"""Measure fixed-shape model inference, not data loading or training.

Run from the repository root, for example::

    python -m benchmarks.bench_model_inference --model EEGNet --device cpu
    python -m benchmarks.bench_model_inference --model EEGConformer \
        --device cuda --compile --precision bfloat16

Every registered model, one JSON line per model (eager and, with
``--compile``, compiled medians in milliseconds), each built with its test
fixture configuration::

    python -m benchmarks.bench_model_inference --models all --device cpu \
        --dtype float32 --batch 64 --warmup 5 --repeats 20 --out bench.jsonl

Uses PyTorch's benchmark Timer, with explicit accelerator synchronization also
covering MPS. Reports first candidate execution after an eager FP32 reference
separately from warmed-up block timings. This is NOT cold-device startup or
necessarily a cold compiler-cache measurement.
Inputs stay on device; weights are randomly initialized; no data is downloaded.
Run each configuration in a fresh process on otherwise idle hardware. Compare
identical shapes, threads, precision, software versions and backend settings.
"""

import argparse
import gc
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.benchmark import Timer

import braindecode
from braindecode import models


def positive_int(value):
    """Parse a positive integer for shapes and warmup counts."""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def synchronize(device):
    """Wait for pending work on the explicitly selected accelerator."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "hpu":
        torch.hpu.synchronize()


def check_device(device):
    """Refuse unsupported devices; import the Gaudi bridge only for ``hpu``."""
    if device.type not in {"cpu", "cuda", "mps", "hpu"}:
        raise ValueError("Supported devices are cpu, cuda, mps and hpu")
    if device.type == "hpu":
        import habana_frameworks.torch  # noqa: F401


def benchmark(args):
    """Return JSON-compatible timing, numerical-check and environment data."""
    device = torch.device(args.device)
    check_device(device)
    if args.precision == "float16" and device.type != "cuda":
        raise ValueError("float16 is supported by this benchmark only on CUDA")
    if args.precision == "bfloat16" and device.type == "mps":
        raise ValueError("bfloat16 on MPS is not covered by this benchmark")
    if not math.isfinite(args.min_run_time) or args.min_run_time <= 0:
        raise ValueError("min_run_time must be positive and finite")

    torch.set_num_threads(args.threads)
    # Initialize on CPU without seeding or consuming accelerator RNG streams.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(args.seed)
        model = (
            getattr(models, args.model)(
                n_chans=args.n_chans, n_outputs=4, n_times=args.n_times, sfreq=250
            )
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        x = torch.randn(
            args.batch_size, args.n_chans, args.n_times, dtype=torch.float32
        ).to(device)
    dtype = getattr(torch, args.precision)

    def precision_context():
        return torch.autocast(
            device_type=device.type,
            dtype=dtype if dtype != torch.float32 else None,
            enabled=dtype != torch.float32,
        )

    # The same model/input form the reference, without consuming a second RNG
    # stream. No training, dropout, optimizer or autograd work is timed.
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        reference = model(x).float()
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    def forward():
        with torch.inference_mode(), precision_context():
            return model(x)

    def synchronized_forward():
        result = forward()
        synchronize(device)
        return result

    synchronize(device)
    start = perf_counter()
    actual = synchronized_forward()
    first_call_s = perf_counter() - start
    actual = actual.float()
    # Fail loudly: a numerically invalid execution must not produce a speedup
    # claim. These tolerances check this synthetic input, not task accuracy.
    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (1e-2, 5e-2)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    max_abs_error = (actual - reference).abs().max().item()
    for _ in range(args.warmup):
        synchronized_forward()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measurement = Timer(
        stmt="run()",
        globals={"run": synchronized_forward},
        num_threads=args.threads,
    ).blocked_autorange(min_run_time=args.min_run_time)

    return {
        "scope": "model-only fixed-shape inference; inputs already on device",
        "config": vars(args),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "braindecode": braindecode.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.machine()
            ),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "threads": torch.get_num_threads(),
        },
        "numerical_check": {"max_abs_error": max_abs_error, "atol": atol, "rtol": rtol},
        "first_candidate_after_reference_s": first_call_s,
        "median_batch_s": measurement.median,
        "iqr_batch_s": measurement.iqr,
        "windows_per_s": args.batch_size / measurement.median,
        "measurement_blocks": len(measurement.raw_times),
        "calls_per_block": measurement.number_per_run,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def _median_ms(run, device, warmup, repeats):
    """Median wall time of ``run()`` in milliseconds, after ``warmup`` calls."""
    for _ in range(warmup):
        run()
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        run()
        synchronize(device)
        times.append((perf_counter() - start) * 1e3)
    return statistics.median(times)


def sweep(args):
    """Yield one timing record per registered model (test fixture shapes)."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from braindecode.models.util import models_mandatory_parameters
    from test.unit_tests.models._fixtures import _build_case, _materialize

    device = torch.device(args.device)
    check_device(device)
    dtype = getattr(torch, args.precision)
    torch.set_num_threads(args.threads)
    cases = {case[0]: case for case in models_mandatory_parameters}
    names = list(cases) if args.models == "all" else args.models.split(",")
    for name in names:
        torch.manual_seed(args.seed)
        model, x = _build_case(*cases[name])
        _materialize(model, x)
        x = torch.randn(args.batch_size, *x.shape[1:]).to(device)
        model = model.to(device)
        record = {"model": name, "device": device.type, "dtype": args.precision}
        record.update(batch=args.batch_size, eager_ms=None, compile_ms=None)

        def timed(module, x):
            def run():
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device.type, dtype=dtype, enabled=dtype != torch.float32
                    ),
                ):
                    return module(x)

            return _median_ms(run, device, args.warmup, args.repeats)

        # A sweep reports a model that fails (for example in bfloat16)
        # instead of stopping at it.
        try:
            record["eager_ms"] = timed(model, x)
            if args.compile:
                compiled = torch.compile(model, mode=args.compile_mode)
                record["compile_ms"] = timed(compiled, x)
        except (RuntimeError, ValueError, TypeError, AssertionError) as error:
            record["error"] = f"{type(error).__name__}: {error}".splitlines()[0]
        yield record
        del model, x
        gc.collect()


def parser():
    """Build the standalone benchmark CLI."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model",
        choices=["EEGNet", "ShallowFBCSPNet", "EEGConformer"],
        default="EEGNet",
    )
    result.add_argument(
        "--models",
        default=None,
        help="'all' or comma-separated registered models: sweep mode, JSON lines",
    )
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--precision",
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    result.add_argument("--compile", action="store_true")
    result.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    result.add_argument("--batch-size", "--batch", type=positive_int, default=16)
    result.add_argument("--n-chans", type=positive_int, default=22)
    result.add_argument("--n-times", type=positive_int, default=1000)
    result.add_argument("--threads", type=positive_int, default=1)
    result.add_argument("--warmup", type=positive_int, default=5)
    result.add_argument("--repeats", type=positive_int, default=20)
    result.add_argument("--out", type=Path, default=None, help="JSON lines file")
    result.add_argument("--min-run-time", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=0)
    return result


def main(argv=None):
    """Run one model (pretty JSON) or a sweep (JSON lines to stdout or --out)."""
    args = parser().parse_args(argv)
    if args.models is None:
        print(json.dumps(benchmark(args), indent=2, allow_nan=False))
        return
    out = args.out.open("w") if args.out is not None else sys.stdout
    for record in sweep(args):
        out.write(json.dumps(record) + "\n")
        out.flush()
    if args.out is not None:
        out.close()


if __name__ == "__main__":
    main()

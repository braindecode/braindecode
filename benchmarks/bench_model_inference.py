"""Measure fixed-shape model inference, not data loading or training.

Run from the repository root, for example::

    python -m benchmarks.bench_model_inference --model EEGNet --device cpu
    python -m benchmarks.bench_model_inference --model EEGConformer \
        --device cuda --compile --precision bfloat16

Uses PyTorch's benchmark Timer, with explicit accelerator synchronization also
covering MPS. Reports first candidate execution after an eager FP32 reference
separately from warmed-up block timings. This is NOT cold-device startup or
necessarily a cold compiler-cache measurement.
Inputs stay on device; weights are randomly initialized; no data is downloaded.
Run each configuration in a fresh process on otherwise idle hardware. Compare
identical shapes, threads, precision, software versions and backend settings.
"""

import argparse
import json
import math
import platform
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


def benchmark(args):
    """Return JSON-compatible timing, numerical-check and environment data."""
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices are cpu, cuda and mps")
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


def parser():
    """Build the standalone benchmark CLI."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model",
        choices=["EEGNet", "ShallowFBCSPNet", "EEGConformer"],
        default="EEGNet",
    )
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--precision", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    result.add_argument("--compile", action="store_true")
    result.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    result.add_argument("--batch-size", type=positive_int, default=16)
    result.add_argument("--n-chans", type=positive_int, default=22)
    result.add_argument("--n-times", type=positive_int, default=1000)
    result.add_argument("--threads", type=positive_int, default=1)
    result.add_argument("--warmup", type=positive_int, default=5)
    result.add_argument("--min-run-time", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=0)
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(parser().parse_args()), indent=2, allow_nan=False))

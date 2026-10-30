"""CPU log-statistics qualification and separate actual-model attribution.

``run_case`` exposes correction/clamp directly for its historical Python API.
Use ``schema_record`` or ``save_result`` for strict schema-1 serialization.
Compilation/setup, correctness and instrumented profiling are outside headline
steady-state timing. This runner makes no CUDA or MPS qualification claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import platform
import signal
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch

from benchmarks.acceleration._common import measure_step, write_result
from braindecode.modules import LogPowerLayer, LogVarLayer, StatLayer

ROOT = Path(__file__).resolve().parents[2]
BOUNDS = {"logvar": [1e-6, 1e6], "logpower": [1e-4, 1e4]}
LAYERS = {"logvar": LogVarLayer, "logpower": LogPowerLayer}
DTYPES = {name: getattr(torch, name) for name in
          ("float16", "bfloat16", "float32", "float64")}


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _source():
    paths = set(_git("diff", "--name-only", "HEAD").splitlines())
    paths.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    hashes = {}
    for name in sorted(paths):
        path = ROOT / name
        content = path.read_bytes() if path.is_file() else b"<deleted>"
        hashes[name] = hashlib.sha256(content).hexdigest()
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(hashes),
            "dirty_source_hashes": hashes}


def _input(value):
    return {"name": "x", "shape": list(value.shape), "dtype": str(value.dtype),
            "strides": list(value.stride()), "stride_unit": "elements"}


def _record(case_id, implementation, device, dtype, mode, parameters):
    return {
        "schema_version": 1, "case_id": case_id, "implementation": implementation,
        "status": "unsupported",
        "config": {
            "inputs": None, "seed": 231, "threads": torch.get_num_threads(),
            "workers": 0, "cache_policy": "synthetic fixed inputs; warmup; OS cache uncontrolled",
            "autograd": "no_grad" if mode == "forward" else "forward plus fresh gradients each call",
            "precision": dtype, "parameters": {"mode": mode, **parameters},
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {"torch": str(torch.__version__)},
            "device": str(device), "device_name": platform.node() + "/" + platform.machine(),
        },
        "source": _source(),
        "correctness": {"status": "not_checked", "method": "not executed", "metrics": {}},
        "timing": None, "memory": {"peak_cuda_allocated_bytes": None}, "notes": [],
    }


def _reference(x, kind, dim, keepdim):
    reduced = (torch.var(x, dim=dim, correction=1, keepdim=keepdim)
               if kind == "logvar" else torch.mean(x**2, dim=dim, keepdim=keepdim))
    dtype = reduced.dtype
    promoted = reduced.to(torch.promote_types(dtype, torch.float32))
    return torch.log(torch.clamp(promoted, *BOUNDS[kind])).to(dtype)


def _tensor(shape, dtype, layout):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(231), dtype=dtype)
    if layout == "transposed":
        if x.ndim < 2:
            raise ValueError("transposed layout requires at least two dimensions")
        x = x.transpose(0, -1).contiguous().transpose(0, -1)
    elif layout == "sliced":
        x = torch.stack((x, x), dim=-1)[..., 0]
    elif layout != "contiguous":
        raise ValueError(f"Unknown layout: {layout}")
    return x


def _check_mode(mode):
    if mode not in ("forward", "forward_backward"):
        raise ValueError(f"Unknown mode: {mode}")


def _step(function, x, mode, model=None):
    def run():
        if mode == "forward":
            with torch.no_grad():
                return function(x)
        x.grad = None
        if model is not None:
            model.zero_grad(set_to_none=True)
        y = function(x)
        y.sum().backward()
        return y
    return run


def _model_attribution(model, run):
    """Instrument only this extra call, never the headline timing region.

    The numerator counts forward operator self time inside actual StatLayer
    ranges, not similarly named operators elsewhere or unscoped backward work.
    The denominator covers the whole instrumented forward(/backward) step.
    """
    handles, modules, ranges = [], {}, {}

    def before(name):
        def hook(module, inputs):
            modules[name] = {"type": type(module).__name__, "input": _input(inputs[0]),
                             "dim": module.dim, "keepdim": module.keepdim,
                             "clamp": list(module.clamp_range) if module.clamp_range else None}
            scope = torch.profiler.record_function("statistics::" + name)
            scope.__enter__()
            ranges[name] = scope
        return hook

    def after(name):
        def hook(module, inputs, output):
            if output is not None:
                modules[name]["output_shape"] = list(output.shape)
                modules[name]["output_dtype"] = str(output.dtype)
            scope = ranges.pop(name, None)
            if scope is not None:
                scope.__exit__(None, None, None)
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, StatLayer):
                handles.append(module.register_forward_pre_hook(before(name)))
                handles.append(module.register_forward_hook(after(name), always_call=True))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            run()
    finally:
        for handle in handles:
            handle.remove()
        for scope in ranges.values():
            scope.__exit__(None, None, None)
    events = prof.events()
    total = sum(max(0.0, event.self_cpu_time_total) for event in events)
    reduction = 0.0
    for event in events:
        parent = event.cpu_parent
        while parent is not None:
            if parent.name.startswith("statistics::"):
                reduction += max(0.0, event.self_cpu_time_total)
                break
            parent = parent.cpu_parent
    operators = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    return {
        "profiler_separate_from_timing": True,
        "total_self_cpu_time_us": total,
        "reduction_chain_self_cpu_time_us": reduction,
        "reduction_chain_self_cpu_fraction": reduction / total if total else 0.0,
        "scope": "forward operator self time within actual StatLayer hooks / full profiled step; backward reduction work is not assigned",
        "statistic_modules": modules,
        "operators": [{"name": e.key, "calls": e.count,
                       "self_cpu_time_us": e.self_cpu_time_total,
                       "cpu_time_us": e.cpu_time_total} for e in operators[:30]],
    }


def profile_model(model, *, device="cpu", mode="forward", min_run_time=0.1,
                  batch=2, n_chans=22, n_times=1000):
    """Profile the actual default FBCNet/IFNet, in eval mode with optional grad.

    Filters, convolutions, statistical pooling and the classifier all execute.
    Construction is excluded; no download, pretrained weights or altered filter
    algorithm is used. Backward includes input and all trainable parameters.
    """
    from braindecode.models import FBCNet, IFNet

    _check_mode(mode)
    if model not in ("FBCNet", "IFNet"):
        raise ValueError(f"Unknown model: {model}")
    parameters = {"model": model, "training": False, "batch": batch,
                  "n_chans": n_chans, "n_times": n_times, "sfreq": 250,
                  "n_outputs": 4, "constructor": "remaining model defaults"}
    result = _record(f"statistics/model/{model}/{mode}", "reference", device,
                     "float32", mode, parameters)
    if str(device) != "cpu":
        result["notes"] = ["Only CPU is qualified by this runner; no GPU execution."]
        return result
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(231)
        network = {"FBCNet": FBCNet, "IFNet": IFNet}[model](
            n_chans=n_chans, n_times=n_times, n_outputs=4, sfreq=250).eval()
    x = _tensor((batch, n_chans, n_times), torch.float32, "contiguous")
    x.requires_grad_(mode == "forward_backward")
    result["config"]["inputs"] = [_input(x)]
    clone = copy.deepcopy(network)
    y = x.detach().clone().requires_grad_(x.requires_grad)
    run = _step(network, x, mode, network)
    expected = _step(clone, y, mode, clone)()
    actual = run()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all(), "Nonfinite model output"
    gradient_count = 0
    if x.requires_grad:
        for left, right in [(x, y), *zip(network.parameters(), clone.parameters())]:
            if left.grad is None and right.grad is None:
                continue
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            assert torch.isfinite(left.grad).all(), "Nonfinite model gradient"
            gradient_count += 1
    del clone, expected, actual, y
    result["correctness"] = {
        "status": "passed", "method": "identical eval-model clone output and gradient repeatability; finite values (not an independent model oracle)",
        "metrics": {"gradient_checked": bool(gradient_count), "gradient_tensors": gradient_count},
    }
    result.update(measure_step(run, device=torch.device("cpu"), min_run_time=min_run_time))
    result["config"]["parameters"]["attribution"] = _model_attribution(network, run)
    result["status"] = "ok"
    result["notes"] = [
        "Headline is uninstrumented full-model eval execution; forward_backward includes input/parameter gradients and gradient reset, not optimizer updates.",
        "Attribution is a separate instrumented call and is not headline latency or a predicted speedup; the reduction numerator excludes backward work.",
        "Synthetic inputs only. CPU results do not establish CUDA performance or block CUDA exploration.",
    ]
    return result


def schema_record(result):
    """Deep-copy the Python API result into strict shared schema 1.

    No shared validator relaxation: correction and clamp belong in parameters
    on disk. The original record and its nested policy arrays remain untouched.
    """
    adapted = copy.deepcopy(result)
    config = adapted["config"]
    for name in ("correction", "clamp"):
        if name in config:
            config["parameters"][name] = config.pop(name)
    return adapted


def save_result(result, path):
    """Adapt, validate and atomically write to an existing parent directory."""
    write_result(schema_record(result), Path(path))


@contextmanager
def _compile_deadline(seconds):
    if seconds <= 0:
        raise ValueError("compile_timeout_s must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise NotImplementedError("Bounded CPU compilation requires the main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise NotImplementedError("Cannot replace an active process alarm for compilation")
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError(f"Compilation exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run_case(kind, shape, dim, keepdim, *, device, dtype, implementation,
             mode="forward", layout="contiguous", min_run_time=0.1,
             compile_timeout_s=90):
    """Qualify and time an exact native/compiled reduction on synthetic CPU data.

    Native/reference means the current production StatLayer, checked against an
    explicit independent expression. Forward uses no_grad. Forward_backward
    includes sum-loss construction, backward and input-gradient reset. Setup,
    compilation (including backward compilation) and checks are not timed.
    CPU float16 is refused before tensor/backend execution.
    """
    _check_mode(mode)
    if kind not in LAYERS:
        raise ValueError(f"Unknown statistic: {kind}")
    if implementation not in ("reference", "compiled"):
        raise ValueError(f"Unknown implementation: {implementation}")
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype}")
    if not shape or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("shape must contain nonnegative integer sizes")
    if not -len(shape) <= dim < len(shape):
        raise ValueError("dim is outside shape")
    parameters = {"kind": kind, "dim": dim, "keepdim": keepdim,
                  "layout": layout, "compile_setup_s": 0.0,
                  "compile_timeout_s": compile_timeout_s,
                  "compile_backend": "inductor" if implementation == "compiled" else None,
                  "fullgraph": implementation == "compiled"}
    result = _record(f"statistics/{kind}/{mode}/{layout}", implementation,
                     device, dtype, mode, parameters)
    result["config"].update(correction=1 if kind == "logvar" else None,
                            clamp=list(BOUNDS[kind]))
    if str(device) != "cpu" or dtype == "float16":
        result["notes"] = [
            "CPU float16 deliberately unsupported; known unsafe backend stress is not executed."
            if str(device) == "cpu" else "Only CPU is qualified by this runner; no GPU execution."
        ]
        return result
    x = _tensor(shape, DTYPES[dtype], layout)
    x.requires_grad_(mode == "forward_backward")
    result["config"]["inputs"] = [_input(x)]
    layer = LAYERS[kind](dim, keepdim)
    function = layer
    if implementation == "compiled":
        start = perf_counter()
        try:
            with _compile_deadline(compile_timeout_s):
                function = torch.compile(layer, backend="inductor", fullgraph=True)
                _step(function, x, mode)()
        except (TimeoutError, NotImplementedError) as exc:
            result["notes"] = [f"Compiled case unsupported: {type(exc).__name__}: {exc}"]
            return result
        except RuntimeError as exc:
            result["status"] = "failed"
            result["notes"] = [f"Compilation failed: {type(exc).__name__}: {exc}"]
            return result
        finally:
            result["config"]["parameters"]["compile_setup_s"] = perf_counter() - start
    y = x.detach().clone().requires_grad_(x.requires_grad)
    run = _step(function, x, mode)
    try:
        actual = run()
        expected = _step(lambda t: _reference(t, kind, dim, keepdim), y, mode)()
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, equal_nan=True)
        if x.requires_grad:
            torch.testing.assert_close(x.grad, y.grad, equal_nan=True)
        result["correctness"] = {
            "status": "passed",
            "method": "explicit var(correction=1) or square/mean; promote clamp/log then cast back; output dtype/shape and autograd",
            "metrics": {"gradient_checked": x.requires_grad},
        }
    except (AssertionError, RuntimeError) as exc:
        result["status"] = "failed"
        result["correctness"] = {"status": "failed", "method": "explicit reference comparison",
                                 "metrics": {"gradient_checked": False}}
        result["notes"] = [f"Correctness failure: {type(exc).__name__}: {exc}"]
        return result
    del actual, expected, y
    result.update(measure_step(run, device=torch.device("cpu"), min_run_time=min_run_time))
    result["status"] = "ok"
    result["notes"] = [
        "No dtype/autocast policy change; native reduction precedes float32-minimum clamp/log and cast-back.",
        "Compilation, correctness and setup excluded from headline timing; forward_backward includes gradient reset and sum-loss backward.",
        "CPU only; no CUDA performance inference. CPU timing does not gate CUDA prototype exploration.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=list(LAYERS), default="logvar")
    parser.add_argument("--model", choices=["FBCNet", "IFNet"])
    parser.add_argument("--shape", type=int, nargs="+", default=[16, 288, 4, 250])
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--keepdim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    parser.add_argument("--implementation", choices=["reference", "compiled"], default="reference")
    parser.add_argument("--layout", choices=["contiguous", "transposed", "sliced"], default="contiguous")
    parser.add_argument("--mode", choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument("--compile-timeout-s", type=float, default=90)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if args.require_clean and _git("status", "--porcelain"):
        parser.error("Retained paired evidence requires a clean committed source tree")
    torch.set_num_threads(args.threads)
    if args.model:
        if args.implementation != "reference":
            parser.error("Model attribution measures actual native models, not a compiled surrogate")
        result = profile_model(args.model, mode=args.mode, min_run_time=args.min_run_time)
    else:
        result = run_case(args.kind, tuple(args.shape), args.dim, args.keepdim,
                          device="cpu", dtype=args.dtype, implementation=args.implementation,
                          mode=args.mode, layout=args.layout, min_run_time=args.min_run_time,
                          compile_timeout_s=args.compile_timeout_s)
    save_result(result, args.output)
    print(f"{result['case_id']}: {result['status']} -> {args.output}")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

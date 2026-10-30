"""Compiled-training and exported-execution qualification against an eager oracle.

Three execution modes, all FP32 on CPU:

- ``eager``: one forward/backward/optimizer-update step (cross-entropy against
  fixed targets, SGD) of the configured model; this is the oracle.
- ``compiled``: the same training step with the module wrapped by
  ``torch.compile`` (inductor). The module forward and its AOT backward are
  compiled; ``zero_grad``, the loss and ``optimizer.step`` run eagerly. The
  timed model is the *same configured model* as eager mode (equal work).
  Correctness is a set of structured checks in ``correctness.metrics.checks``:
  dropout-off parity (``dropout_override=0.0``), a partial-batch step with
  recompile detection, dropout-on parity with
  ``torch._inductor.config.fallback_random=True`` (eager RNG kernels), and
  finiteness of the timed configured model's first step. Scoped limitation:
  dropout masks of inductor's default RNG, which the timed model uses, differ
  from eager's and are not compared.
- ``exported``: forward-only execution of a ``torch.export`` artifact after an
  actual save/reload round trip (not a type check), checked against an
  eval-mode eager forward with ``assert_close(rtol=1e-4, atol=1e-5)``; with
  ``dynamic_batch`` one artifact is executed at several batch sizes.

Status semantics: ``ok`` means every check passed; ``failed`` means a
compile/execution error, a compiler timeout or a numeric mismatch;
``unsupported`` means the runner cannot qualify this request (non-CPU device,
invalid model configuration, export not possible, no bounded-compile support).
Setup, correctness and diagnostics are outside headline timing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import math
import os
import pickle
import platform
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.acceleration._common import measure_step, write_result
from braindecode.models import EEGNet, ShallowFBCSPNet
from braindecode.models.luna import _RotarySelfAttentionBlock

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no getrusage
    resource = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
MODES = ("eager", "compiled", "exported")
SEED = 231
PARITY = {"rtol": 1e-4, "atol": 1e-5}
# torch.export surfaces many distinct exception types across versions/graphs;
# a blind `except Exception` is deliberately avoided so anything else
# propagates as a hard error instead of being recorded as "unsupported".
_EXPORT_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    NotImplementedError,
    AssertionError,
)
# Errors while compiling, executing, saving or loading: recorded as "failed".
# TimeoutError (the compile deadline) is an OSError subclass.
_EXECUTION_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    NotImplementedError,
    AssertionError,
    OSError,
    KeyError,
    IndexError,
    pickle.PickleError,
)
_RSS_SCOPE = (
    "process-level resident-set high-water mark (getrusage ru_maxrss), "
    "including imports and all earlier work in this process; not a per-step "
    "tensor allocation peak. memory.peak_cuda_allocated_bytes is CUDA-only "
    "and stays null on CPU."
)


class CompileTimeout(TimeoutError):
    """Raised by the compile deadline; may arrive wrapped by dynamo/inductor."""


class LUNARotaryBlockModel(nn.Module):
    """Small standalone classifier built from LUNA's internal rotary-attention block.

    Not the full LUNA foundation model: a single rotary self-attention block
    plus a linear head, used only to qualify execution modes on one LUNA
    primitive at low cost. ``n_times`` becomes the block's sequence length;
    ``n_chans`` becomes its embedding dimension and must divide ``num_heads``.
    """

    def __init__(self, n_chans, n_outputs, n_times=None, num_heads=2, dropout=0.1):
        super().__init__()
        del n_times  # sequence length is runtime (input-shape) determined
        if n_chans % num_heads != 0:
            raise ValueError(
                f"n_chans ({n_chans}) must be divisible by num_heads ({num_heads})"
            )
        self.block = _RotarySelfAttentionBlock(
            dim=n_chans,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.head = nn.Linear(n_chans, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_chans, n_times) -> (batch, n_times, n_chans)
        sequence = x.transpose(1, 2)
        features = self.block(sequence)
        pooled = features.mean(dim=1)
        return self.head(pooled)


def _eegnet(n_chans, n_times, n_outputs, **kwargs):
    return EEGNet(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, **kwargs)


def _shallow(n_chans, n_times, n_outputs, **kwargs):
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        n_filters_time=4,
        filter_time_length=5,
        n_filters_spat=4,
        pool_time_length=4,
        pool_time_stride=2,
        **kwargs,
    )


def _luna_block(n_chans, n_times, n_outputs, **kwargs):
    return LUNARotaryBlockModel(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, **kwargs
    )


MODELS = {
    "EEGNet": _eegnet,
    "ShallowFBCSPNet": _shallow,
    "LUNARotaryBlock": _luna_block,
}

# Each model's dropout-probability constructor keyword, used only to build the
# zero-dropout pair of the compiled ``dropout_off_parity`` check. Timed eager
# and timed compiled cases always use the configured (default) dropout.
_DROPOUT_KWARG = {
    "EEGNet": "drop_prob",
    "ShallowFBCSPNet": "drop_prob",
    "LUNARotaryBlock": "dropout",
}


def build_model(
    model_name: str, *, n_chans: int, n_times: int, n_outputs: int, **kwargs
) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name](
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, **kwargs
    )


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _source():
    paths = set(_git("diff", "--name-only", "HEAD").splitlines())
    paths.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    hashes = {}
    for name in sorted(paths):
        path = ROOT / name
        content = path.read_bytes() if path.is_file() else b""
        hashes[name] = hashlib.sha256(content).hexdigest()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(hashes),
        "dirty_source_hashes": hashes,
    }


def _input(value):
    return {
        "name": "x",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.stride()),
        "stride_unit": "elements",
    }


def _max_rss_bytes():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _cache_policy(mode):
    base = "synthetic fixed inputs; warmup; OS cache uncontrolled; "
    if mode != "compiled":
        return base + "compiler cache: not_applicable"
    return base + (
        "compiler cache: torch._dynamo.reset() before every compile; the "
        "inductor FX-graph/disk cache (TORCHINDUCTOR_CACHE_DIR or the torch "
        "default) is not cleared by this runner; hits/misses of the timed "
        "compile are in compile_diagnostics"
    )


def _record(case_id, mode, device, parameters):
    return {
        "schema_version": 1,
        "case_id": case_id,
        "implementation": mode,
        "status": "unsupported",
        "config": {
            "inputs": None,
            "seed": SEED,
            "threads": torch.get_num_threads(),
            "workers": 0,
            "cache_policy": _cache_policy(mode),
            "autograd": (
                "no_grad (forward only; timed calls run under torch.no_grad)"
                if mode == "exported"
                else "forward plus fresh gradients and an optimizer update each call"
            ),
            "precision": "float32",
            "parameters": parameters,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {"torch": str(torch.__version__)},
            "device": str(device),
            "device_name": platform.node() + "/" + platform.machine(),
        },
        "source": _source(),
        "correctness": {
            "status": "not_checked",
            "method": "not executed",
            "metrics": {},
        },
        "timing": None,
        "memory": {"peak_cuda_allocated_bytes": None},
        "notes": [],
    }


def _deadline_unavailable():
    """Reason bounded compilation is impossible here, or None if it is possible."""
    if not hasattr(signal, "setitimer"):
        return "Bounded compilation needs signal.setitimer (unavailable here)"
    if threading.current_thread() is not threading.main_thread():
        return "Bounded compilation requires the main thread"
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        return "Cannot replace an active process alarm for compilation"
    return None


@contextmanager
def _compile_deadline(seconds):
    if not seconds > 0:
        raise ValueError("compile_timeout_s must be positive")
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise CompileTimeout(f"Compilation timed out after {seconds} s")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _timed_out(exc):
    """True if a compile-deadline timeout is anywhere in the exception chain."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, CompileTimeout):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _seeded_build(model_name, model_kwargs, seed, *, dropout=None):
    """Deterministic parameter initialization, isolated from the caller's RNG.

    ``dropout`` overrides the dropout-probability constructor kwarg (see
    ``_DROPOUT_KWARG``); ``None`` keeps the model's configured default.
    """
    kwargs = dict(model_kwargs)
    if dropout is not None:
        kwargs[_DROPOUT_KWARG[model_name]] = dropout
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build_model(model_name, **kwargs)


def _train_step(call, x, target, optimizer, seed):
    """One seeded forward, cross-entropy backward and optimizer update.

    ``call`` may be an ``nn.Module`` or a ``torch.compile`` wrapper; the
    optimizer must reference the underlying module's parameters. Cross-entropy
    against fixed targets keeps the loss bounded over many timing repeats.
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        y = call(x)
    F.cross_entropy(y, target).backward()
    optimizer.step()
    return y


def _assert_state_close(left, right, **tolerance):
    """Parameters and buffers (e.g. BatchNorm running stats), strict key set."""
    left_state, right_state = left.state_dict(), right.state_dict()
    if set(left_state) != set(right_state):
        raise AssertionError("state_dict keys differ")
    for name, value in left_state.items():
        torch.testing.assert_close(value, right_state[name], msg=name, **tolerance)
    return len(left_state)


def _finite_state(module):
    tensors = list(module.state_dict().values())
    tensors += [p.grad for p in module.parameters() if p.grad is not None]
    return all(bool(torch.isfinite(t).all()) for t in tensors if t.is_floating_point())


def _fail(result, check_name, notes, *, checks=None, method=None, mismatch=False):
    """Record a failed case; correctness is failed, never left as passed."""
    result["status"] = "failed"
    result["correctness"] = {
        "status": "failed",
        "method": method or "see notes",
        "metrics": {
            "failed_check": check_name,
            "kind": "mismatch" if mismatch else "error",
            "checks": checks or {},
        },
    }
    result["notes"] = list(notes)
    return result


def _execution_failure(result, check_name, exc, checks=None, method=None):
    parameters = result["config"]["parameters"]
    if _timed_out(exc):
        parameters["compile_timed_out"] = True
        note = (
            f"Compilation timed out (deadline {parameters['compile_timeout_s']} s) "
            f"in check {check_name}; a compiler stall is a reported failure."
        )
    else:
        note = f"Execution failed in check {check_name}."
    return _fail(
        result,
        check_name,
        [note, f"{type(exc).__name__}: {exc}"],
        checks=checks,
        method=method,
    )


def _mismatch_failure(result, check_name, exc, checks=None, method=None):
    return _fail(
        result,
        check_name,
        [f"Correctness mismatch in check {check_name}: {exc}"],
        checks=checks,
        method=method,
        mismatch=True,
    )


def _finish_timing(result, run, module, min_run_time):
    """Headline timing, then a post-timing finiteness check of the timed state."""
    result.update(
        measure_step(run, device=torch.device("cpu"), min_run_time=min_run_time)
    )
    finite = _finite_state(module)
    result["config"]["parameters"]["post_timing_finite"] = finite
    if not finite:
        checks = result["correctness"]["metrics"].get("checks", {})
        return _fail(
            result,
            "post_timing_finite",
            ["Non-finite parameters, buffers or gradients after timing."],
            checks=checks,
            method=result["correctness"]["method"],
        )
    result["status"] = "ok"
    return result


def _run_eager(result, model_name, model_kwargs, x, target, seed, lr, min_run_time):
    """Eager mode is the oracle; its own check is deterministic self-repeatability."""
    method = (
        "deterministic self-repeatability: an identically-seeded clone reproduces "
        "output, parameters and buffers after one update (rtol=1e-5, atol=1e-6); "
        "not an independent reference"
    )
    model = _seeded_build(model_name, model_kwargs, seed).train()
    clone = copy.deepcopy(model)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_clone = torch.optim.SGD(clone.parameters(), lr=lr)

    def run():
        return _train_step(model, x, target, optimizer_model, seed)

    try:
        y_model = run()
        y_clone = _train_step(clone, x, target, optimizer_clone, seed)
    except _EXECUTION_EXCEPTIONS as exc:
        return _execution_failure(result, "self_repeatability", exc, method=method)
    try:
        torch.testing.assert_close(y_model, y_clone, rtol=1e-5, atol=1e-6)
        checked = _assert_state_close(model, clone, rtol=1e-5, atol=1e-6)
    except AssertionError as exc:
        return _mismatch_failure(result, "self_repeatability", exc, method=method)
    del clone, y_clone
    checks = {"self_repeatability": {"status": "passed", "state_entries": checked}}
    result["correctness"] = {
        "status": "passed",
        "method": method,
        "metrics": {"checks": checks},
    }
    parameters = result["config"]["parameters"]
    parameters["effective_samples"] = x.shape[0]
    parameters["optimizer_updates"] = 1
    result["notes"] = [
        (
            "Eager mode is the oracle for compiled/exported comparisons; its own "
            "check is deterministic self-repeatability, not an independent "
            "reference implementation."
        ),
        (
            "effective_samples/optimizer_updates describe one run() invocation; "
            "measure_step repeats run() many times for timing statistics only, "
            "not a multiplied count."
        ),
        (
            "Headline includes zero_grad, seeded forward, cross-entropy backward "
            "and optimizer.step of the configured model (default dropout)."
        ),
        "CPU only; no CUDA/MPS qualification claim.",
    ]
    return _finish_timing(result, run, model, min_run_time)


def _graph_count():
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def _compile_diagnostics():
    """Compiler invocation/cache conditions and graph counts of the timed compile.

    Read after the timed model's first (compiling) call, outside headline
    timing. Dynamo's process-wide counters are cleared right after
    ``torch._dynamo.reset()`` and before that compile, so they describe it only.
    """
    from torch._dynamo.utils import counters
    from torch._inductor import config as inductor_config

    return {
        "backend": "inductor",
        "fullgraph": False,
        "dynamo_reset_before_compile": True,
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "fxgraph_cache_hit": int(counters["inductor"]["fxgraph_cache_hit"]),
        "fxgraph_cache_miss": int(counters["inductor"]["fxgraph_cache_miss"]),
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "fx_graph_cache": bool(inductor_config.fx_graph_cache),
    }


def _fresh_compile_state():
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()


def _compiled_pair(model_name, model_kwargs, seed, lr, dropout):
    reference = _seeded_build(model_name, model_kwargs, seed, dropout=dropout).train()
    target = copy.deepcopy(reference)
    return (
        reference,
        target,
        torch.optim.SGD(reference.parameters(), lr=lr),
        torch.optim.SGD(target.parameters(), lr=lr),
    )


def _compare_step(y_target, y_reference, target, reference):
    torch.testing.assert_close(y_target, y_reference, **PARITY)
    return _assert_state_close(target, reference, **PARITY)


_COMPILED_METHOD = (
    "structured checks (correctness.metrics.checks) at rtol=1e-4, atol=1e-5: "
    "compiled vs identically-seeded eager clone with dropout off "
    "(dropout_off_parity, then partial_batch_parity on the same pair), "
    "dropout on under inductor fallback_random "
    "(dropout_on_fallback_random_parity), and finiteness of the timed "
    "configured model's first compiled step (timed_configuration_finite)"
)


def _check_equal_work(module, call, x, target, optimizer, seed):
    """Execute the actual timed step and count samples, forwards and updates.

    Hooks are removed even on error. This is not equality inferred from config.
    """
    observed = {"effective_samples": 0, "optimizer_updates": 0, "forwards": 0}

    def forward(owner, args):
        observed["effective_samples"] += args[0].shape[0]
        observed["forwards"] += 1

    def update(owner, args, kwargs):
        observed["optimizer_updates"] += 1

    with module.register_forward_pre_hook(forward), optimizer.register_step_post_hook(update):
        output = _train_step(call, x, target, optimizer, seed)
    assert output.shape[0] == x.shape[0]
    assert observed == {"effective_samples": x.shape[0], "optimizer_updates": 1, "forwards": 1}
    return {"status": "passed", **observed}


def _run_compiled(
    result, model_name, model_kwargs, x, target, seed, lr, timeout_s, min_run_time
):
    """Compiled training of the configured model, qualified by structured checks.

    A compiler timeout or compile/execution error is ``failed``; a numeric
    mismatch is ``failed``; only an impossible bounded-compile setup is
    ``unsupported``. Every check records its own dropout/config override.
    """
    unavailable = _deadline_unavailable()
    if unavailable is not None:
        result["notes"] = [f"Compiled training unsupported: {unavailable}"]
        return result
    parameters = result["config"]["parameters"]
    checks = {}

    def failure(check, exc):
        return _execution_failure(result, check, exc, checks, _COMPILED_METHOD)

    def mismatch(check, exc):
        return _mismatch_failure(result, check, exc, checks, _COMPILED_METHOD)

    # 1) dropout_off_parity and 2) partial_batch_parity on one dropout-free pair.
    _fresh_compile_state()
    reference, model, opt_reference, opt_model = _compiled_pair(
        model_name, model_kwargs, seed, lr, dropout=0.0
    )
    start = perf_counter()
    try:
        with _compile_deadline(timeout_s):
            compiled = torch.compile(model, backend="inductor", fullgraph=False)
            y_model = _train_step(compiled, x, target, opt_model, seed)
    except _EXECUTION_EXCEPTIONS as exc:
        return failure("dropout_off_parity", exc)
    compile_s = perf_counter() - start
    y_reference = _train_step(reference, x, target, opt_reference, seed)
    try:
        entries = _compare_step(y_model, y_reference, model, reference)
    except AssertionError as exc:
        return mismatch("dropout_off_parity", exc)
    checks["dropout_off_parity"] = {
        "status": "passed",
        "dropout_override": 0.0,
        "inductor_config_patch": {},
        "compared": "output, then parameters and buffers after one update",
        "state_entries": entries,
        "compile_s": compile_s,
    }

    partial = x.shape[0] - 1
    if partial < 1:
        checks["partial_batch_parity"] = {
            "status": "not_applicable",
            "reason": "batch_size 1 has no smaller partial batch",
        }
    else:
        graphs_before = _graph_count()
        try:
            with _compile_deadline(timeout_s):
                y_model = _train_step(
                    compiled, x[:partial], target[:partial], opt_model, seed
                )
        except _EXECUTION_EXCEPTIONS as exc:
            return failure("partial_batch_parity", exc)
        graphs_after = _graph_count()
        y_reference = _train_step(
            reference, x[:partial], target[:partial], opt_reference, seed
        )
        try:
            _compare_step(y_model, y_reference, model, reference)
        except AssertionError as exc:
            return mismatch("partial_batch_parity", exc)
        checks["partial_batch_parity"] = {
            "status": "passed",
            "batch_size": partial,
            "dropout_override": 0.0,
            "unique_graphs_before": graphs_before,
            "unique_graphs_after": graphs_after,
            "recompiled": graphs_after > graphs_before,
            "compared": "second update at a partial final batch, same pair",
        }
    del reference, model, compiled, opt_reference, opt_model

    # 3) dropout_on_fallback_random_parity: configured dropout, eager RNG kernels.
    _fresh_compile_state()
    reference, model, opt_reference, opt_model = _compiled_pair(
        model_name, model_kwargs, seed, lr, dropout=None
    )
    from torch._inductor import config as inductor_config

    start = perf_counter()
    try:
        with inductor_config.patch(fallback_random=True), _compile_deadline(timeout_s):
            compiled = torch.compile(model, backend="inductor", fullgraph=False)
            y_model = _train_step(compiled, x, target, opt_model, seed)
    except _EXECUTION_EXCEPTIONS as exc:
        return failure("dropout_on_fallback_random_parity", exc)
    compile_s = perf_counter() - start
    y_reference = _train_step(reference, x, target, opt_reference, seed)
    try:
        entries = _compare_step(y_model, y_reference, model, reference)
    except AssertionError as exc:
        return mismatch("dropout_on_fallback_random_parity", exc)
    checks["dropout_on_fallback_random_parity"] = {
        "status": "passed",
        "dropout_override": None,
        "inductor_config_patch": {"fallback_random": True},
        "compared": "output, then parameters and buffers after one update",
        "state_entries": entries,
        "compile_s": compile_s,
    }
    del reference, model, compiled, opt_reference, opt_model

    # 4) The timed configured model: default inductor config, default dropout.
    _fresh_compile_state()
    model = _seeded_build(model_name, model_kwargs, seed).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start = perf_counter()
    try:
        with _compile_deadline(timeout_s):
            compiled = torch.compile(model, backend="inductor", fullgraph=False)
            y_model = _train_step(compiled, x, target, optimizer, seed)
    except _EXECUTION_EXCEPTIONS as exc:
        parameters["compile_setup_s"] = perf_counter() - start
        return failure("timed_configuration_finite", exc)
    parameters["compile_setup_s"] = perf_counter() - start
    parameters["compile_diagnostics"] = _compile_diagnostics()
    if not (bool(torch.isfinite(y_model).all()) and _finite_state(model)):
        return _fail(
            result,
            "timed_configuration_finite",
            ["Non-finite output, gradients or state after the first compiled step."],
            checks=checks,
            method=_COMPILED_METHOD,
        )
    try:
        checks["executed_equal_work"] = _check_equal_work(
            model, compiled, x, target, optimizer, seed
        )
    except (AssertionError, ValueError) as exc:
        return mismatch("executed_equal_work", exc)
    checks["timed_configuration_finite"] = {
        "status": "passed",
        "dropout_override": None,
        "inductor_config_patch": {},
        "compared": (
            "finite output, gradients, parameters and buffers after the first "
            "compiled step; dropout masks of inductor's default RNG not compared"
        ),
    }
    result["correctness"] = {
        "status": "passed",
        "method": _COMPILED_METHOD,
        "metrics": {"checks": checks},
    }
    parameters["effective_samples"] = x.shape[0]
    parameters["optimizer_updates"] = 1

    def run():
        return _train_step(compiled, x, target, optimizer, seed)

    result["notes"] = [
        (
            "Timed work equals eager mode's: same configured model (default "
            "dropout), input, targets, loss and optimizer. torch.compile wraps the "
            "module (forward and AOT backward compiled); zero_grad, the loss and "
            "optimizer.step run eagerly."
        ),
        (
            "Scoped limitation: dropout masks of inductor's default RNG (used by "
            "the timed model) differ from eager's and are not compared; numerics "
            "are compared with dropout off, and dropout-on parity only under "
            "torch._inductor.config.fallback_random=True (eager RNG kernels)."
        ),
        (
            "Compilation/setup, compile diagnostics and correctness are outside "
            "headline timing; headline is the steady-state compiled call measured "
            "via torch.utils.benchmark.Timer."
        ),
        (
            "effective_samples/optimizer_updates describe one run() invocation; "
            "measure_step repeats run() many times for timing statistics only, "
            "not a multiplied count."
        ),
        (
            "Direct-module qualification only; skorch estimator compile "
            "behaviour is not qualified here."
        ),
        "CPU only; no CUDA/MPS qualification claim.",
    ]
    graphs_before = _graph_count()
    outcome = _finish_timing(result, run, model, min_run_time)
    graphs_after = _graph_count()
    parameters["timing_compile_diagnostics"] = {
        "unique_graphs_before": graphs_before,
        "unique_graphs_after": graphs_after,
        "recompiled": graphs_after > graphs_before,
        "after": _compile_diagnostics(),
    }
    return outcome


def _export_program(module, args, dynamic_shapes):
    """Export seam (patched in tests to inject a deliberately wrong artifact).

    Non-strict explicitly, the default from torch 2.8: in strict export every
    call shares one Dynamo cache entry, whose limit (8) torch 2.6 reaches.
    """
    return torch.export.export(
        module, args, dynamic_shapes=dynamic_shapes, strict=False
    )


def _batch_sizes(batch_size, dynamic_batch):
    """Example batch, then batch 1, a partial batch and a larger batch if dynamic."""
    if not dynamic_batch:
        return [batch_size]
    sizes = []
    for size in (batch_size, 1, batch_size - 1, batch_size + 1):
        if size >= 1 and size not in sizes:
            sizes.append(size)
    return sizes


_EXPORTED_METHOD = (
    "torch.export round trip (export, save, load, module() executed) vs "
    "eval-mode eager forward at rtol=1e-4, atol=1e-5, for every batch size "
    "in the checks list (one artifact)"
)


def _run_exported(
    result, model_name, model_kwargs, x, seed, dynamic_batch, min_run_time
):
    """Forward-only exported artifact: executed after save/reload, not type-checked.

    Export failure is ``unsupported`` (recorded with notes); a save/load/unlift
    or execution error, or a mismatch at any checked batch size, is ``failed``
    with failed correctness.
    """
    parameters = result["config"]["parameters"]
    oracle = _seeded_build(model_name, model_kwargs, seed).eval()
    dynamic_shapes = None
    if dynamic_batch:
        batch_dim = torch.export.Dim("batch", min=1, max=max(x.shape[0] * 4, 8))
        dynamic_shapes = ({0: batch_dim},)
    setup = {}
    start = perf_counter()
    try:
        # Export an independent copy so the artifact never aliases the oracle.
        exported = _export_program(copy.deepcopy(oracle), (x,), dynamic_shapes)
    except _EXPORT_EXCEPTIONS as exc:
        result["notes"] = [f"Export unsupported: {type(exc).__name__}: {exc}"]
        return result
    setup["export_s"] = perf_counter() - start
    try:
        start = perf_counter()
        buffer = io.BytesIO()
        torch.export.save(exported, buffer)
        setup["save_s"] = perf_counter() - start
        buffer.seek(0)
        start = perf_counter()
        reloaded = torch.export.load(buffer)
        setup["load_s"] = perf_counter() - start
        start = perf_counter()
        module = reloaded.module()  # unlift once, outside headline timing
        setup["unlift_s"] = perf_counter() - start
    except _EXECUTION_EXCEPTIONS as exc:
        parameters["export_setup_s"] = setup
        return _execution_failure(result, "save_load", exc, method=_EXPORTED_METHOD)
    parameters["export_setup_s"] = setup

    checks = []
    for size in _batch_sizes(x.shape[0], dynamic_batch):
        name = f"batch_{size}"
        sample = x
        if size != x.shape[0]:
            sample = torch.randn(
                size,
                x.shape[1],
                x.shape[2],
                generator=torch.Generator().manual_seed(seed + size),
            )
        with torch.no_grad():
            expected = oracle(sample)
            try:
                actual = module(sample)
            except _EXECUTION_EXCEPTIONS as exc:
                failure = _execution_failure(
                    result, name, exc, {"batch_checks": checks}, _EXPORTED_METHOD
                )
                parameters["dynamic_batch_checks"] = checks if dynamic_batch else None
                return failure
        try:
            torch.testing.assert_close(actual, expected, **PARITY)
        except AssertionError as exc:
            parameters["dynamic_batch_checks"] = checks if dynamic_batch else None
            return _mismatch_failure(
                result, name, exc, {"batch_checks": checks}, _EXPORTED_METHOD
            )
        checks.append({"batch_size": size, "status": "passed"})
    parameters["dynamic_batch_checks"] = checks if dynamic_batch else None
    result["correctness"] = {
        "status": "passed",
        "method": _EXPORTED_METHOD,
        "metrics": {"checks": {"batch_checks": checks}},
    }
    parameters["effective_samples"] = x.shape[0]
    parameters["optimizer_updates"] = 0

    def run():
        with torch.no_grad():
            return module(x)

    result["notes"] = [
        (
            "Exported mode is forward-only: eval mode, timed calls under "
            "torch.no_grad; training occurs only in eager/compiled modes."
        ),
        (
            "The exported artifact is saved, reloaded and actually executed, not "
            "only type-checked. reloaded.module() is unlifted once, outside "
            "headline timing; export/save/load/unlift costs are in export_setup_s."
        ),
        "CPU only; no CUDA/MPS qualification claim.",
    ]
    result.update(
        measure_step(run, device=torch.device("cpu"), min_run_time=min_run_time)
    )
    parameters["post_timing_finite"] = _finite_state(module)
    result["status"] = "ok"
    return result


def run_case(
    model_name,
    *,
    mode,
    device,
    batch_size,
    n_times,
    n_chans=8,
    n_outputs=2,
    seed=SEED,
    learning_rate=0.1,
    compile_timeout_s=90,
    min_run_time=0.1,
    dynamic_batch=False,
):
    """Qualify and time one execution mode for one model on synthetic CPU data.

    ``mode`` is ``eager`` (training step, the oracle), ``compiled`` (training
    step of the same configured model under ``torch.compile``, with structured
    parity checks) or ``exported`` (forward-only ``torch.export`` round trip vs
    an eval-mode eager forward). Setup, correctness and diagnostics are
    excluded from headline timing. CPU only.
    """
    _check_mode(mode)
    if (isinstance(compile_timeout_s, bool)
            or not math.isfinite(compile_timeout_s) or compile_timeout_s <= 0):
        raise ValueError("compile_timeout_s must be positive and finite")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if any(
        type(value) is not int or value <= 0
        for value in (batch_size, n_times, n_chans, n_outputs)
    ):
        raise ValueError(
            "batch_size, n_times, n_chans and n_outputs must be positive integers"
        )
    if dynamic_batch and mode != "exported":
        raise ValueError("dynamic_batch only applies to mode='exported'")

    training = mode != "exported"
    parameters = {
        "model": model_name,
        "batch_size": batch_size,
        "n_times": n_times,
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "learning_rate": learning_rate if training else None,
        "loss": "cross_entropy" if training else None,
        "timed_configuration": {"dropout_override": None, "train_mode": training},
        "compile_setup_s": None,
        "compile_timeout_s": compile_timeout_s,
        "compile_timed_out": False,
        "compile_backend": "inductor" if mode == "compiled" else None,
        "fullgraph": False if mode == "compiled" else None,
        "export_backend": "torch.export" if mode == "exported" else None,
        "export_setup_s": None,
        "dynamic_batch": dynamic_batch,
        "dynamic_batch_checks": None,
        "effective_samples": 0,
        "optimizer_updates": 0,
        "post_timing_finite": None,
        "cpu_peak_rss": {
            "ru_maxrss_bytes_before": _max_rss_bytes(),
            "ru_maxrss_bytes_after": None,
            "scope": _RSS_SCOPE,
        },
    }
    result = _record(f"execution/{model_name}/{mode}", mode, device, parameters)
    try:
        if str(device) != "cpu":
            result["notes"] = [
                "Only CPU is qualified by this runner; no GPU execution."
            ]
            return result

        model_kwargs = {"n_chans": n_chans, "n_times": n_times, "n_outputs": n_outputs}
        try:
            build_model(model_name, **model_kwargs)
        except (ValueError, RuntimeError) as exc:
            result["notes"] = [
                f"Unsupported model configuration: {type(exc).__name__}: {exc}"
            ]
            return result

        x = torch.randn(
            batch_size, n_chans, n_times, generator=torch.Generator().manual_seed(seed)
        )
        target = torch.arange(batch_size) % n_outputs
        result["config"]["inputs"] = [_input(x)]
        if mode == "eager":
            return _run_eager(
                result,
                model_name,
                model_kwargs,
                x,
                target,
                seed,
                learning_rate,
                min_run_time,
            )
        if mode == "compiled":
            return _run_compiled(
                result,
                model_name,
                model_kwargs,
                x,
                target,
                seed,
                learning_rate,
                compile_timeout_s,
                min_run_time,
            )
        return _run_exported(
            result, model_name, model_kwargs, x, seed, dynamic_batch, min_run_time
        )
    finally:
        parameters["cpu_peak_rss"]["ru_maxrss_bytes_after"] = _max_rss_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--mode", choices=list(MODES), default="eager")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-times", type=int, default=128)
    parser.add_argument("--n-chans", type=int, default=8)
    parser.add_argument("--n-outputs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--compile-timeout-s", type=float, default=90)
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if args.require_clean and _git("status", "--porcelain"):
        parser.error("Retained paired evidence requires a clean committed source tree")
    torch.set_num_threads(args.threads)
    result = run_case(
        args.model,
        mode=args.mode,
        device="cpu",
        batch_size=args.batch_size,
        n_times=args.n_times,
        n_chans=args.n_chans,
        n_outputs=args.n_outputs,
        learning_rate=args.learning_rate,
        compile_timeout_s=args.compile_timeout_s,
        min_run_time=args.min_run_time,
        dynamic_batch=args.dynamic_batch,
    )
    from benchmarks.acceleration._qualification import historical_only

    historical_only(result, "Direct-module diagnostics; no complete matrix or launch attestation.")
    write_result(result, args.output)
    print(f"{result['case_id']}: {result['status']} -> {args.output}")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Paired CPU LUNA rotary-setup qualification; no pretrained weights or downloads.

Each invocation compares the frozen pre-reuse expression with production.
Uninstrumented steady-state timing excludes construction, correctness, profiling
and cold compilation (reported separately). Eval forward/backward is not a
training epoch or optimizer step. Full LUNA executes the actual classifier with
explicit synthetic channel locations, using its public forward argument.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from benchmarks.acceleration._common import measure_step, write_result
from braindecode.functional import apply_rotary
from braindecode.models import LUNA
from braindecode.models.luna import _RotarySelfAttentionBlock

ROOT = Path(__file__).resolve().parents[2]
BASE = "3991b3e6c6700b1b43f1e1d58c7b081ab6456a83"


class _BeforeReuse(_RotarySelfAttentionBlock):
    """Frozen old forward/rotation, never candidate rotary helpers."""

    def _rotate_queries_or_keys(self, tensor: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            tensor.shape[-2], device=tensor.device, dtype=tensor.dtype
        )
        freqs = self.rotary_emb(positions)
        tensor.reshape(list(tensor.shape[:-1]) + [-1, 2])
        return apply_rotary(tensor, freqs.cos(), freqs.sin()).to(tensor.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q = self._rotate_queries_or_keys(q)
        k = self._rotate_queries_or_keys(k)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale,
        )
        return self.proj_drop(self.proj(attn.transpose(1, 2).flatten(2)))


def _old_blocks(module):
    for name, child in module.named_children():
        if isinstance(child, _RotarySelfAttentionBlock):
            old = _BeforeReuse(
                child.dim, child.num_heads, child.qkv_proj.bias is not None,
                child.scale, child.attn_drop, child.proj_drop.p,
            )
            old.load_state_dict(child.state_dict(), strict=True)
            setattr(module, name, old)
        else:
            _old_blocks(child)


def _git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def _source():
    paths = set(_git("diff", "--name-only", "HEAD").splitlines())
    paths.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    hashes = {
        name: hashlib.sha256(
            (ROOT / name).read_bytes() if (ROOT / name).is_file() else b"<deleted>"
        ).hexdigest()
        for name in sorted(paths)
    }
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(hashes), "dirty_source_hashes": hashes,
    }


def _input(name, value):
    return {
        "name": name, "shape": list(value.shape), "dtype": str(value.dtype),
        "strides": list(value.stride()), "stride_unit": "elements",
    }


def _step(function, module, x, locations, mode):
    def run():
        if mode == "forward":
            with torch.no_grad():
                return function(x) if locations is None else function(
                    x, channel_locations=locations
                )
        module.zero_grad(set_to_none=True)
        x.grad = None
        out = function(x) if locations is None else function(
            x, channel_locations=locations
        )
        out.square().mean().backward()
        return out
    return run


def _gradients(module, x):
    return {
        name: None if value.grad is None else value.grad.detach().clone()
        for name, value in [("input", x), *module.named_parameters()]
    }


def _parity(actual, expected, module, x, gradients, mode):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    count = 0
    if mode == "forward_backward":
        current = _gradients(module, x)
        assert set(current) == set(gradients)
        for name, value in current.items():
            reference = gradients[name]
            if reference is None:
                assert value is None, name
            else:
                assert value is not None and torch.isfinite(value).all(), name
                torch.testing.assert_close(value, reference, rtol=5e-4, atol=3e-5)
                count += 1
    return {
        "gradient_tensors": count,
        "max_abs_output_error": (actual - expected).abs().max().item(),
        "output_rtol": 2e-4, "output_atol": 2e-5,
        "gradient_rtol": 5e-4, "gradient_atol": 3e-5,
    }


def _profile(module, run, output):
    """Extra eager call only: native setup counts, not headline latency."""
    calls = []

    def hook(_module, _args, _output):
        calls.append(1)

    handles = [
        child.rotary_emb.register_forward_hook(hook)
        for child in module.modules()
        if isinstance(child, _RotarySelfAttentionBlock)
    ]
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profiler:
            run()
    finally:
        for handle in handles:
            handle.remove()
    profiler.export_chrome_trace(str(output))
    selected = ("aten::arange", "aten::cos", "aten::sin", "aten::repeat_interleave")
    return {
        "separate_instrumented_eager_call": True,
        "rotary_module_forward_calls": len(calls),
        "operators": [
            {"name": event.key, "calls": event.count,
             "self_cpu_time_us": event.self_cpu_time_total}
            for event in profiler.key_averages() if event.key in selected
        ],
        "trace": str(output),
    }


def run_pair(subject, backend, mode, *, output_dir, threads=1,
             min_run_time=0.5, order="old-first", require_clean=False):
    """Write two strict records; return non-ok evidence without inventing parity."""
    if subject not in ("block", "luna") or backend not in ("eager", "compiled"):
        raise ValueError("Unknown subject/backend")
    if mode not in ("forward", "forward_backward"):
        raise ValueError("Unknown mode")
    if order not in ("old-first", "new-first") or threads < 1:
        raise ValueError("Invalid order/threads")
    source = _source()
    if require_clean and source["dirty"]:
        raise ValueError("Retained evidence requires clean committed source")
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError("output_dir must already exist")
    torch.set_num_threads(threads)
    torch.manual_seed(231)
    started = perf_counter()
    locations = None
    if subject == "block":
        candidate = _RotarySelfAttentionBlock(256, num_heads=8)
        old = _BeforeReuse(256, num_heads=8)
        shape = (2, 25, 256)
        constructor = {"dim": 256, "num_heads": 8}
    else:
        constructor = {"n_chans": 22, "n_times": 1000, "n_outputs": 4}
        candidate = LUNA(**constructor)
        old = copy.deepcopy(candidate)
        _old_blocks(old)
        shape = (2, 22, 1000)
        locations = torch.randn(2, 22, 3)
    old.load_state_dict(candidate.state_dict(), strict=True)
    assert set(old.state_dict()) == set(candidate.state_dict())
    old.eval()
    candidate.eval()
    x = torch.randn(shape, requires_grad=mode == "forward_backward")
    reference_x = x.detach().clone().requires_grad_(x.requires_grad)
    expected = _step(old, old, reference_x, locations, mode)().detach()
    gradients = _gradients(old, reference_x)
    setup_s = perf_counter() - started
    names = ("old", "new") if order == "old-first" else ("new", "old")
    records = []
    for name in names:
        module = old if name == "old" else candidate
        xi = x.detach().clone().requires_grad_(x.requires_grad)
        parameters = {
            "subject": subject, "backend": backend, "mode": mode,
            "constructor": constructor, "training": False,
            "pair_order": order, "oracle_base": BASE,
            "setup_and_reference_s": setup_s, "compile_cold_s": 0.0,
            "fullgraph": backend == "compiled",
            "checkpoint_keys_identical": True,
            "frequency_requires_grad": False,
            "loss": "square().mean()" if mode == "forward_backward" else None,
        }
        record = {
            "schema_version": 1,
            "case_id": f"rotary/{subject}/{backend}/{mode}",
            "implementation": "before_reuse" if name == "old" else "shared_tables",
            "status": "failed",
            "config": {
                "inputs": [_input("x", xi)] + (
                    [_input("channel_locations", locations)] if locations is not None else []
                ),
                "seed": 231, "threads": threads, "workers": 0,
                "cache_policy": "job-private compiler cache; warmup; OS cache uncontrolled",
                "autograd": "no_grad" if mode == "forward" else "input and parameter gradients; reset each call",
                "precision": "float32; no autocast or precision-default changes",
                "parameters": parameters,
            },
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {"torch": str(torch.__version__)},
                "device": "cpu", "device_name": platform.node() + "/" + platform.machine(),
            },
            "source": source,
            "correctness": {
                "status": "not_checked", "method": "frozen old-expression oracle",
                "metrics": {},
            },
            "timing": None, "memory": {"peak_cuda_allocated_bytes": None},
            "notes": [],
        }
        function = module
        compile_started = None
        try:
            if backend == "compiled":
                torch.compiler.reset()
                compile_started = perf_counter()
                function = torch.compile(module, backend="inductor", fullgraph=True)
                _step(function, module, xi, locations, mode)()
                parameters["compile_cold_s"] = perf_counter() - compile_started
            run = _step(function, module, xi, locations, mode)
            actual = run().detach()
            metrics = _parity(actual, expected, module, xi, gradients, mode)
            record["correctness"] = {
                "status": "passed",
                "method": "frozen pre-reuse block expression; strict state keys/load; full-model output and named gradients",
                "metrics": metrics,
            }
            record.update(measure_step(
                run, device=torch.device("cpu"), min_run_time=min_run_time
            ))
            if backend == "eager":
                parameters["profile"] = _profile(
                    module, run, output_dir / f"{subject}-{mode}-{name}-trace.json"
                )
            record["status"] = "ok"
            record["notes"] = [
                "CPU only. No inference about native GPU or end-to-end training throughput.",
                "Construction/reference checks, compilation and profiling excluded from steady-state timing.",
                "compile_cold_s includes wrapper creation and first forward/backward; cold relative to this process, not guaranteed cold shared OS/compiler caches.",
                "Whole-LUNA uses real default architecture and explicit synthetic channel locations; no pretrained weights.",
            ]
        except torch._dynamo.exc.Unsupported as exc:
            record["status"] = "unsupported"
            record["notes"] = [f"Baseline/backend capability must be reviewed: {type(exc).__name__}: {exc}"]
        except (AssertionError, RuntimeError) as exc:
            record["correctness"]["status"] = "failed"
            record["notes"] = [f"{type(exc).__name__}: {exc}"]
        finally:
            if compile_started is not None and parameters["compile_cold_s"] == 0.0:
                parameters["compile_cold_s"] = perf_counter() - compile_started
        path = output_dir / f"{subject}-{backend}-{mode}-{name}.json"
        write_result(record, path)
        print(f"{path.name}: {record['status']}", flush=True)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", choices=["block", "luna"], required=True)
    parser.add_argument("--backend", choices=["eager", "compiled"], required=True)
    parser.add_argument("--mode", choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--order", choices=["old-first", "new-first"], default="old-first")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    records = run_pair(
        args.subject, args.backend, args.mode, output_dir=args.output_dir,
        threads=args.threads, min_run_time=args.min_run_time, order=args.order,
        require_clean=args.require_clean,
    )
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

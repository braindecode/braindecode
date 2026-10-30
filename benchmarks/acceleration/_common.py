"""Synchronized native timing and strict version-one experiment records."""

import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.benchmark import Timer


def measure_step(
    run: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 5,
    min_run_time: float = 1.0,
) -> dict:
    """Measure the caller's work without changing precision, grad or RNG policy.

    Returns ``timing`` and ``memory`` sections for :func:`write_result`. Output
    conversion/checking belongs to the caller, outside this function. CUDA peak
    covers first call, warmup, calibration and measured blocks, including live
    allocations that preceded the call. The CUDA peak counter is reset.
    """
    if not math.isfinite(min_run_time) or min_run_time <= 0:
        raise ValueError("min_run_time must be positive and finite")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if device.type == "cpu":
        synchronize = lambda: None
    elif device.type == "cuda" and torch.cuda.is_available():
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise NotImplementedError(f"Unavailable device: {device}")
        synchronize = lambda: torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        synchronize = torch.mps.synchronize
    else:
        raise NotImplementedError(f"Unqualified or unavailable device: {device}")

    def synchronized_run():
        output = run()
        synchronize()
        return output

    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    first_output = synchronized_run()
    first_call_s = perf_counter() - start
    del first_output  # outside the first-call clock; no casts or checks
    for _ in range(warmup):
        synchronized_run()
    measurement = Timer(
        stmt="run()",
        globals={"run": synchronized_run},
        timer=perf_counter,  # never let Timer synchronize unrelated CUDA on CPU
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=min_run_time)
    return {
        "timing": {
            "first_call_s": first_call_s,
            "median_s": measurement.median,
            "iqr_s": measurement.iqr,
            "raw_block_s": list(measurement.raw_times),
            "measurement_blocks": len(measurement.raw_times),
            "calls_per_block": measurement.number_per_run,
            "warmup": warmup,
            "min_run_time_s": min_run_time,
        },
        "memory": {
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            )
        },
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, keys, name):
    _require(
        type(value) is dict and set(value) == set(keys.split()),
        f"Invalid {name} fields",
    )


def _text(value):
    return type(value) is str and bool(value.strip())


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _json_value(value):
    """Reject lossy JSON coercions, nonfinite numbers and arbitrary objects."""
    if type(value) is dict:
        _require(all(type(k) is str for k in value), "JSON keys must be strings")
        _require(
            "speedup" not in value,
            "speedup belongs in separate paired evidence, not v1 records",
        )
        for item in value.values():
            _json_value(item)
    elif type(value) is list:
        for item in value:
            _json_value(item)
    else:
        _require(
            value is None
            or type(value) in (str, bool, int)
            or (type(value) is float and math.isfinite(value)),
            "Expected finite, native JSON values",
        )


def _validate(result):
    _json_value(result)
    _keys(
        result,
        "schema_version case_id status implementation config environment source correctness timing memory notes",
        "envelope",
    )
    _require(
        type(result["schema_version"]) is int and result["schema_version"] == 1,
        "Expected schema_version=1",
    )
    _require(
        _text(result["case_id"]) and _text(result["implementation"]),
        "Case and implementation must be named",
    )
    status = result["status"]
    _require(status in ("ok", "unsupported", "failed"), "Invalid status")
    _require(
        type(result["notes"]) is list and all(_text(n) for n in result["notes"]),
        "notes must be strings",
    )
    if status != "ok":
        _require(bool(result["notes"]), "Non-ok cases require explanatory notes")

    config = result["config"]
    _keys(
        config,
        "inputs seed threads workers cache_policy autograd precision parameters",
        "config",
    )
    _require(config["seed"] is None or _integer(config["seed"]), "Invalid seed")
    _require(
        _integer(config["threads"], 1) and _integer(config["workers"]),
        "Invalid threads/workers",
    )
    for name in ("cache_policy", "autograd", "precision"):
        _require(_text(config[name]), f"Missing {name}")
    _require(type(config["parameters"]) is dict, "parameters must be a dict")
    inputs = config["inputs"]
    if inputs is None:
        _require(status != "ok", "ok cases require input metadata")
    else:
        _require(type(inputs) is list, "inputs must be a list")
        for item in inputs:
            _keys(item, "name shape dtype strides stride_unit", "input")
            _require(
                _text(item["name"]) and _text(item["dtype"]), "Invalid input identity"
            )
            _require(
                type(item["shape"]) is list and all(_integer(n) for n in item["shape"]),
                "Invalid shape",
            )
            _require(
                type(item["strides"]) is list
                and len(item["strides"]) == len(item["shape"])
                and all(type(n) is int for n in item["strides"]),
                "Invalid strides",
            )
            _require(
                item["stride_unit"] in ("bytes", "elements"), "Invalid stride unit"
            )

    env = result["environment"]
    _keys(env, "python platform dependencies device device_name", "environment")
    for name in ("python", "platform", "device"):
        _require(_text(env[name]), f"Missing {name}")
    _require(
        _text(env["device_name"]) or (status != "ok" and env["device_name"] is None),
        "Missing device identity",
    )
    _require(
        type(env["dependencies"]) is dict
        and bool(env["dependencies"])
        and all(
            _text(k) and (_text(v) or v is None) for k, v in env["dependencies"].items()
        ),
        "Invalid dependencies",
    )

    source = result["source"]
    _keys(source, "commit dirty dirty_source_hashes", "source")
    _require(
        type(source["commit"]) is str
        and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source["commit"]) is not None,
        "Expected full source commit",
    )
    _require(type(source["dirty"]) is bool, "dirty must be boolean")
    hashes = source["dirty_source_hashes"]
    _require(
        type(hashes) is dict
        and all(
            _text(k) and type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
            for k, v in hashes.items()
        ),
        "Expected SHA256 dirty-source hashes",
    )
    _require(
        source["dirty"] == bool(hashes),
        "Dirty source requires hashes; clean source must have none",
    )

    correctness = result["correctness"]
    _keys(correctness, "status method metrics", "correctness")
    _require(
        correctness["status"] in ("passed", "failed", "not_checked"),
        "Invalid correctness status",
    )
    _require(
        _text(correctness["method"]) and type(correctness["metrics"]) is dict,
        "Invalid correctness details",
    )
    if status == "ok":
        _require(correctness["status"] == "passed", "ok requires passed correctness")
    if correctness["status"] == "failed":
        _require(status == "failed", "Failed correctness requires failed status")

    timing = result["timing"]
    if timing is None:
        _require(status != "ok", "ok requires timing")
    else:
        _keys(
            timing,
            "first_call_s median_s iqr_s raw_block_s measurement_blocks calls_per_block warmup min_run_time_s",
            "timing",
        )
        for name in ("first_call_s", "median_s", "min_run_time_s"):
            _require(_number(timing[name]) and timing[name] > 0, f"Invalid {name}")
        _require(_number(timing["iqr_s"]), "Invalid iqr_s")
        for name in ("measurement_blocks", "calls_per_block"):
            _require(_integer(timing[name], 1), f"Invalid {name}")
        _require(_integer(timing["warmup"]), "Invalid warmup")
        raw = timing["raw_block_s"]
        _require(
            type(raw) is list
            and len(raw) == timing["measurement_blocks"]
            and all(_number(n) and n > 0 for n in raw),
            "Invalid raw blocks",
        )
        # Match native Measurement: float64 linear quantiles of per-call times.
        normalized = torch.tensor(
            sorted(block / timing["calls_per_block"] for block in raw),
            dtype=torch.float64,
            device="cpu",
        )
        median = normalized.quantile(0.5).item()
        iqr = (normalized.quantile(0.75) - normalized.quantile(0.25)).item()
        for name, expected in (("median_s", median), ("iqr_s", iqr)):
            _require(
                math.isclose(timing[name], expected, rel_tol=1e-9, abs_tol=1e-15),
                f"{name} is inconsistent with normalized raw blocks",
            )
    _keys(result["memory"], "peak_cuda_allocated_bytes", "memory")
    peak = result["memory"]["peak_cuda_allocated_bytes"]
    _require(
        peak is None or (_integer(peak) and env["device"].split(":")[0] == "cuda"),
        "CUDA allocated peak is null on other devices",
    )
    if status == "ok" and env["device"].split(":")[0] == "cuda":
        _require(peak is not None, "Successful CUDA timing requires allocated peak")


def write_result(result: dict, path: Path) -> None:
    """Validate and atomically replace a strict JSON file in an existing directory.

    Metadata is supplied explicitly; no environment, repository or device scan is
    performed. No speedup field exists in v1: comparisons are separate evidence.
    Validation/serialization finish before any file is created or replaced.
    """
    _validate(result)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

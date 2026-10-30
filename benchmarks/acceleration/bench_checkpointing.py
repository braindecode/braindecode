"""Fresh-process CPU peak-RSS and step-time checkpointing qualification.

Run the controller on a dedicated CPU Job; every measurement uses a new
interpreter. RSS includes framework/import/model/optimizer memory, not only
activations. Capacity means the largest integer batch under a declared RSS
budget, not a change in scientific batch/optimizer semantics.
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(checkpointing, batch):
    import torch

    from braindecode.models import EEGConformer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(129)
    model = EEGConformer(
        n_chans=22,
        n_times=8192,
        n_outputs=4,
        n_filters_time=64,
        num_heads=8,
        num_layers=6,
        activation_checkpointing=checkpointing,
    ).train()
    x = torch.randn(batch, 22, 8192)
    y = torch.arange(batch) % 4
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def step():
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        torch.nn.functional.cross_entropy(output, y).backward()
        optimizer.step()

    warmups = []
    stable = False
    for _ in range(8):
        start = time.perf_counter()
        step()
        warmups.append(time.perf_counter() - start)
        if len(warmups) >= 3:
            stable = (
                statistics.stdev(warmups[-3:]) / statistics.mean(warmups[-3:]) <= 0.05
            )
            if stable:
                break
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        step()
        samples.append(time.perf_counter() - start)
    print(
        json.dumps(
            {
                "checkpointing": checkpointing,
                "batch": batch,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "step_s": samples,
                "step_median_s": statistics.median(samples),
                "warmup_s": warmups,
                "warmup_stable": stable,
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
                "cpu_allocator_stats_available": hasattr(torch, "memory_stats"),
                "cpu": Path("/proc/cpuinfo")
                .read_text()
                .split("model name")[1]
                .splitlines()[0],
                "affinity": sorted(os.sched_getaffinity(0)),
            }
        ),
        flush=True,
    )


def controller(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "measurements.jsonl"

    def measure(on, batch, phase, pair=None):
        completed = subprocess.run(
            [sys.executable, __file__, "--worker", "--batch", str(batch)]
            + (["--checkpointing"] if on else []),
            check=False,
            text=True,
            capture_output=True,
            timeout=600,
        )
        with (outdir / "workers.log").open("a") as stream:
            stream.write(completed.stderr + completed.stdout)
        if completed.returncode:
            raise RuntimeError(
                f"worker failed: {completed.returncode}; see workers.log"
            )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        result.update(phase=phase, pair=pair)
        with raw.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        return result

    pairs = []
    for pair in range(5):
        cells = {}
        for on in [False, True] if pair % 2 == 0 else [True, False]:
            cells[on] = measure(on, 8, "equal_batch", pair)
        pairs.append(
            {
                "pair": pair,
                "memory_ratio": cells[True]["peak_rss_mib"]
                / cells[False]["peak_rss_mib"],
                "time_ratio": cells[True]["step_median_s"]
                / cells[False]["step_median_s"],
            }
        )
    if statistics.median(p["memory_ratio"] for p in pairs) > 0.9:
        for pair in range(5):
            measure(False, 8, "aa_left", pair)
            measure(False, 8, "aa_right", pair)
    budget = 4096
    capacity = []
    for on in (False, True):
        low, high = 0, 8
        checks = []
        while high <= 256:
            result = measure(on, high, "capacity")
            checks.append(result)
            if result["peak_rss_mib"] > budget:
                break
            low, high = high, high * 2
        if high > 256:
            capacity.append({"checkpointing": on, "max_batch": low, "censored": True})
            continue
        while high - low > 1:
            middle = (low + high) // 2
            result = measure(on, middle, "capacity")
            checks.append(result)
            if result["peak_rss_mib"] <= budget:
                low = middle
            else:
                high = middle
        capacity.append(
            {
                "checkpointing": on,
                "max_batch": low,
                "first_over_budget_batch": high,
                "censored": False,
            }
        )
    summary = {
        "pairs": pairs,
        "capacity": capacity,
        "budget_mib": budget,
        "keep_memory_gate": all(p["memory_ratio"] <= 0.95 for p in pairs),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.checkpointing, args.batch)
    else:
        controller(args.outdir)

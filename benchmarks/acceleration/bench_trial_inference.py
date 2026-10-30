"""Full-call CPU latency/RSS for materialized versus discarded trial predictions.

Run each mode in a fresh process (the parent CLI does this) so peak RSS includes
materialized output without contamination from the other mode. Synthetic windows
are created on demand; outputs and time targets are independently checked against
the known signal. RSS includes Python/Torch baseline and loader/model/output cost.
No claim of constant memory for materialization or a single unbounded trial.
"""

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmarks.acceleration._qualification import historical_only, require_linux_rss
from benchmarks.acceleration.bench_repeated_fits import _affinity, physical_cores
from braindecode.training.scoring import iter_trial_predictions, predict_trials


class SyntheticWindows(Dataset):
    def __init__(self, n_trials, variable):
        self.lengths = np.array(
            [4 + (i % 4 if variable else 0) for i in range(n_trials)]
        )
        self.ends = np.cumsum(self.lengths)
        self.reads = 0

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        self.reads += 1
        trial = int(np.searchsorted(self.ends, index, side="right"))
        window = index - (0 if trial == 0 else self.ends[trial - 1])
        start = int(window) * 256
        signal = np.broadcast_to(
            np.arange(start, start + 512, dtype=np.float32)[None] + trial,
            (8, 512),
        ).copy()
        return signal, signal.copy(), [int(window), start, start + 512]


class IdentityScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, X):
        return X * self.scale


def consume(module, dataset, mode, batch_size, verify=False):
    start_reads = dataset.reads
    result = (
        predict_trials(module, dataset, batch_size=batch_size)
        if mode == "materialized"
        else iter_trial_predictions(module, dataset, batch_size=batch_size)
    )
    pairs = zip(*result, strict=True) if mode == "materialized" else result
    count = output_bytes = 0
    for trial, (prediction, target) in enumerate(pairs):
        count += 1
        output_bytes += prediction.nbytes + target.nbytes
        if verify:
            expected = np.broadcast_to(
                np.arange(512 + (dataset.lengths[trial] - 1) * 256)[None] + trial,
                prediction.shape,
            )
            np.testing.assert_array_equal(prediction, expected)
            np.testing.assert_array_equal(target, expected)
            assert prediction.dtype == target.dtype == np.float32
    assert count == len(dataset.lengths)
    assert dataset.reads - start_reads == len(dataset)
    return output_bytes


def measure(args):
    require_linux_rss()
    torch.set_num_threads(1)
    module = IdentityScale().eval()
    dataset = SyntheticWindows(args.trials, args.variable)
    checkpoint = {key: value.clone() for key, value in module.state_dict().items()}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started = time.perf_counter()
    output_bytes = consume(module, dataset, args.mode, args.batch_size, verify=True)
    cold = time.perf_counter() - started
    warmup = []
    started = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        consume(module, dataset, args.mode, args.batch_size)
        warmup.append(time.perf_counter() - t0)
        elapsed = time.perf_counter() - started
        steady = len(warmup) >= 5 and max(warmup[-5:]) / min(warmup[-5:]) <= 1.03
        if (elapsed >= args.warmup_floor and steady) or elapsed >= args.warmup_cap:
            break
    warmup_s = time.perf_counter() - started
    durations = []
    for _ in range(args.samples):
        gc.collect()
        t0 = time.perf_counter()
        consume(module, dataset, args.mode, args.batch_size)
        durations.append(time.perf_counter() - t0)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, checkpoint[key], rtol=0, atol=0)
    clone = IdentityScale()
    clone.load_state_dict(module.state_dict(), strict=True)
    affinity = _affinity()
    cores = physical_cores(affinity)
    assert cores >= 2
    return {
        "mode": args.mode,
        "variable": args.variable,
        "batch_size": args.batch_size,
        "trials": args.trials,
        "windows": len(dataset),
        "output_bytes": output_bytes,
        "pre_call_process_high_water_rss_bytes": rss_before,
        "rss_scope": "Linux process high-water marks, not current RSS or incremental allocation",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cold_verified_s": cold,
        "warmup_s": warmup_s,
        "warmup": warmup,
        "steady": steady,
        "durations_s": durations,
        "median_s": float(np.median(durations)),
        "affinity": affinity,
        "physical_cores": cores,
        "torch_threads": torch.get_num_threads(),
        "host": os.uname().nodename,
        "correctness": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["materialized", "iterator"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--variable", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-floor", type=float, default=20)
    parser.add_argument("--warmup-cap", type=float, default=120)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode:
        record = historical_only(measure(args), "External launch/source/policy receipts required; not a new qualification tool.")
        args.output.write_text(json.dumps(record, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ["materialized", "iterator"]
    if args.run % 2:
        modes.reverse()
    for variable in (False, True):
        for batch_size in (1, 16, 64):
            for mode in modes:
                destination = args.output / f"{variable}-{batch_size}-{mode}.json"
                command = [
                    sys.executable,
                    __file__,
                    "--mode",
                    mode,
                    "--batch-size",
                    str(batch_size),
                    "--trials",
                    str(args.trials),
                    "--samples",
                    str(args.samples),
                    "--warmup-floor",
                    str(args.warmup_floor),
                    "--warmup-cap",
                    str(args.warmup_cap),
                    "--output",
                    str(destination),
                ]
                if variable:
                    command.append("--variable")
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

"""Historical synthetic long-trial latency diagnostic (not a memory qualification).

No EEG is stored: windows are generated lazily at 250 Hz, 8 output channels,
1024 samples, stride 512, including an end-aligned overlapping final window.
The reference is iter_trial_predictions (not all-trial materialization).
"""

import argparse
import gc
import hashlib
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
from braindecode.training.scoring import (
    iter_trial_prediction_chunks,
    iter_trial_predictions,
)


class LongTrial(Dataset):
    def __init__(self, hours):
        self.length = hours * 3600 * 250
        self.regular = (self.length - 1024) // 512 + 1
        self.extra = (self.length - 1024) % 512 != 0

    def __len__(self):
        return self.regular + self.extra

    def __getitem__(self, index):
        stop = min(1024 + index * 512, self.length)
        signal = np.broadcast_to(
            np.arange(stop - 1024, stop, dtype=np.float32)[None], (8, 1024)
        ).copy()
        return signal, signal.copy(), [index, stop - 1024, stop]


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, X):
        return X * self.scale


def consume(module, dataset, mode, verify=False):
    iterator = (
        iter_trial_predictions(module, dataset, batch_size=16)
        if mode == "trial"
        else iter_trial_prediction_chunks(
            module, dataset, windows_per_chunk=32, batch_size=16
        )
    )
    offset = 0
    digests = [hashlib.sha256() for _ in range(16)]
    try:
        for result in iterator:
            if mode == "trial":
                prediction, target = result
            else:
                assert result.trial_index == 0
                assert result.prediction_start == offset
                prediction, target = result.prediction, result.target
            assert prediction.shape == target.shape
            if verify:
                expected = np.arange(
                    offset, offset + prediction.shape[1], dtype=np.float32
                )
                for array in (prediction, target):
                    assert array.dtype == np.float32
                    for row in array:
                        assert row.tobytes() == expected.tobytes()
                for i, row in enumerate(np.concatenate((prediction, target), axis=0)):
                    digests[i].update(row.tobytes())
            offset += prediction.shape[1]
            del prediction, target, result
    finally:
        iterator.close()
    assert offset == dataset.length
    return [d.hexdigest() for d in digests] if verify else offset


def measure(args):
    require_linux_rss()
    torch.set_num_threads(1)
    module = Identity()
    dataset = LongTrial(args.hours)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t0 = time.perf_counter()
    digest = consume(module, dataset, args.mode, verify=True)
    cold = time.perf_counter() - t0
    warmup = []
    begin = time.perf_counter()
    while True:
        gc.collect()
        t0 = time.perf_counter()
        consume(module, dataset, args.mode)
        warmup.append(time.perf_counter() - t0)
        steady = len(warmup) >= 5 and max(warmup[-5:]) / min(warmup[-5:]) <= 1.05
        if (
            steady and time.perf_counter() - begin >= 3
        ) or time.perf_counter() - begin >= 20:
            break
    samples = []
    for _ in range(5):
        gc.collect()
        t0 = time.perf_counter()
        consume(module, dataset, args.mode)
        samples.append(time.perf_counter() - t0)
    return {
        "mode": args.mode,
        "hours": args.hours,
        "windows": len(dataset),
        "samples_s": samples,
        "median_s": float(np.median(samples)),
        "warmup_s": warmup,
        "steady": steady,
        "cold_verified_s": cold,
        "pre_call_process_high_water_rss_bytes": baseline_rss,
        "rss_scope": "Includes analytic correctness allocations; not consumption-only RSS",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "digest": digest,
        "length": dataset.length,
        "correctness": True,
        "host": os.uname().nodename,
        "threads": torch.get_num_threads(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["trial", "chunk"])
    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode:
        record = historical_only(measure(args), "Consumption-only memory gate is external historical evidence, not reproduced here.")
        args.output.write_text(json.dumps(record, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for hours in (1, 4):
        for pair in range(5):
            modes = ["trial", "chunk"] if pair % 2 == 0 else ["chunk", "trial"]
            records = []
            for mode in modes:
                destination = args.output / f"{hours}h-{pair}-{mode}.json"
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--mode",
                        mode,
                        "--hours",
                        str(hours),
                        "--output",
                        str(destination),
                    ],
                    check=True,
                )
                records.append(json.loads(destination.read_text()))
            assert records[0]["digest"] == records[1]["digest"]
            print(f"verified {hours}h pair {pair}", flush=True)


if __name__ == "__main__":
    main()

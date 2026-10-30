"""Opt-in CPU/Gloo evaluation recipe using real WindowsDataset and trial scoring.

No estimator distribution is enabled: preserve identity outside the estimator's
index-discarding loader, gather windows, restore chronology, then reconstruct.
Unpadded rank-strided shards avoid duplicate forwards. No collectives occur in
the forward loop, so empty/uneven ranks are supported. Trusted local ranks only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from benchmarks.acceleration.bench_distributed_contract import (
    _terminate,
    balanced_accuracy,
    gather_ordered_predictions,
)
from braindecode.datasets import WindowsDataset
from braindecode.training._trial_aggregation import iter_trial_preds

RTOL, ATOL = 1e-5, 1e-6


def make_dataset(n_trials=7, n_chans=4, width=32, scale=1):
    """Real preloaded MNE windows: uneven trials and end-aligned overlap."""
    rng = np.random.default_rng(19)
    signals, metadata, lengths = [], [], []
    for trial in range(n_trials):
        length = width + (trial % 5) * (width // 2) * scale + trial % 7
        lengths.append(length)
        signal = rng.standard_normal((n_chans, length)).astype(np.float32)
        starts = list(range(0, length - width + 1, width // 2))
        if starts[-1] != length - width:
            starts.append(length - width)
        for window, start in enumerate(starts):
            signals.append(signal[:, start : start + width])
            metadata.append((window, start, start + width, trial % 3))
    frame = pd.DataFrame(
        metadata,
        columns=["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial", "target"],
    )
    epochs = mne.EpochsArray(
        np.stack(signals),
        mne.create_info(n_chans, 128, "eeg"),
        metadata=frame,
        verbose=False,
    )
    return WindowsDataset(epochs), lengths


class IdentifiedWindows(Dataset):
    """Carry global source index alongside untouched WindowsDataset item."""

    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x, target, crop = self.windows[index]
        return x, target, crop, index


class DenseModel(nn.Module):
    """Batch-independent dense float32 convolution; no dropout or batch stats."""

    def __init__(self, n_chans=4, exact=False):
        super().__init__()
        self.exact = exact
        with torch.random.fork_rng():
            torch.manual_seed(91)
            self.layers = nn.Sequential(
                nn.Conv1d(n_chans, 32, 5, padding=2),
                nn.Tanh(),
                nn.Conv1d(32, 3, 1),
            )

    def forward(self, x):
        if self.exact:
            return x[:, :3] * 0.5 + 0.25
        return self.layers(x)


def shard_indices(size, rank, world_size, shuffle):
    order = torch.arange(size)
    if shuffle:
        order = order[torch.randperm(size, generator=torch.Generator().manual_seed(37))]
    return order[rank::world_size].tolist()


def reconstruct(preds, crops, *, chunk=None):
    return list(
        iter_trial_preds(
            preds.numpy(),
            crops[:, 0].tolist(),
            crops[:, 2].tolist(),
            windows_per_chunk=chunk,
        )
    )


def evaluate(model, dataset, rank=0, world_size=1, shuffle=False, batch_size=5):
    """Gather joint triples with exact identity/target/crop completeness."""
    indices = shard_indices(len(dataset), rank, world_size, shuffle)
    loader = DataLoader(
        IdentifiedWindows(dataset),
        batch_size=batch_size,
        sampler=indices,
        num_workers=0,
        drop_last=False,
    )
    ids, predictions, rows, batches = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, target, crop, identity in loader:
            ids.append(identity)
            predictions.append(model(x))
            rows.append(torch.stack([target, *crop], dim=1))
            batches.append(len(identity))
    width = dataset[0][0].shape[-1]
    local_ids = torch.cat(ids) if ids else torch.empty(0, dtype=torch.int64)
    local_preds = torch.cat(predictions) if ids else torch.empty(0, 3, width)
    local_rows = torch.cat(rows) if ids else torch.empty(0, 4, dtype=torch.int64)
    ordered_ids, preds, targets_crop = gather_ordered_predictions(
        local_ids,
        local_preds,
        local_rows,
        known_padding=0,
        expected_ids=torch.arange(len(dataset)),
    )
    # This identity is the chronological global dataset index, not trial-local
    # i_window_in_trial (which repeats). Reconstruct only AFTER ordering.
    trials = reconstruct(preds, targets_crop[:, 1:])
    first = targets_crop[:, 1] == 0
    targets = targets_crop[first, 0]
    scores = torch.from_numpy(np.stack([p.mean(axis=1) for p in trials]))
    counts = [None] * world_size
    info = {"ids": local_ids.tolist(), "batches": batches}
    if world_size > 1:
        dist.all_gather_object(counts, info)
    else:
        counts = [info]
    return {
        "ids": ordered_ids,
        "preds": preds,
        "rows": targets_crop,
        "trial_scores": scores,
        "trial_targets": targets,
        "metric": balanced_accuracy(scores, targets),
        "ranks": counts,
    }


def rank_main(rank, size, config, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/store",
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=60),
    )
    try:
        result = run_local(config, rank, size)
        if rank == 0:
            torch.save(result, f"{directory}/result.pt")
    finally:
        dist.destroy_process_group()


def run_local(config, rank=0, size=1):
    dataset, lengths = make_dataset(**config["dataset"])
    model = DenseModel(config["dataset"].get("n_chans", 4), config.get("exact", False))
    args = {
        "rank": rank,
        "world_size": size,
        "shuffle": config.get("shuffle", True),
        "batch_size": config.get("batch_size", 5),
    }
    warmups, steady = [], False
    for _ in range(config.get("warmups", 0)):
        start = time.perf_counter()
        evaluate(model, dataset, **args)
        warmups.append(time.perf_counter() - start)
        steady = len(warmups) >= 3 and max(warmups[-3:]) / min(warmups[-3:]) <= 1.05
        # Every rank must stop warming up together, otherwise a fast rank's
        # measured collective can pair with another rank's warmup collective.
        agreement = torch.tensor(int(steady))
        if size > 1:
            dist.all_reduce(agreement, op=dist.ReduceOp.MIN)
        steady = bool(agreement.item())
        if steady:
            break
    start = time.perf_counter()
    result = evaluate(model, dataset, **args)
    elapsed = time.perf_counter() - start
    result.update(
        lengths=lengths,
        evaluation_s=elapsed,
        warmup_s=warmups,
        steady=steady,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        threads=torch.get_num_threads(),
    )
    return result


def run(config, size=1, timeout=180):
    """Bounded local spawn; all children reaped even on failed collectives."""
    torch.set_num_threads(1)
    if size == 1:
        return run_local(config)
    with tempfile.TemporaryDirectory(prefix="bd-eval-") as directory:
        context = mp.start_processes(
            rank_main,
            args=(size, config, directory),
            nprocs=size,
            join=False,
            start_method="spawn",
        )
        try:
            deadline = time.monotonic() + timeout
            while not context.join(timeout=0.5):
                if time.monotonic() > deadline:
                    raise TimeoutError("evaluation rank deadline exceeded")
            return torch.load(f"{directory}/result.pt", weights_only=False)
        finally:
            _terminate(context.processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks:
        config = {
            "dataset": {"n_trials": 53, "n_chans": 32, "width": 512, "scale": 12},
            "batch_size": 32,
            "warmups": 10,
        }
        result = run(config, args.ranks)
        record = {
            k: v
            for k, v in result.items()
            if k not in {"ids", "preds", "rows", "trial_scores", "trial_targets"}
        }
        record.update(
            ranks=args.ranks,
            windows=len(result["ids"]),
            windows_per_s=len(result["ids"]) / result["evaluation_s"],
            host=os.uname().nodename,
        )
        args.output.write_text(json.dumps(record, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for size in (2, 4):
        for pair in range(5):
            for ranks in [1, size] if pair % 2 == 0 else [size, 1]:
                destination = args.output / f"{size}-{pair}-{ranks}.json"
                start = time.perf_counter()
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        __name__
                        if __name__ != "__main__"
                        else "benchmarks.acceleration.bench_distributed_windows",
                        "--ranks",
                        str(ranks),
                        "--output",
                        str(destination),
                    ],
                    check=True,
                    timeout=300,
                )
                record = json.loads(destination.read_text())
                record["including_startup_s"] = time.perf_counter() - start
                destination.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

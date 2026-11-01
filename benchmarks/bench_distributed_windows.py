"""CPU/Gloo distributed evaluation of real windows, with trial reconstruction.

A recipe, not an estimator backend: ``EEGClassifier`` and ``EEGRegressor`` still
predict in one process. Run from the repository root, one fresh process per
measurement, for example::

    python -m benchmarks.bench_distributed_windows --ranks 4 --output 4.json

Each rank forwards an unpadded rank-strided shard of the global window indices,
so every window is forwarded exactly once and ranks may be empty. Predictions,
targets and crop indices are gathered together with the global window index;
the gather fails unless every index arrives exactly once. Only then are the
chronologically ordered windows reconstructed into trials, and the metric is
computed on all trials, never averaged across ranks.

Scope: trusted local CPU processes, preloaded fixed-width windows, scalar
classification targets and batch-independent eval models. Every rank holds the
full dataset and the full gathered predictions. ``evaluation_s`` excludes
process launch, imports and warmup, which can dominate a one-shot run.
"""

import argparse
import json
import math
import os
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

from braindecode.datasets import WindowsDataset
from braindecode.training.scoring import trial_preds_from_window_preds

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
    """Return the global window index next to the untouched WindowsDataset item.

    Trial-local ``i_window_in_trial`` repeats across trials, so it cannot
    identify a window.
    """

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


def _all_gather_rows(tensor, counts):
    """Gather variable-length row blocks: pad to the longest, then trim."""
    longest = max(counts)
    if longest == 0:  # every rank is empty: skip the collective on all ranks
        return tensor
    padded = tensor.new_zeros((longest, *tensor.shape[1:]))
    padded[: len(tensor)] = tensor
    buffers = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(buffers, padded)
    return torch.cat([buffer[:n] for buffer, n in zip(buffers, counts)])


def gather_by_identity(ids, preds, rows, counts, n_windows):
    """Gather (id, prediction, target/crop row) triples and order them by id.

    Raises ``ValueError`` unless the ids are exactly ``0..n_windows - 1``, so a
    missing, repeated or foreign window can never be merged silently.
    """
    if len(counts) > 1:
        ids, preds, rows = (_all_gather_rows(t, counts) for t in (ids, preds, rows))
    order = torch.argsort(ids)
    ids, preds, rows = ids[order], preds[order], rows[order]
    if not torch.equal(ids, torch.arange(n_windows)):
        raise ValueError("gathered window ids are not every window exactly once")
    return ids, preds, rows


def balanced_accuracy(scores, targets):
    """Mean recall over the classes present in ``targets``."""
    predicted = scores.argmax(dim=1)
    recalls = [
        (predicted[targets == c] == c).double().mean().item()
        for c in torch.unique(targets).tolist()
    ]
    return math.fsum(recalls) / len(recalls)


def evaluate(model, dataset, rank=0, world_size=1, shuffle=False, batch_size=5):
    """Forward this rank's shard, gather all ranks, reconstruct and score trials."""
    indices = shard_indices(len(dataset), rank, world_size, shuffle)
    loader = DataLoader(
        IdentifiedWindows(dataset), batch_size=batch_size, sampler=indices
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
    info = {"n_windows": len(local_ids), "batches": batches}
    ranks = [info]
    if world_size > 1:
        ranks = [None] * world_size
        dist.all_gather_object(ranks, info)
    ordered_ids, preds, targets_crop = gather_by_identity(
        local_ids,
        local_preds,
        local_rows,
        [r["n_windows"] for r in ranks],
        len(dataset),
    )
    # The id is the chronological global dataset index, so reconstruct only
    # AFTER ordering by it.
    trials = trial_preds_from_window_preds(
        preds.numpy(), targets_crop[:, 1].tolist(), targets_crop[:, 3].tolist()
    )
    targets = targets_crop[targets_crop[:, 1] == 0, 0]
    scores = torch.from_numpy(np.stack([p.mean(axis=1) for p in trials]))
    return {
        "ids": ordered_ids,
        "preds": preds,
        "rows": targets_crop,
        "trial_scores": scores,
        "trial_targets": targets,
        "metric": balanced_accuracy(scores, targets),
        "ranks": ranks,
    }


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
    result.update(
        lengths=lengths,
        evaluation_s=time.perf_counter() - start,
        warmup_s=warmups,
        steady=steady,
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        threads=torch.get_num_threads(),
    )
    return result


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


def run(config, size=1):
    """Evaluate in one process, or spawn ``size`` local Gloo ranks."""
    torch.set_num_threads(1)
    if size == 1:
        return run_local(config)
    # ponytail: no parent deadline; the 60 s process-group timeout bounds a
    # stuck collective, and mp.spawn terminates the other ranks on a failure.
    with tempfile.TemporaryDirectory(prefix="bd-eval-") as directory:
        mp.spawn(rank_main, args=(size, config, directory), nprocs=size)
        return torch.load(f"{directory}/result.pt", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()

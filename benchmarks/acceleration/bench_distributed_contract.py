"""Two-rank CPU/Gloo evaluation identity contract, private lab only.

Example (results stay outside the checkout)::

    python -m benchmarks.acceleration.bench_distributed_contract \
        --n-records 7 --batch-size 3 --repeats 3 --work-dir /tmp/dc \
        --output /tmp/dc/record.json

This is a native PyTorch reference recipe, not a Braindecode feature. It adds no
distributed estimator flag and does not claim that ``EEGClassifier`` or
``AccelerateMixin`` evaluate correctly across ranks (``ThrowAwayIndexLoader``
drops the identities this contract needs). It fixes the accounting that a later
distributed evaluation must satisfy:

- each rank evaluates its ``DistributedSampler`` shard (``drop_last=False``,
  which pads with repeated records so every rank runs the same number of
  steps) or an explicit uneven shard, with partial last batches and unequal
  trial lengths;
- :func:`gather_ordered_predictions` gathers (identity, prediction, target)
  triples jointly from every rank and orders them by source identity. The
  caller must declare the sampler padding (``known_padding``): only rows in
  the ``DistributedSampler`` padding positions that repeat the record the
  sampler pads with are removed. Any other repeated identity is rejected,
  whether its rows agree (a double count, or distinct records sharing an id)
  or not (a conflict); with ``expected_ids`` every real record must appear
  exactly once;
- the metric (balanced accuracy, nonlinear in shard composition) is computed
  once on the gathered global predictions; rank scores are recorded only to
  show that their mean differs;
- ranks are spawned children with an explicit process-group timeout and a
  parent deadline. Every rank takes part in every collective, validation
  errors are raised on all ranks alike, a failing rank records its error
  before tearing down its process group, and a failing or hung rank surfaces
  as :class:`DistributedEvaluationError` after every child is terminated and
  reaped (bounded joins). Only rank zero writes the run result.

The reference model sums integer-valued float64 signals with integer weights,
so its logits are exact and independent of padding and batch composition:
bit-equality between one-process and two-rank results tests the accounting,
not floating-point reduction order. Timings measure contract overhead (spawn,
rendezvous, evaluation, gather) on CPU and are not accelerator scaling claims.
The record keeps per-step global batch sizes, optimizer steps (zero here) and
sampler metadata for later native GPU/HPU qualification.

The output is this recipe's own record (``schema``), not the plan-00 v1
envelope of ``_common``: as for the repeated-fit recipe, this is a one-shot
contract study rather than repeated-call timing, so the record carries an
explicit ``status`` (``ok``, ``failed``, or ``unsupported`` when Gloo is
missing, CLI exit code 3), named ``checks`` and thread settings, and reports
no ratios. The ``file://`` rendezvous needs a filesystem with working
``fcntl`` locks for the run directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

DEFAULT_CONFIG = {
    "n_records": 7,
    "n_chans": 3,
    "min_len": 5,
    "max_len": 17,
    "n_classes": 3,
    "seed": 0,
    "batch_size": 3,
    "shuffle": False,
    "shards": None,
}
TIMING_CLAIM = "contract overhead only; not a scaling claim"
SCHEMA = "distributed-evaluation-contract/1"
SCHEMA_NOTE = (
    "recipe-specific contract record, not the plan-00 v1 envelope: a one-shot "
    "contract study with explicit status and checks, not repeated-call timing "
    "(same ruling as the repeated-fit recipe)"
)
EXIT_UNSUPPORTED = 3
RANK_THREADS = 1
FAULT_KINDS = ("raise", "exit", "hang")
FAULT_WHEN = ("before_init", "after_init")


def gloo_available() -> bool:
    """Whether torch.distributed with the Gloo backend can be used here."""
    return dist.is_available() and dist.is_gloo_available()


class DistributedEvaluationError(RuntimeError):
    """A spawned rank failed, hung or the parent deadline expired.

    ``child_pids`` are the spawned ranks (all terminated and reaped when this
    is raised, unless the message lists ranks still alive after SIGKILL).
    ``failed_rank`` is the first rank the parent saw exit, which can be a peer
    that only reacted to the root cause. ``rank_errors`` maps each rank to the
    error it recorded before tearing down its process group, and
    ``first_error_rank`` is the rank that recorded the earliest one (the root
    cause when a rank raised).
    """

    def __init__(
        self,
        message,
        *,
        child_pids=(),
        failed_rank=None,
        elapsed_s=None,
        rank_errors=None,
        first_error_rank=None,
    ):
        super().__init__(message)
        self.child_pids = tuple(child_pids)
        self.failed_rank = failed_rank
        self.elapsed_s = elapsed_s
        self.rank_errors = dict(rank_errors or {})
        self.first_error_rank = first_error_rank


class SyntheticRecords(Dataset):
    """Records with unequal lengths and shuffled, non-contiguous source IDs.

    Signals are small integers stored as float64. Item: (signal, target, id).
    """

    def __init__(self, config: dict):
        n = config["n_records"]
        gen = torch.Generator().manual_seed(config["seed"])
        self.source_ids = torch.randperm(4 * n + 1, generator=gen)[:n] + 100
        self.lengths = torch.randint(
            config["min_len"], config["max_len"] + 1, (n,), generator=gen
        )
        self.signals = [
            torch.randint(-4, 5, (config["n_chans"], int(length)), generator=gen).to(
                torch.float64
            )
            for length in self.lengths
        ]
        self.targets = torch.randint(0, config["n_classes"], (n,), generator=gen)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index], self.targets[index], self.source_ids[index]


def collate_padded(batch):
    """Zero-pad unequal trials to the longest one; keep lengths and IDs."""
    signals, targets, ids = zip(*batch)
    lengths = torch.tensor([signal.shape[1] for signal in signals])
    x = signals[0].new_zeros(len(signals), signals[0].shape[0], int(lengths.max()))
    for i, signal in enumerate(signals):
        x[i, :, : signal.shape[1]] = signal
    return x, lengths, torch.stack(targets), torch.stack(ids)


class ExactMeanLogits(nn.Module):
    """Logits = (integer weights applied to per-channel sums) / trial length.

    Integer sums are exact in float64 and the division is correctly rounded,
    so outputs do not depend on padding, batch size or reduction order.
    """

    def __init__(self, n_chans: int, n_classes: int, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + 1)
        weight = torch.randint(-2, 3, (n_chans, n_classes), generator=gen)
        self.register_buffer("weight", weight.to(torch.float64))

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        return (x.sum(dim=2) @ self.weight) / lengths.to(x.dtype).unsqueeze(1)


def balanced_accuracy(preds: Tensor, targets: Tensor) -> float | None:
    """Mean recall over classes present in ``targets``; None without records."""
    if targets.numel() == 0:
        return None
    predicted = preds.argmax(dim=1)
    recalls = [
        (predicted[targets == c] == c).double().mean().item()
        for c in torch.unique(targets).tolist()
    ]
    return math.fsum(recalls) / len(recalls)


def _describe_local(ids: Tensor, preds: Tensor, targets: Tensor) -> dict:
    error = None
    if ids.ndim != 1 or ids.dtype.is_floating_point or ids.dtype == torch.bool:
        error = f"ids must be a 1-D integer tensor, got {ids.dtype} {tuple(ids.shape)}"
    elif preds.ndim == 0 or targets.ndim == 0:
        error = "predictions and targets need a leading record dimension"
    elif preds.shape[0] != ids.shape[0] or targets.shape[0] != ids.shape[0]:
        error = (
            f"rows differ: {ids.shape[0]} ids, {preds.shape[0]} predictions, "
            f"{targets.shape[0]} targets"
        )
    return {
        "n": int(ids.shape[0]) if ids.ndim else 0,
        "error": error,
        "signature": (
            str(ids.dtype),
            tuple(preds.shape[1:]),
            str(preds.dtype),
            tuple(targets.shape[1:]),
            str(targets.dtype),
        ),
    }


def _all_gather_object(obj, world_size: int) -> list:
    if world_size == 1:
        return [obj]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def _all_gather_rows(tensor: Tensor, counts: list[int]) -> Tensor:
    """Gather variable-length row blocks: pad to the longest, then trim."""
    if tensor.dtype == torch.bool:  # Gloo does not reduce/gather bool
        return _all_gather_rows(tensor.to(torch.uint8), counts).bool()
    longest = max(counts)
    if longest == 0:  # every rank is empty: skip the collective on all ranks
        return tensor.new_empty((0, *tensor.shape[1:]))
    padded = tensor.new_zeros((longest, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    buffers = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(buffers, padded)
    return torch.cat([buffer[:n] for buffer, n in zip(buffers, counts)])


def _rows_equal(a: Tensor, b: Tensor) -> Tensor:
    equal = a == b
    if a.is_floating_point() or a.is_complex():
        equal = equal | (torch.isnan(a) & torch.isnan(b))
    return equal.reshape(a.shape[0], -1).all(dim=1)


def _padding_positions(counts: list[int], known_padding: int) -> list[tuple[int, int]]:
    """(padded row, row it repeats) in the gathered rank-major row order.

    ``DistributedSampler(drop_last=False)`` gives every rank the same count
    ``m``, extends the ``n`` real sampler positions with positions
    ``n, ..., m * world_size - 1`` that repeat positions ``0, 1, ...``
    cyclically, and hands sampler position ``g`` to rank ``g % world_size``
    at local index ``g // world_size``. Raises ``ValueError`` if the declared
    padding cannot come from that layout.
    """
    if known_padding == 0:
        return []
    world_size, local = len(counts), counts[0]
    if known_padding >= world_size or local == 0 or len(set(counts)) != 1:
        raise ValueError(
            f"declared sampler padding {known_padding} does not match the "
            f"DistributedSampler layout: per-rank counts {counts} must be equal "
            f"and non-zero, and padding below the world size {world_size}"
        )
    n_real = local * world_size - known_padding

    def row(position: int) -> int:
        return (position % world_size) * local + position // world_size

    return [(row(n_real + j), row(j % n_real)) for j in range(known_padding)]


def _deduplicate_and_order(ids, preds, targets, counts, known_padding, expected_ids):
    """Reject conflicts, drop declared padding rows, reject other repeats."""
    if ids.shape[0]:
        order = torch.argsort(ids, stable=True)
        s_ids, s_preds, s_targets = ids[order], preds[order], targets[order]
        _, inverse, run_lengths = torch.unique_consecutive(
            s_ids, return_inverse=True, return_counts=True
        )
        first = (torch.cumsum(run_lengths, dim=0) - run_lengths)[inverse]
        same = _rows_equal(s_preds, s_preds[first]) & _rows_equal(
            s_targets, s_targets[first]
        )
        if not bool(same.all()):
            bad = torch.unique(s_ids[~same]).tolist()
            raise ValueError(
                f"conflicting duplicate identity: source ids {bad[:10]} were "
                "reported more than once with different predictions or "
                "targets; refusing to drop either"
            )
    keep = torch.ones(ids.shape[0], dtype=torch.bool)
    for padded, source in _padding_positions(counts, known_padding):
        if ids[padded] != ids[source]:
            raise ValueError(
                "declared sampler padding does not match the DistributedSampler "
                f"layout: row {padded} (id {int(ids[padded])}) should repeat "
                f"row {source} (id {int(ids[source])})"
            )
        keep[padded] = False
    ids, preds, targets = ids[keep], preds[keep], targets[keep]
    unique_ids, id_counts = torch.unique(ids, return_counts=True)
    if bool((id_counts > 1).any()):
        repeated = unique_ids[id_counts > 1].tolist()
        raise ValueError(
            "repeated identity not declared as sampler padding: source ids "
            f"{repeated[:10]} were reported more than once with identical rows "
            "(a record evaluated more than once, or distinct records sharing "
            "an id); refusing to merge them"
        )
    order = torch.argsort(ids)
    ids, preds, targets = ids[order], preds[order], targets[order]
    if expected_ids is not None:
        expected = torch.as_tensor(expected_ids).flatten().to(ids.dtype)
        if torch.unique(expected).numel() != expected.numel():
            raise ValueError("expected_ids must be unique")
        if not torch.equal(ids, torch.sort(expected).values):
            got, want = set(ids.tolist()), set(expected.tolist())
            raise ValueError(
                "gathered ids are not every expected record exactly once: "
                f"missing {sorted(want - got)[:10]}, "
                f"unexpected {sorted(got - want)[:10]}"
            )
    return ids, preds, targets


def gather_ordered_predictions(
    local_ids: Tensor,
    local_preds: Tensor,
    local_targets: Tensor,
    *,
    known_padding: int,
    expected_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather (id, prediction, target) triples from all ranks, ordered by id.

    Collective: every rank of the default process group must call it with the
    same ``known_padding`` and ``expected_ids`` and with possibly empty local
    tensors whose trailing shapes and dtypes agree across ranks. Without an
    initialized process group it checks and orders one process's triples
    (world size 1, so ``known_padding`` must be 0). Rows stay aligned because
    the three tensors are gathered with the same per-rank row counts.

    ``known_padding`` is required: the number of rows that
    ``DistributedSampler(drop_last=False)`` appended (``total_size -
    len(dataset)``), or 0 for unpadded shards. Exactly those rows are removed,
    and only if they sit in the sampler's padding positions and repeat the
    record the sampler pads with. Every other repeated identity raises
    ``ValueError`` on every rank: differing rows are a conflict, identical rows
    a double count or distinct records sharing an id, and they are never
    merged. With ``expected_ids`` the result must be exactly those ids. Rows
    are compared exactly (NaN equals NaN); real floating-point models will need
    a tolerance policy. Returns CPU tensors sorted by source id.
    """
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    local_ids, local_preds, local_targets = (
        t.detach().cpu() for t in (local_ids, local_preds, local_targets)
    )
    meta = _describe_local(local_ids, local_preds, local_targets)
    if type(known_padding) is not int or known_padding < 0:
        meta["error"] = meta["error"] or (
            f"known_padding must be a non-negative integer, got {known_padding!r}"
        )
    meta["known_padding"] = repr(known_padding)
    metas = _all_gather_object(meta, world_size)
    errors = [f"rank {r}: {m['error']}" for r, m in enumerate(metas) if m["error"]]
    if errors:
        raise ValueError("invalid local triples: " + "; ".join(errors))
    if len({m["signature"] for m in metas}) != 1:
        raise ValueError(
            "ranks disagree on id/prediction/target trailing shapes or dtypes: "
            f"{[m['signature'] for m in metas]}"
        )
    if len({m["known_padding"] for m in metas}) != 1:
        raise ValueError(
            f"ranks disagree on known_padding: {[m['known_padding'] for m in metas]}"
        )
    counts = [m["n"] for m in metas]
    if distributed:
        ids = _all_gather_rows(local_ids, counts)
        preds = _all_gather_rows(local_preds, counts)
        targets = _all_gather_rows(local_targets, counts)
    else:
        ids, preds, targets = local_ids, local_preds, local_targets
    return _deduplicate_and_order(
        ids, preds, targets, counts, known_padding, expected_ids
    )


def _sampler(dataset, config: dict, rank: int, world_size: int):
    """Return (sampler, name, known padding) for this rank."""
    if config.get("shards") is not None:
        if len(config["shards"]) != world_size:
            raise ValueError(f"need {world_size} shards, got {len(config['shards'])}")
        return list(config["shards"][rank]), "explicit shards", 0
    if world_size == 1:
        return None, "sequential", 0
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config["shuffle"],
        seed=config["seed"],
        drop_last=False,
    )
    return sampler, "DistributedSampler", sampler.total_size - len(dataset)


def evaluate_local(model, dataset, sampler, batch_size: int, n_classes: int):
    """Evaluate one rank's shard; return local ids, preds, targets, batch sizes."""
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_padded
    )
    ids, preds, targets, batch_sizes = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, lengths, y, batch_ids in loader:
            ids.append(batch_ids)
            preds.append(model(x, lengths))
            targets.append(y)
            batch_sizes.append(len(batch_ids))
    if not ids:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, n_classes, dtype=torch.float64),
            torch.empty(0, dtype=torch.int64),
            [],
        )
    return torch.cat(ids), torch.cat(preds), torch.cat(targets), batch_sizes


def _evaluate_task(config: dict, rank: int, world_size: int) -> dict:
    dataset = SyntheticRecords(config)
    model = ExactMeanLogits(config["n_chans"], config["n_classes"], config["seed"])
    sampler, sampler_name, padding = _sampler(dataset, config, rank, world_size)
    ids, preds, targets, batch_sizes = evaluate_local(
        model, dataset, sampler, config["batch_size"], config["n_classes"]
    )
    # Same gathered data on every rank, so every rank raises alike.
    g_ids, g_preds, g_targets = gather_ordered_predictions(
        ids, preds, targets, known_padding=padding, expected_ids=dataset.source_ids
    )
    local = {
        "count": int(ids.shape[0]),
        "ids": ids.tolist(),
        "batch_sizes": batch_sizes,
        "metric": balanced_accuracy(preds, targets),
    }
    ranks = _all_gather_object(local, world_size)
    steps = [r["batch_sizes"] for r in ranks]
    contributions = sum(r["count"] for r in ranks)
    return {
        "ids": g_ids,
        "preds": g_preds,
        "targets": g_targets,
        "metric": balanced_accuracy(g_preds, g_targets),
        "accounting": {
            "world_size": world_size,
            "backend": dist.get_backend() if world_size > 1 else None,
            "sampler": sampler_name,
            "shuffle": bool(config["shuffle"]),
            "drop_last": False,
            "batch_size": config["batch_size"],
            "n_records": len(dataset),
            "local_counts": [r["count"] for r in ranks],
            "raw_ids": [r["ids"] for r in ranks],
            "batch_sizes": steps,
            "steps_per_rank": [len(s) for s in steps],
            "global_batch_sizes": [
                sum(step) for step in itertools.zip_longest(*steps, fillvalue=0)
            ],
            "optimizer_steps": 0,  # evaluation only
            "contributions": contributions,
            "expected_padding": padding,
            "padding_removed": contributions - int(g_ids.shape[0]),
            "rank_metrics": [r["metric"] for r in ranks],
        },
    }


def _gather_task(payload, rank: int, world_size: int) -> dict:
    ids, preds, targets = payload["triples"][rank]
    expected = payload.get("expected_ids")
    g_ids, g_preds, g_targets = gather_ordered_predictions(
        ids,
        preds,
        targets,
        known_padding=payload["known_padding"],
        expected_ids=None if expected is None else torch.as_tensor(expected),
    )
    total = sum(len(triple[0]) for triple in payload["triples"])
    return {
        "ids": g_ids,
        "preds": g_preds,
        "targets": g_targets,
        "padding_removed": total - int(g_ids.shape[0]),
    }


TASKS = {"evaluate": _evaluate_task, "gather": _gather_task}


def evaluate_one_process(config: dict) -> dict:
    """Reference: the same recipe in one process (world size 1, no padding)."""
    return _evaluate_task(dict(config, shards=None), 0, 1)


def reference_predictions(config: dict) -> tuple[Tensor, Tensor]:
    """Per-record (batch size 1) predictions and targets in source-id order."""
    dataset = SyntheticRecords(config)
    model = ExactMeanLogits(config["n_chans"], config["n_classes"], config["seed"])
    order = torch.argsort(dataset.source_ids).tolist()
    preds = torch.empty(0, config["n_classes"], dtype=torch.float64)
    with torch.no_grad():
        rows = [model(*collate_padded([dataset[i]])[:2]) for i in order]
    if rows:
        preds = torch.cat(rows)
    return preds, dataset.targets[order]


def _inject_fault(rank: int, fault: dict | None, when: str) -> None:
    """Contract-test hook: make one rank (or all) raise, exit or hang."""
    if not fault or fault["rank"] not in (rank, "all"):
        return
    if fault.get("when", "after_init") != when:
        return
    if fault["kind"] == "raise":
        raise RuntimeError(f"injected failure on rank {rank}")
    if fault["kind"] == "exit":
        os._exit(17)
    time.sleep(3600)  # "hang": bounded only by the timeouts under test


def _validate_fault(fault: dict | None) -> None:
    if fault is None:
        return
    if (
        not isinstance(fault, dict)
        or "rank" not in fault
        or fault.get("kind") not in FAULT_KINDS
        or fault.get("when", "after_init") not in FAULT_WHEN
    ):
        raise ValueError(
            f"invalid fault {fault!r}: needs a rank, kind in {FAULT_KINDS} and "
            f"when in {FAULT_WHEN}"
        )


def _record_error(run_dir: Path, rank: int, exc: BaseException) -> None:
    """Write this rank's error atomically (time-stamped) for the parent."""
    error = {"time": time.time(), "error": f"{type(exc).__name__}: {exc}"}
    partial = run_dir / f"rank-{rank}.err.partial"
    partial.write_text(json.dumps(error))
    partial.rename(run_dir / f"rank-{rank}.err")


def _rank_main(rank, world_size, task, payload, run_dir, pg_timeout_s, fault):
    run_dir = Path(run_dir)
    torch.set_num_threads(RANK_THREADS)
    try:
        _inject_fault(rank, fault, "before_init")
        dist.init_process_group(
            "gloo",
            init_method=(run_dir / "rendezvous").as_uri(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=pg_timeout_s),
        )
    except Exception as exc:
        _record_error(run_dir, rank, exc)
        raise
    try:
        _inject_fault(rank, fault, "after_init")
        result = TASKS[task](payload, rank, world_size)
        if rank == 0:  # only rank zero writes the run result
            partial = run_dir / "result.pt.partial"
            torch.save(result, partial)
            partial.rename(run_dir / "result.pt")
    except Exception as exc:
        # Record the root cause before teardown: destroying the process group
        # makes the peers fail too, and the parent may then terminate us.
        _record_error(run_dir, rank, exc)
        raise
    finally:
        dist.destroy_process_group()


_NO_RANK_ERROR = (
    "none recorded (a rank exited or was killed without raising, or failed "
    "before rank code started; see the ranks' stderr)"
)


def _terminate(processes, grace_s: float = 5.0) -> list[int]:
    """SIGTERM, then SIGKILL after ``grace_s``; return pids still not reaped."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    unreaped = []
    for process in processes:
        process.join(timeout=grace_s)
        if process.is_alive():
            process.kill()
            process.join(timeout=grace_s)
            if process.is_alive():
                unreaped.append(process.pid)
    return unreaped


def run_ranks(
    task: str,
    payload,
    *,
    workdir,
    world_size: int = 2,
    timeout_s: float = 180.0,
    pg_timeout_s: float = 60.0,
    fault: dict | None = None,
) -> dict:
    """Run ``task`` on ``world_size`` spawned Gloo ranks; return rank 0's result.

    Rendezvous uses a ``file://`` store in a fresh private directory under
    ``workdir``, removed afterwards. The process group times out collectives
    after ``pg_timeout_s``; the parent terminates every rank after
    ``timeout_s``. Any failure raises :class:`DistributedEvaluationError`
    after all children are terminated and reaped.
    """
    if not gloo_available():
        raise RuntimeError("torch.distributed Gloo backend unavailable")
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    _validate_fault(fault)
    run_dir = Path(tempfile.mkdtemp(prefix="ranks-", dir=workdir))
    start = time.monotonic()
    failure, failed_rank, processes, report, unreaped = None, None, [], "", []
    try:
        try:
            context = mp.start_processes(
                _rank_main,
                args=(world_size, task, payload, str(run_dir), pg_timeout_s, fault),
                nprocs=world_size,
                join=False,
                start_method="spawn",
            )
            processes = context.processes
            while True:
                remaining = start + timeout_s - time.monotonic()
                if remaining <= 0:
                    failure = f"exceeded the parent deadline of {timeout_s:.1f} s"
                    break
                try:
                    if context.join(timeout=min(1.0, remaining)):
                        break
                except (mp.ProcessRaisedException, mp.ProcessExitedException) as exc:
                    failed_rank = exc.error_index
                    failure = f"failed (rank {failed_rank} first)"
                    report = str(exc).strip()[-2000:]
                    break
        finally:
            unreaped = _terminate(processes)
        if failure is None:
            return torch.load(run_dir / "result.pt", weights_only=True)
        records = {
            int(path.name.split("-")[1].split(".")[0]): json.loads(path.read_text())
            for path in sorted(run_dir.glob("rank-*.err"))
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    elapsed = time.monotonic() - start
    exit_codes = {rank: p.exitcode for rank, p in enumerate(processes)}
    rank_errors = {rank: records[rank]["error"] for rank in sorted(records)}
    first_error_rank = (
        min(records, key=lambda rank: records[rank]["time"]) if records else None
    )
    if first_error_rank is None:
        errors = f"rank errors {_NO_RANK_ERROR}"
    else:
        errors = (
            f"first error on rank {first_error_rank}: "
            f"{rank_errors[first_error_rank]}; rank errors {rank_errors}"
        )
    raise DistributedEvaluationError(
        f"{world_size}-rank {task!r} run {failure} after {elapsed:.1f} s "
        f"(process-group timeout {pg_timeout_s:.1f} s, parent deadline "
        f"{timeout_s:.1f} s); exit codes {exit_codes}; {errors}"
        + (f"; ranks still alive after SIGKILL: pids {unreaped}" if unreaped else "")
        + (f"; first failure report: {report}" if report else ""),
        child_pids=[p.pid for p in processes],
        failed_rank=failed_rank,
        elapsed_s=elapsed,
        rank_errors=rank_errors,
        first_error_rank=first_error_rank,
    )


def _real_contributions(raw_ids: list[list[int]], known_padding: int) -> list[int]:
    """Raw per-rank contributions minus the rows in sampler padding positions."""
    counts = [len(rank_ids) for rank_ids in raw_ids]
    padded = {row for row, _ in _padding_positions(counts, known_padding)}
    flat = [i for rank_ids in raw_ids for i in rank_ids]
    return [i for row, i in enumerate(flat) if row not in padded]


def compare_results(config: dict, single: dict, multi: dict) -> dict:
    """Named contract checks between the one-process and two-rank results."""
    dataset = SyntheticRecords(config)
    expected_ids = torch.sort(dataset.source_ids).values
    acct = multi["accounting"]
    real = _real_contributions(acct["raw_ids"], acct["expected_padding"])
    return {
        "ids_equal_expected": torch.equal(multi["ids"], expected_ids),
        "ids_equal_one_process": torch.equal(multi["ids"], single["ids"]),
        "preds_equal_one_process": torch.equal(multi["preds"], single["preds"]),
        "targets_equal_one_process": torch.equal(multi["targets"], single["targets"]),
        "raw_contributions_cover_each_record_once": sorted(real)
        == expected_ids.tolist(),
        "padding_removed_is_known_padding": (
            acct["padding_removed"] == acct["expected_padding"]
        ),
        "global_metric_equal_one_process": multi["metric"] == single["metric"],
    }


def _environment() -> dict:
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gloo_available": gloo_available(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _plain(result: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, Tensor) else value
        for key, value in result.items()
    }


def _threads() -> dict:
    return {
        "torch_threads_per_rank": RANK_THREADS,
        "one_process_torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def run_contract(
    config: dict,
    *,
    workdir,
    repeats: int = 3,
    timeout_s: float = 180.0,
    pg_timeout_s: float = 60.0,
) -> dict:
    """Compare one-process and two-rank evaluation; time both (overhead only).

    Status is ``ok`` only if every contract check passes on every repeat. The
    two-rank time includes spawning, rendezvous, evaluation and gather.
    Without the Gloo backend nothing runs and the status is ``unsupported``.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    base = {"schema": SCHEMA, "schema_note": SCHEMA_NOTE, "config": config}
    if not gloo_available():
        return {
            **base,
            "status": "unsupported",
            "reason": "torch.distributed Gloo backend unavailable; the two-rank "
            "contract was not run (UNVERIFIED)",
            "checks": {},
            "timing": None,
            "threads": _threads(),
            "environment": _environment(),
        }
    one_process_s, two_rank_s, checks = [], [], {}  # type: ignore[var-annotated]
    for _ in range(repeats):
        start = time.perf_counter()
        single = evaluate_one_process(config)
        middle = time.perf_counter()
        multi = run_ranks(
            "evaluate",
            config,
            workdir=workdir,
            world_size=2,
            timeout_s=timeout_s,
            pg_timeout_s=pg_timeout_s,
        )
        one_process_s.append(middle - start)
        two_rank_s.append(time.perf_counter() - middle)
        for name, passed in compare_results(config, single, multi).items():
            checks[name] = checks.get(name, True) and passed
    rank_metrics = [m for m in multi["accounting"]["rank_metrics"] if m is not None]
    return {
        **base,
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "one_process": _plain(single),
        "two_rank": _plain(multi),
        "metric": {
            "name": "balanced_accuracy",
            "global": multi["metric"],
            "one_process": single["metric"],
            "mean_of_rank_scores_invalid": (
                statistics.fmean(rank_metrics) if rank_metrics else None
            ),
        },
        "timing": {
            "claim": TIMING_CLAIM,
            "repeats": repeats,
            "one_process_s": one_process_s,
            "two_rank_s": two_rank_s,
            "median_one_process_s": statistics.median(one_process_s),
            "median_two_rank_s": statistics.median(two_rank_s),
            "timeouts": {"parent_s": timeout_s, "process_group_s": pg_timeout_s},
        },
        "threads": _threads(),
        "environment": _environment(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-records", type=int, default=DEFAULT_CONFIG["n_records"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--pg-timeout-s", type=float, default=60.0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = dict(
        DEFAULT_CONFIG,
        n_records=args.n_records,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = run_contract(
        config,
        workdir=args.work_dir,
        repeats=args.repeats,
        timeout_s=args.timeout_s,
        pg_timeout_s=args.pg_timeout_s,
    )
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": record["status"], "checks": record["checks"]}))
    if record["status"] == "unsupported":
        return EXIT_UNSUPPORTED
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

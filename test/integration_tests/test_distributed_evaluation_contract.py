"""Two-rank (CPU/Gloo) evaluation identity contract.

Distributed evaluation must count every real record exactly once and compute
metrics on the globally gathered predictions, not as a mean of rank scores.
Ranks are spawned children with an explicit process-group timeout and a parent
deadline, so a failing or hung rank fails the test in bounded time. These are
contract checks on tiny synthetic data, not performance claims.
"""

import json
import math
import os
import re
import time

import pytest
import torch
import torch.distributed as dist
from sklearn.metrics import balanced_accuracy_score

from benchmarks.acceleration import bench_distributed_contract as dc

GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()
requires_gloo = pytest.mark.skipif(
    not GLOO_AVAILABLE,
    reason="UNVERIFIED: torch.distributed Gloo backend unavailable",
)
pytestmark = [pytest.mark.integration, pytest.mark.slow]

PARENT_TIMEOUT_S = 180.0
PG_TIMEOUT_S = 60.0


@pytest.fixture(autouse=True)
def isolate_distributed_state(acceleration_caller_state):
    yield


def _config(n_records, **overrides):
    config = dict(dc.DEFAULT_CONFIG, n_records=n_records)
    config.update(overrides)
    return config


def _two_rank(task, payload, workdir, **kwargs):
    kwargs.setdefault("timeout_s", PARENT_TIMEOUT_S)
    kwargs.setdefault("pg_timeout_s", PG_TIMEOUT_S)
    return dc.run_ranks(task, payload, workdir=workdir, world_size=2, **kwargs)


def _assert_children_reaped(pids):
    assert len(pids) == 2
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _from_ids(ids):
    """Triples whose prediction/target rows depend only on the id."""
    ids = torch.tensor(ids, dtype=torch.int64)
    return ids, torch.stack([ids, -ids], dim=1).to(torch.float64), ids % 2


def _gather(ranks, workdir, *, known_padding=0, expected_ids=None):
    payload = {
        "triples": ranks,
        "known_padding": known_padding,
        "expected_ids": expected_ids,
    }
    return _two_rank("gather", payload, workdir)


def test_gather_single_process_orders_by_source_identity():
    ids = torch.tensor([42, 7, 19])
    preds = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    targets = torch.tensor([1, 0, 1])
    out_ids, out_preds, out_targets = dc.gather_ordered_predictions(
        ids, preds, targets, known_padding=0
    )
    assert torch.equal(out_ids, torch.tensor([7, 19, 42]))
    assert torch.equal(out_preds, preds[[1, 2, 0]])
    assert torch.equal(out_targets, targets[[1, 2, 0]])


def test_gather_requires_declared_padding():
    ids, preds, targets = _from_ids([5, 6])
    with pytest.raises(TypeError, match="known_padding"):
        dc.gather_ordered_predictions(ids, preds, targets)
    with pytest.raises(ValueError, match="known_padding"):
        dc.gather_ordered_predictions(ids, preds, targets, known_padding=-1)
    # One process has no sampler padding to declare.
    with pytest.raises(ValueError, match="DistributedSampler layout"):
        dc.gather_ordered_predictions(ids, preds, targets, known_padding=1)


def test_gather_single_process_rejects_any_duplicate_and_bad_shapes():
    # One process has no sampler padding, so even an identical repeat is a
    # double count, and a conflicting repeat must never be silently dropped.
    preds = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match=r"not declared as sampler padding.*\[3\]"):
        dc.gather_ordered_predictions(
            torch.tensor([3, 3]), preds, torch.tensor([1, 1]), known_padding=0
        )
    with pytest.raises(ValueError, match="conflicting duplicate identity"):
        dc.gather_ordered_predictions(
            torch.tensor([3, 3]), preds, torch.tensor([1, 0]), known_padding=0
        )
    with pytest.raises(ValueError, match="rows"):
        dc.gather_ordered_predictions(
            torch.tensor([1, 2]), preds[:1], torch.tensor([0, 1]), known_padding=0
        )
    with pytest.raises(ValueError, match=r"missing \[7\], unexpected \[\]"):
        dc.gather_ordered_predictions(
            *_from_ids([5, 6]), known_padding=0, expected_ids=torch.tensor([7, 5, 6])
        )
    with pytest.raises(ValueError, match=r"missing \[\], unexpected \[6\]"):
        dc.gather_ordered_predictions(
            *_from_ids([5, 6]), known_padding=0, expected_ids=torch.tensor([5])
        )


@requires_gloo
@pytest.mark.parametrize(
    "n_records, rank_batches",
    [(0, []), (1, [1]), (3, [2]), (7, [3, 1])],
)
def test_two_rank_identity_matches_one_process(n_records, rank_batches, tmp_path):
    config = _config(n_records, batch_size=3)
    single = dc.evaluate_one_process(config)
    multi = _two_rank("evaluate", config, tmp_path)

    dataset = dc.SyntheticRecords(config)
    expected_ids = torch.sort(dataset.source_ids).values
    ref_preds, ref_targets = dc.reference_predictions(config)
    if n_records >= 3:
        assert len(set(dataset.lengths.tolist())) > 1  # unequal trial lengths
    for result in (single, multi):
        assert torch.equal(result["ids"], expected_ids)
        assert torch.equal(result["preds"], ref_preds)
        assert torch.equal(result["targets"], ref_targets)
    assert multi["metric"] == single["metric"]

    # Exactly one contribution per real record after known padding removal:
    # the raw rank contributions are every record once plus the sampler's
    # padding, which repeats the first record in the trailing rank-1 position.
    acct = multi["accounting"]
    total = 2 * math.ceil(n_records / 2)  # DistributedSampler, drop_last=False
    assert acct["world_size"] == 2 and acct["backend"] == "gloo"
    assert acct["local_counts"] == [total // 2, total // 2]
    assert acct["contributions"] == total
    assert acct["expected_padding"] == acct["padding_removed"] == total - n_records
    assert acct["batch_sizes"] == [rank_batches, rank_batches]  # partial batches
    assert acct["global_batch_sizes"] == [2 * b for b in rank_batches]
    assert acct["optimizer_steps"] == 0
    padding_ids = dataset.source_ids[: total - n_records].tolist()
    raw = [i for rank_ids in acct["raw_ids"] for i in rank_ids]
    assert sorted(raw) == sorted(dataset.source_ids.tolist() + padding_ids)
    if padding_ids:
        assert acct["raw_ids"][1][-1] == padding_ids[0]
    assert single["accounting"]["padding_removed"] == 0
    assert single["accounting"]["local_counts"] == [n_records]


@requires_gloo
def test_two_rank_identity_with_shuffled_sampler(tmp_path):
    config = _config(7, batch_size=2, shuffle=True, seed=3)
    single = dc.evaluate_one_process(config)
    multi = _two_rank("evaluate", config, tmp_path)
    for key in ("ids", "preds", "targets"):
        assert torch.equal(multi[key], single[key])
    acct = multi["accounting"]
    assert acct["padding_removed"] == 1
    assert acct["batch_sizes"] == [[2, 2], [2, 2]]
    raw = [i for rank_ids in acct["raw_ids"] for i in rank_ids]
    assert len(raw) == 8 and len(set(raw)) == 7
    assert acct["raw_ids"][1][-1] == acct["raw_ids"][0][0]  # pads with position 0


def _triples(ids, preds, targets):
    return (
        torch.tensor(ids, dtype=torch.int64),
        torch.tensor(preds, dtype=torch.float64).reshape(len(ids), 2),
        torch.tensor(targets, dtype=torch.int64),
    )


@requires_gloo
@pytest.mark.parametrize("field", ["preds", "targets"])
def test_conflicting_duplicate_identity_rejected(field, tmp_path):
    rank0 = _triples([10, 11, 12], [[1, 0], [0, 1], [1, 1]], [0, 1, 0])
    preds1 = [[2, 2], [0, 9]] if field == "preds" else [[2, 2], [0, 1]]
    targets1 = [1, 1] if field == "preds" else [1, 0]
    rank1 = _triples([13, 11], preds1, targets1)  # id 11 on both ranks
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _gather([rank0, rank1], tmp_path)
    message = str(excinfo.value)
    assert "conflicting duplicate identity" in message and "[11]" in message
    # Every rank raises the same error, so no rank is left in a collective.
    assert "timed out" not in message.lower()
    _assert_children_reaped(excinfo.value.child_pids)


@requires_gloo
def test_declared_sampler_padding_removed_by_position(tmp_path):
    rank0, padded = _from_ids([20, 21]), _from_ids([22, 20])  # trailing repeat
    result = _gather(
        [rank0, padded], tmp_path, known_padding=1, expected_ids=[20, 21, 22]
    )
    assert torch.equal(result["ids"], torch.tensor([20, 21, 22]))
    assert result["padding_removed"] == 1
    # The same hand-built repeat is a double count unless padding is declared.
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _gather([rank0, padded], tmp_path)
    assert re.search(r"not declared as sampler padding.*\[20\]", str(excinfo.value))


@requires_gloo
@pytest.mark.parametrize(
    "rank_ids, known_padding, expected",
    [
        # valid trailing padding plus a record evaluated on both ranks
        ([[20, 21, 22], [21, 23, 20]], 1, r"not declared as sampler padding.*\[21\]"),
        # both ranks evaluated everything; nothing was padded
        ([[20, 21], [20, 21]], 0, r"not declared as sampler padding.*\[20, 21\]"),
        # the repeat is not in the trailing padded position
        ([[20, 21], [20, 22]], 1, "DistributedSampler layout"),
        # the trailing row repeats a record the sampler does not pad with
        ([[20, 21], [22, 21]], 1, "DistributedSampler layout"),
        # padding only ever happens on equal-length shards
        ([[20, 21, 22], [20]], 1, "DistributedSampler layout"),
        # DistributedSampler pads fewer than world_size rows
        ([[20, 21], [20, 21]], 2, "DistributedSampler layout"),
    ],
)
def test_double_count_rejected(rank_ids, known_padding, expected, tmp_path):
    ranks = [_from_ids(ids) for ids in rank_ids]
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _gather(ranks, tmp_path, known_padding=known_padding)
    assert re.search(expected, str(excinfo.value)), str(excinfo.value)
    assert "timed out" not in str(excinfo.value).lower()


@requires_gloo
def test_distinct_records_sharing_an_id_not_merged(tmp_path):
    # Equal rows (labels, one-hot or saturated predictions) must not make two
    # different records that share an id look like one record plus padding.
    ranks = [_triples([30, 31], [[1, 0], [0, 1]], [0, 1])]
    ranks.append(_triples([31, 32], [[0, 1], [1, 0]], [1, 0]))
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _gather(ranks, tmp_path)
    assert re.search(r"not declared as sampler padding.*\[31\]", str(excinfo.value))


@requires_gloo
@pytest.mark.parametrize("split", ["sampler", "uneven"])
# sklearn warns when a rank's own records miss a predicted class.
@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_uneven_rank_metric_matches_global_reference(split, tmp_path):
    config = _config(9, batch_size=2)
    single = dc.evaluate_one_process(config)
    predicted = single["preds"].argmax(dim=1)
    by_id = dict(
        zip(single["ids"].tolist(), zip(single["targets"].tolist(), predicted.tolist()))
    )
    if split == "uneven":
        # Rank 1 gets one misclassified record (rank score 0), rank 0 the rest.
        ids = dc.SyntheticRecords(config).source_ids.tolist()
        wrong = [i for i, sid in enumerate(ids) if by_id[sid][0] != by_id[sid][1]]
        assert wrong, "fixture needs a misclassified record"
        config["shards"] = [[i for i in range(9) if i != wrong[0]], [wrong[0]]]
    multi = _two_rank("evaluate", config, tmp_path)

    reference = balanced_accuracy_score(single["targets"].numpy(), predicted.numpy())
    assert multi["metric"] == pytest.approx(reference, rel=1e-12, abs=1e-15)
    assert multi["metric"] == single["metric"]
    acct = multi["accounting"]
    if split == "sampler":
        assert acct["local_counts"] == [5, 5] and acct["padding_removed"] == 1
    else:
        assert acct["local_counts"] == [8, 1] and acct["padding_removed"] == 0
        assert acct["batch_sizes"] == [[2, 2, 2, 2], [1]]
        assert acct["global_batch_sizes"] == [3, 2, 2, 2]
    # Each rank's own score (on its possibly padded records), recomputed
    # independently: balanced accuracy is nonlinear in shard composition, so
    # their mean is not the global metric and must never be reported as one.
    rank_scores = []
    for rank_ids in acct["raw_ids"]:
        rank_true, rank_pred = zip(*(by_id[i] for i in rank_ids))
        rank_scores.append(balanced_accuracy_score(rank_true, rank_pred))
    assert acct["rank_metrics"] == pytest.approx(rank_scores, rel=1e-12, abs=1e-15)
    rank_mean = sum(rank_scores) / len(rank_scores)
    assert rank_mean != pytest.approx(reference, abs=1e-6)


@requires_gloo
@pytest.mark.parametrize(
    "kind, expected",
    [
        ("raise", r"injected failure on rank 1"),
        ("exit", r"exit codes \{0: -?\d+, 1: 17\}"),
        ("hang", r"timed? ?out"),
    ],
)
def test_distributed_worker_failure_times_out(kind, expected, tmp_path):
    pg_timeout_s = 5.0
    start = time.monotonic()
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _two_rank(
            "evaluate",
            _config(7),
            tmp_path,
            pg_timeout_s=pg_timeout_s,
            fault={"rank": 1, "kind": kind},
        )
    elapsed = time.monotonic() - start
    message = str(excinfo.value)
    assert re.search(expected, message, flags=re.IGNORECASE), message
    assert "exit codes" in message
    # A hung peer is detected by the surviving rank's process-group timeout,
    # well before the parent deadline.
    assert elapsed < PARENT_TIMEOUT_S - 60
    if kind == "hang":
        assert excinfo.value.failed_rank == 0
        assert elapsed >= pg_timeout_s
    _assert_children_reaped(excinfo.value.child_pids)
    assert not list(tmp_path.iterdir())  # rendezvous/run files removed


@requires_gloo
@pytest.mark.parametrize("repeat", [0, 1])
@pytest.mark.parametrize("failing_rank", [0, 1])
def test_root_cause_error_recorded_before_teardown(failing_rank, repeat, tmp_path):
    # The failing rank records its error before it tears down the process
    # group, so the peer's resulting transport error cannot hide it.
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _two_rank(
            "evaluate",
            _config(7),
            tmp_path,
            pg_timeout_s=5.0,
            fault={"rank": failing_rank, "kind": "raise"},
        )
    error = excinfo.value
    expected = f"RuntimeError: injected failure on rank {failing_rank}"
    assert error.rank_errors[failing_rank] == expected
    assert error.first_error_rank == failing_rank
    assert f"first error on rank {failing_rank}: {expected}" in str(error)
    _assert_children_reaped(error.child_pids)


@requires_gloo
@pytest.mark.parametrize("kind", ["raise", "exit"])
def test_failure_before_rendezvous_is_bounded(kind, tmp_path):
    start = time.monotonic()
    with pytest.raises(dc.DistributedEvaluationError) as excinfo:
        _two_rank(
            "evaluate",
            _config(3),
            tmp_path,
            pg_timeout_s=PG_TIMEOUT_S,
            fault={"rank": 1, "kind": kind, "when": "before_init"},
        )
    elapsed = time.monotonic() - start
    error = excinfo.value
    assert elapsed < 30  # parent sees the dead rank; rank 0 does not wait 60 s
    assert error.failed_rank == 1
    if kind == "raise":
        assert error.rank_errors[1] == "RuntimeError: injected failure on rank 1"
        assert error.first_error_rank == 1
    else:
        assert error.rank_errors == {}
        assert "1: 17}" in str(error) and "none recorded" in str(error)
    _assert_children_reaped(error.child_pids)
    assert not list(tmp_path.iterdir())


@requires_gloo
def test_invalid_fault_rejected_before_spawn(tmp_path):
    with pytest.raises(ValueError, match="fault"):
        _two_rank("evaluate", _config(1), tmp_path, fault={"rank": 1, "kind": "nap"})
    with pytest.raises(ValueError, match="fault"):
        _two_rank(
            "evaluate",
            _config(1),
            tmp_path,
            fault={"rank": 1, "kind": "exit", "when": "later"},
        )
    assert not list(tmp_path.iterdir())


@requires_gloo
def test_parent_deadline_terminates_hung_ranks(tmp_path):
    timeout_s = 20.0
    start = time.monotonic()
    with pytest.raises(dc.DistributedEvaluationError, match="parent deadline") as e:
        _two_rank(
            "evaluate",
            _config(3),
            tmp_path,
            timeout_s=timeout_s,
            pg_timeout_s=600.0,
            fault={"rank": "all", "kind": "hang"},
        )
    elapsed = time.monotonic() - start
    assert timeout_s <= elapsed < timeout_s + 30
    assert "exit codes {0: -" in str(e.value)  # both ranks killed by a signal
    _assert_children_reaped(e.value.child_pids)
    assert not list(tmp_path.iterdir())


@requires_gloo
def test_contract_record_compares_one_process_and_two_ranks(tmp_path):
    record = dc.run_contract(
        _config(7, batch_size=3),
        workdir=tmp_path,
        repeats=1,
        timeout_s=PARENT_TIMEOUT_S,
        pg_timeout_s=PG_TIMEOUT_S,
    )
    assert record["status"] == "ok"
    assert all(record["checks"].values()) and len(record["checks"]) >= 6
    assert record["checks"]["raw_contributions_cover_each_record_once"]
    assert "one_contribution_per_record" not in record["checks"]
    assert "plan-00" in record["schema_note"]
    assert record["threads"]["torch_threads_per_rank"] == 1
    acct = record["two_rank"]["accounting"]
    assert acct["global_batch_sizes"] == [6, 2]
    assert acct["optimizer_steps"] == 0
    assert record["timing"]["claim"] == "contract overhead only; not a scaling claim"
    assert len(record["timing"]["one_process_s"]) == 1
    assert len(record["timing"]["two_rank_s"]) == 1


def test_missing_gloo_gives_unsupported_record(monkeypatch, tmp_path):
    monkeypatch.setattr(dc, "gloo_available", lambda: False)
    record = dc.run_contract(_config(3), workdir=tmp_path, repeats=1)
    assert record["status"] == "unsupported"
    assert "Gloo" in record["reason"]
    assert record["checks"] == {} and record["timing"] is None
    assert "plan-00" in record["schema_note"]
    output = tmp_path / "record.json"
    argv = ["--n-records", "3", "--repeats", "1", "--work-dir", str(tmp_path / "w")]
    assert dc.main([*argv, "--output", str(output)]) == dc.EXIT_UNSUPPORTED
    assert json.loads(output.read_text())["status"] == "unsupported"
    with pytest.raises(RuntimeError, match="Gloo"):
        dc.run_ranks("evaluate", _config(1), workdir=tmp_path)

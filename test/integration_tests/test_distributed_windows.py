"""Real WindowsDataset/DataLoader multi-rank chronology and metric contracts."""

import numpy as np
import pytest
import torch
import torch.distributed as dist

from benchmarks.acceleration import bench_distributed_windows as recipe
from braindecode import EEGClassifier
from braindecode.training.scoring import iter_trial_predictions

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="Gloo unavailable",
    ),
]


@pytest.fixture(autouse=True)
def isolate_distributed_state(acceleration_caller_state):
    yield


@pytest.mark.parametrize(
    "size,shuffle,exact,n_trials,batch",
    [
        (2, False, True, 7, 5),
        (2, True, False, 7, 5),
        (4, True, True, 7, 5),
        (4, False, False, 7, 4),
        (4, True, False, 1, 3),
    ],
)
def test_real_windows(size, shuffle, exact, n_trials, batch):
    config = {
        "dataset": {"n_trials": n_trials},
        "shuffle": shuffle,
        "exact": exact,
        "batch_size": batch,
    }
    # Exercise collective warmup agreement without timing claims in correctness.
    config["warmups"] = 3
    single = recipe.run(config)
    multi = recipe.run(config, size)
    dataset, lengths = recipe.make_dataset(**config["dataset"])
    model = recipe.DenseModel(exact=exact)
    public = list(iter_trial_predictions(model, dataset, batch_size=batch))
    net = EEGClassifier(
        model, cropped=True, batch_size=batch, train_split=None, verbose=0
    ).initialize()
    estimator, estimator_targets = net.predict_trials(dataset)
    assert multi["ids"].tolist() == list(range(len(dataset)))
    raw_ids = [identity for rank in multi["ranks"] for identity in rank["ids"]]
    assert sorted(raw_ids) == list(range(len(dataset)))
    assert len(raw_ids) == len(set(raw_ids))
    assert torch.equal(multi["rows"], single["rows"])
    assert torch.equal(multi["trial_targets"], single["trial_targets"])
    assert multi["metric"] == single["metric"]
    assert torch.equal(
        multi["trial_scores"].argmax(1), single["trial_scores"].argmax(1)
    )
    tolerance = (
        {"rtol": 0, "atol": 0} if exact else {"rtol": recipe.RTOL, "atol": recipe.ATOL}
    )
    torch.testing.assert_close(multi["preds"], single["preds"], **tolerance)
    reconstructed = recipe.reconstruct(multi["preds"], multi["rows"][:, 1:])
    for index, ((oracle, target), actual) in enumerate(
        zip(public, reconstructed, strict=True)
    ):
        assert actual.shape == (3, lengths[index])
        np.testing.assert_allclose(actual, oracle, **tolerance)
        np.testing.assert_allclose(actual, estimator[index], **tolerance)
        assert target == estimator_targets[index] == multi["trial_targets"][index]
    # Bounded long-trial chunks use the same actual integrated overlap owner.
    chunks = recipe.reconstruct(multi["preds"], multi["rows"][:, 1:], chunk=2)
    offset = 0
    for actual, count in zip(
        reconstructed,
        np.bincount(np.cumsum(multi["rows"][:, 1].numpy() == 0) - 1),
        strict=True,
    ):
        n_chunks = (int(count) + 1) // 2
        np.testing.assert_array_equal(
            np.concatenate(chunks[offset : offset + n_chunks], 1), actual
        )
        offset += n_chunks
    assert offset == len(chunks)
    if n_trials > 1:
        assert any(any(b < batch for b in rank["batches"]) for rank in multi["ranks"])
    else:
        assert sum(not rank["ids"] for rank in multi["ranks"]) == 3


def test_identity_not_trial_local_index():
    dataset, _ = recipe.make_dataset()
    rows = torch.tensor([[dataset[i][1], *dataset[i][2]] for i in range(len(dataset))])
    predictions = torch.zeros(len(dataset), 3, 32)
    from benchmarks.acceleration.bench_distributed_contract import (
        gather_ordered_predictions,
    )

    # Trial-local IDs can collide with different metadata: that is a conflict,
    # not a benign duplicate. Both must fail closed.
    with pytest.raises(ValueError, match="conflicting duplicate identity"):
        gather_ordered_predictions(
            rows[:, 1],
            predictions,
            rows,
            known_padding=0,
            expected_ids=torch.arange(len(dataset)),
        )
    with pytest.raises(ValueError, match="not declared as sampler padding"):
        gather_ordered_predictions(
            torch.tensor([0, 0]),
            predictions[:2],
            rows[:1].repeat(2, 1),
            known_padding=0,
            expected_ids=torch.arange(2),
        )
    with pytest.raises(ValueError, match="missing"):
        gather_ordered_predictions(
            torch.arange(len(dataset) - 1),
            predictions[:-1],
            rows[:-1],
            known_padding=0,
            expected_ids=torch.arange(len(dataset)),
        )

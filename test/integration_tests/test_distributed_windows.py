"""Real WindowsDataset/DataLoader multi-rank chronology and metric contracts."""

import numpy as np
import pytest
import torch
import torch.distributed as dist

from benchmarks import bench_distributed_windows as recipe
from braindecode import EEGClassifier
from braindecode.training.scoring import (
    iter_trial_predictions,
    trial_preds_from_window_preds,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.usefixtures("acceleration_caller_state"),
    pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="Gloo unavailable",
    ),
]


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
        # Exercise collective warmup agreement without timing claims.
        "warmups": 3,
    }
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
    reconstructed = trial_preds_from_window_preds(
        multi["preds"].numpy(),
        multi["rows"][:, 1].tolist(),
        multi["rows"][:, 3].tolist(),
    )
    for index, ((oracle, target), actual) in enumerate(
        zip(public, reconstructed, strict=True)
    ):
        assert actual.shape == (3, lengths[index])
        np.testing.assert_allclose(actual, oracle, **tolerance)
        np.testing.assert_allclose(actual, estimator[index], **tolerance)
        assert target == estimator_targets[index] == multi["trial_targets"][index]
    if n_trials > 1:
        assert any(b < batch for rank in multi["ranks"] for b in rank["batches"])
    else:
        assert sum(rank["n_windows"] == 0 for rank in multi["ranks"]) == 3


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([0, 0, 1]),  # repeated, e.g. trial-local window indices
        torch.tensor([0, 1, 3]),  # missing and foreign
    ],
)
def test_gather_rejects_missing_or_repeated_windows(ids):
    preds = torch.zeros(3, 3, 32)
    rows = torch.zeros(3, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="exactly once"):
        recipe.gather_by_identity(ids, preds, rows, [3], 3)

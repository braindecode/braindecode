"""CPU RNG restoration boundaries for post-epoch training reevaluation."""

import numpy as np
import pytest
import torch
from skorch import NeuralNet
from skorch.utils import to_numpy
from torch.utils.data import Dataset

from braindecode.training.scoring import PostEpochTrainScoring

pytestmark = pytest.mark.usefixtures("acceleration_caller_state")


class _ReevaluationError(RuntimeError):
    pass


class _Dataset(Dataset):
    def __init__(self, error=None):
        self.X = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        self.y = torch.arange(6, dtype=torch.int64) % 2
        self.error = error
        self.loaded = []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        self.loaded.append(index)
        if self.error is not None and index == 2:
            raise self.error
        return self.X[index], self.y[index]


class _Module(torch.nn.Module):
    def __init__(self, error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.error = error
        self.calls = []

    def forward(self, X):
        self.calls.append((self.training, torch.is_grad_enabled()))
        if self.error is not None and len(self.calls) == 2:
            # Exercise RNG consumed within evaluation, not just DataLoader seeding.
            torch.rand(3)
            raise self.error
        return X @ self.weight


def _setup(scoring, *, failure=None):
    error = _ReevaluationError("reevaluation failed")
    dataset = _Dataset(error if failure == "loading" else None)
    net = NeuralNet(
        _Module,
        module__error=error if failure == "evaluation" else None,
        criterion=torch.nn.MSELoss,
        batch_size=2,
        iterator_valid__shuffle=False,
        train_split=None,
        callbacks=[],
        verbose=0,
    ).initialize()
    callback = PostEpochTrainScoring(scoring, name="train_probe").initialize()
    net.callbacks_ = [("train_probe", callback)]
    net.history.new_epoch()
    net.module_.train()
    return net, callback, dataset, error


def _unexpected_score(net, X, y):
    pytest.fail("scorer must not run after failed reevaluation")


@pytest.mark.parametrize("failure", ["iterator", "loading", "evaluation", "target"])
def test_reevaluation_error_restores_cpu_rng_and_propagates(failure, monkeypatch):
    net, callback, dataset, error = _setup(_unexpected_score, failure=failure)
    if failure == "iterator":

        def fail_iterator(*args, **kwargs):
            torch.rand(3)
            raise error

        monkeypatch.setattr(net, "get_iterator", fail_iterator)
    elif failure == "target":
        calls = []

        def fail_target(y):
            calls.append(y)
            if len(calls) == 2:
                torch.rand(3)
                raise error
            return to_numpy(y)

        callback.target_extractor = fail_target

    torch.manual_seed(12345)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(_ReevaluationError) as caught:
        callback.on_epoch_end(net, dataset, None)
    assert caught.value is error
    assert torch.equal(torch.random.get_rng_state(), before)
    assert callback.y_preds_ == []
    assert callback.y_trues_ == []
    assert "forward_iter" not in vars(net)
    assert "train_probe" not in net.history[-1]
    # Preserve skorch's existing mode behavior; do not reset it in the fix.
    assert net.module_.training is (failure == "iterator")
    if failure in {"loading", "evaluation", "target"}:
        assert dataset.loaded[:2] == [0, 1]
        assert net.module_.calls[0] == (False, False)


def test_success_shares_predictions_and_restores_rng_before_scorers():
    observed = []

    def score(net, X, y):
        assert torch.equal(torch.random.get_rng_state(), before)
        assert len(net.module_.calls) == 3
        assert "forward_iter" in vars(net)
        first, second = [cb for _, cb in net.callbacks_]
        assert first.y_preds_ is second.y_preds_
        assert first.y_trues_ is second.y_trues_
        np.testing.assert_array_equal(y, dataset.y.numpy())
        predictions = torch.cat(list(net.forward_iter(X)))
        torch.testing.assert_close(predictions, dataset.X, rtol=0, atol=0)
        assert predictions.device.type == "cpu"
        assert not predictions.requires_grad
        observed.append(predictions)
        return 0.75

    net, first, dataset, _ = _setup(score)
    second = PostEpochTrainScoring(score, name="train_second").initialize()
    net.callbacks_.append(("train_second", second))
    torch.manual_seed(12345)
    before = torch.random.get_rng_state().clone()
    for callback in (first, second):
        assert callback.on_epoch_end(net, dataset, None) is None
        assert "forward_iter" not in vars(net)
    assert len(observed) == 2
    assert net.module_.calls == [(False, False)] * 3
    assert dataset.loaded == list(range(6))
    assert net.module_.training is False
    assert torch.equal(torch.random.get_rng_state(), before)
    assert net.history[-1, "train_probe"] == 0.75
    assert net.history[-1, "train_second"] == 0.75


@pytest.mark.parametrize("raise_in_scorer", [False, True])
def test_scorer_rng_draws_are_not_rolled_back_and_adapter_is_removed(raise_in_scorer):
    error = _ReevaluationError("scorer failed")

    def score(net, X, y):
        assert torch.equal(torch.random.get_rng_state(), before)
        assert "forward_iter" in vars(net)
        torch.rand(5)
        if raise_in_scorer:
            raise error
        return 0.5

    net, callback, dataset, _ = _setup(score)
    torch.manual_seed(12345)
    before = torch.random.get_rng_state().clone()
    torch.rand(5)
    expected = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(before)
    if raise_in_scorer:
        with pytest.raises(_ReevaluationError) as caught:
            callback.on_epoch_end(net, dataset, None)
        assert caught.value is error
        assert "train_probe" not in net.history[-1]
    else:
        callback.on_epoch_end(net, dataset, None)
        assert net.history[-1, "train_probe"] == 0.5
    assert torch.equal(torch.random.get_rng_state(), expected)
    assert "forward_iter" not in vars(net)
    assert net.module_.training is False
    assert len(callback.y_preds_) == 3

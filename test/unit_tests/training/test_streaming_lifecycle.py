# License: BSD-3
"""Public streaming lifetime and cleanup faults."""

import gc
import weakref

import numpy as np
import pytest
import torch

from braindecode.training import scoring

pytestmark = pytest.mark.usefixtures("acceleration_caller_state")


class _Model(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.child = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Identity())
        self.child.eval()
        self.child[1].train()
        self.fail = fail

    def forward(self, x):
        if self.fail:
            raise ValueError("original forward failure")
        return x * self.weight


def _public_iterator(api, module, targets=True):
    if api == "trial":
        return scoring.iter_trial_predictions(module, [], return_targets=targets)
    return scoring.iter_trial_prediction_chunks(
        module, [], windows_per_chunk=2, return_targets=targets
    )


@pytest.mark.parametrize("kind", ["time", "scalar", "vector", "none"])
def test_public_trial_releases_completed_rows(monkeypatch, kind):
    refs = []

    def rows(*args):
        for index in range(2000):
            prediction = np.full((2, 8), index, dtype=np.float32)
            if kind == "time":
                target = prediction.copy()
            elif kind == "vector":
                target = np.array([index, -index], dtype=np.float32)
            elif kind == "scalar":
                target = np.array(index, dtype=np.float32)
            else:
                target = None
            refs.append(weakref.ref(prediction))
            if target is not None:
                refs.append(weakref.ref(target))
            yield prediction, target, index % 2, (index % 2) * 4 + 8

    monkeypatch.setattr(scoring, "_iter_prediction_windows", rows)
    iterator = _public_iterator("trial", _Model(), kind != "none")
    try:
        # 501 rows: over eight CPython tee blocks, across 250 completed trials.
        for trial in range(250):
            output = next(iterator)
            if kind in ("scalar", "vector"):
                # Windows deliberately differ: first-window selection is real.
                target = output[1]
                expected = 2 * trial
                np.testing.assert_array_equal(
                    target, expected if kind == "scalar" else [expected, -expected]
                )
            del output
        gc.collect()
        assert all(ref() is None for ref in refs[:400])
        assert sum(ref() is not None for ref in refs) <= 120
    finally:
        iterator.close()
    gc.collect()
    assert all(ref() is None for ref in refs)


class _CloseFault:
    """An iterator wrapper whose explicit close can fail after closing its owner."""

    def __init__(self, iterator, name, events, fail):
        self.iterator = iterator
        self.name = name
        self.events = events
        self.fail = fail

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def close(self):
        self.events.append(self.name)
        try:
            self.iterator.close()
        finally:
            if self.fail:
                raise RuntimeError(self.name + " close failure")


def _error_chain(error):
    errors = []
    while error is not None and error not in errors:
        errors.append(error)
        error = error.__context__
    return errors


@pytest.mark.parametrize("api", ["trial", "chunk"])
@pytest.mark.parametrize("owner", ["predictions", "targets", "rows", "all"])
@pytest.mark.parametrize("entry", ["close", "error", "exhaustion"])
def test_all_cleanup_attempted_despite_close_errors(monkeypatch, api, owner, entry):
    events = []

    def rows(*args):
        for index in range(8):
            if entry == "error" and index == 3:
                raise ValueError("original row failure")
            value = np.full((1, 4), index, dtype=np.float32)
            yield value, value.copy(), index % 2, (index % 2) * 2 + 4

    def wrap(iterator, name):
        return _CloseFault(iterator, name, events, owner in (name, "all"))

    monkeypatch.setattr(
        scoring, "_iter_prediction_windows", lambda *args: wrap(rows(), "rows")
    )
    reconstruct = scoring._iter_trial_pred_chunks
    targets = scoring._iter_chunk_targets
    monkeypatch.setattr(
        scoring,
        "_iter_trial_pred_chunks",
        lambda *args: wrap(reconstruct(*args), "predictions"),
    )
    monkeypatch.setattr(
        scoring,
        "_iter_chunk_targets",
        lambda *args: wrap(targets(*args), "targets"),
    )
    module = _Model()
    modes = [child.training for child in module.modules()]
    iterator = _public_iterator(api, module)
    with pytest.raises(RuntimeError, match="close failure") as caught:
        if entry == "close":
            next(iterator)
            iterator.close()
        else:
            list(iterator)
    assert set(events) == {"predictions", "targets", "rows"}
    assert [child.training for child in module.modules()] == modes
    if entry == "error":
        assert any(
            isinstance(error, ValueError) and str(error) == "original row failure"
            for error in _error_chain(caught.value)
        )


@pytest.mark.parametrize("api", ["trial", "chunk"])
@pytest.mark.parametrize("entry", ["close", "error", "exhaustion"])
def test_shutdown_fault_still_restores_mixed_modes(monkeypatch, api, entry):
    events = []

    class Batches:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            index = self.index
            if index == 8:
                raise StopIteration
            self.index += 1
            window = index % 2
            return (
                torch.full((1, 1, 4), float(index)),
                torch.full((1, 1, 4), float(index)),
                ([window], [window * 2], [window * 2 + 4]),
            )

        def _shutdown_workers(self):
            events.append("shutdown")
            raise RuntimeError("shutdown failure")

    monkeypatch.setattr(scoring, "_trial_prediction_loader", lambda *args: Batches())
    module = _Model(fail=entry == "error")
    modes = [child.training for child in module.modules()]
    iterator = _public_iterator(api, module)
    with pytest.raises(RuntimeError, match="shutdown failure") as caught:
        if entry == "close":
            next(iterator)
            iterator.close()
        else:
            list(iterator)
    assert events == ["shutdown"]
    assert [child.training for child in module.modules()] == modes
    if entry == "error":
        assert any(
            isinstance(error, ValueError) and str(error) == "original forward failure"
            for error in _error_chain(caught.value)
        )

# License: BSD-3
"""Bounded long-trial output: exact shared overlap and lifecycle contracts."""

import numpy as np
import pytest
import torch

from braindecode.training import scoring
from braindecode.training._trial_aggregation import iter_trial_preds
from test.unit_tests.training.test_scoring import _InferenceModule, _InferenceWindows
from test.unit_tests.training.test_trial_aggregation import (
    _LAYOUTS,
    _QUIRKY,
    _make_case,
    _reference_trial_preds,
)


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    yield


def _equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("size", [1, 2, 3, 100])
@pytest.mark.parametrize("layout", list(_LAYOUTS) + list(_QUIRKY))
def test_primitive_chunks_exact(size, layout):
    if layout in _LAYOUTS:
        _, preds, windows, stops = _make_case(**_LAYOUTS[layout])
    else:
        preds, windows, stops = _QUIRKY[layout]
    expected = _reference_trial_preds(preds, windows, stops)
    actual = list(iter_trial_preds(preds, windows, stops, windows_per_chunk=size))
    # Trial boundaries are preserved, even when the final chunk is short.
    counts = []
    last = -1
    for window in windows:
        if window != last + 1 or not counts:
            counts.append(0)
        counts[-1] += 1
        last = window
    offset = 0
    for count, oracle in zip(counts, expected, strict=True):
        n_chunks = (count + size - 1) // size
        _equal(np.concatenate(actual[offset : offset + n_chunks], axis=1), oracle)
        offset += n_chunks
    assert offset == len(actual)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_chunk_limit(size):
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_trial_preds([[[1]]], [0], [1], windows_per_chunk=size))


@pytest.mark.parametrize("kind", ["scalar", "multi", "time"])
@pytest.mark.parametrize("size", [1, 2, 3, 100])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
@pytest.mark.parametrize("targets", [False, True])
def test_inference_chunks_exact(kind, size, batch_size, targets):
    options = {"batch_size": batch_size}
    module = _InferenceModule()
    expected = list(
        scoring.iter_trial_predictions(
            module, _InferenceWindows(kind), targets, loader_kwargs=options
        )
    )
    actual = list(
        scoring.iter_trial_prediction_chunks(
            module,
            _InferenceWindows(kind),
            windows_per_chunk=size,
            return_targets=targets,
            loader_kwargs=options,
        )
    )
    for trial_index, oracle in enumerate(expected):
        chunks = [c for c in actual if c.trial_index == trial_index]
        prediction, target = oracle if targets else (oracle, None)
        _equal(np.concatenate([c.prediction for c in chunks], axis=1), prediction)
        offset = 0
        for chunk in chunks:
            assert chunk.prediction_start == offset
            assert chunk.prediction.flags.owndata
            offset += chunk.prediction.shape[1]
            if not targets:
                assert chunk.target is None
            elif kind != "time":
                _equal(chunk.target, target)
        if targets and kind == "time":
            _equal(np.concatenate([c.target for c in chunks], axis=1), target)
    assert module.training
    assert torch.is_grad_enabled()


def test_chunks_bounded_and_mode_checkpoint_rng():
    module = _InferenceModule()
    module.child = torch.nn.Dropout().eval()
    dataset = _InferenceWindows(repeats=1000)
    checkpoint = {k: v.clone() for k, v in module.state_dict().items()}
    state = torch.random.get_rng_state()
    iterator = scoring.iter_trial_prediction_chunks(
        module, dataset, windows_per_chunk=1, batch_size=2
    )
    assert dataset.reads == 0
    first = next(iterator)
    assert dataset.reads == 2
    assert torch.is_grad_enabled()
    snapshot = first.prediction.copy()
    second = next(iterator)
    assert not np.shares_memory(first.prediction, second.prediction)
    _equal(first.prediction, snapshot)
    reads = dataset.reads
    iterator.close()
    assert dataset.reads == reads
    assert module.training and not module.child.training
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, checkpoint[key], rtol=0, atol=0)
    chunk_state = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    reference = scoring.iter_trial_predictions(_InferenceModule(), _InferenceWindows())
    next(reference)
    reference.close()
    assert torch.equal(torch.random.get_rng_state(), chunk_state)


@pytest.mark.parametrize(
    "failure", ["close", "exhaustion", "dataset", "forward", "channels"]
)
@pytest.mark.parametrize("workers", [0, 1])
def test_chunks_cleanup(monkeypatch, failure, workers):
    processes = []
    original = scoring.DataLoader

    class TrackingLoader(original):
        def __iter__(self):
            batches = super().__iter__()
            processes.extend(getattr(batches, "_workers", []))
            return batches

    monkeypatch.setattr(scoring, "DataLoader", TrackingLoader)
    module = _InferenceModule(fail=failure == "forward")
    dataset = _InferenceWindows(
        fail_at=2 if failure == "dataset" else None,
        ch_pos=failure == "channels",
    )
    iterator = scoring.iter_trial_prediction_chunks(
        module,
        dataset,
        windows_per_chunk=1,
        loader_kwargs={"num_workers": workers, "persistent_workers": bool(workers)},
    )
    if failure == "close":
        next(iterator)
        iterator.close()
    elif failure == "exhaustion":
        list(iterator)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            list(iterator)
    assert all(not p.is_alive() for p in processes)
    assert module.training and torch.is_grad_enabled()


def test_chunks_special_values_and_targets(monkeypatch):
    def rows(*args):
        for window, stop in enumerate([4, 4, 2, 10, 11]):
            value = np.array([[np.nan, -0.0, np.inf, window]], dtype=np.float32)
            yield value, value.copy(), window, stop

    monkeypatch.setattr(scoring, "_iter_prediction_windows", rows)
    expected = next(scoring.iter_trial_predictions(_InferenceModule(), []))
    chunks = list(
        scoring.iter_trial_prediction_chunks(
            _InferenceModule(), [], windows_per_chunk=2
        )
    )
    _equal(np.concatenate([c.prediction for c in chunks], axis=1), expected[0])
    _equal(np.concatenate([c.target for c in chunks], axis=1), expected[1])
    assert [c.prediction_start for c in chunks] == [0, 8, 14]


@pytest.mark.parametrize("size", [0, -2, 1.5, True, None])
def test_chunk_api_rejects_invalid_limits(size):
    module = _InferenceModule()
    with pytest.raises(ValueError, match="positive integer"):
        list(
            scoring.iter_trial_prediction_chunks(
                module, _InferenceWindows(), windows_per_chunk=size
            )
        )
    assert module.training


def test_chunk_arrays_released_after_advance():
    import weakref

    iterator = scoring.iter_trial_prediction_chunks(
        _InferenceModule(),
        _InferenceWindows("time", repeats=100),
        windows_per_chunk=1,
        return_targets=True,
    )
    first = next(iterator)
    refs = [weakref.ref(first.prediction), weakref.ref(first.target)]
    del first
    next(iterator)
    assert all(ref() is None for ref in refs)
    iterator.close()


def test_long_trial_does_not_pin_tee_history(monkeypatch):
    import weakref

    refs = []

    def rows(*args):
        for window in range(200):
            prediction = np.full((2, 8), window, dtype=np.float32)
            target = prediction.copy()
            refs.extend([weakref.ref(prediction), weakref.ref(target)])
            yield prediction, target, window, window * 4 + 8

    monkeypatch.setattr(scoring, "_iter_prediction_windows", rows)
    iterator = scoring.iter_trial_prediction_chunks(
        _InferenceModule(), [], windows_per_chunk=3
    )
    for _ in range(40):
        next(iterator)
    assert sum(ref() is not None for ref in refs) <= 120
    # tee's internal blocks can hold up to 57 rows, not the entire history.
    assert all(ref() is None for ref in refs[:100])
    iterator.close()

# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#
# License: BSD-3
"""Private streaming reconstruction of trial predictions from window predictions.

``iter_trial_preds`` is the single owner of the overlap-removal algorithm used
by cropped scoring and trial prediction; the public, materialized
``braindecode.training.scoring.trial_preds_from_window_preds`` returns
``list(iter_trial_preds(...))``.
"""

from collections.abc import Iterable, Iterator
from numbers import Integral

import numpy as np


def iter_trial_preds(
    preds: Iterable[np.ndarray],
    i_window_in_trials: Iterable[int],
    i_stop_in_trials: Iterable[int],
    *,
    windows_per_chunk: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield duplicate-free predictions per trial, optionally in bounded chunks.

    Same algorithm and outputs as ``trial_preds_from_window_preds``, but the
    three inputs are consumed lazily and in step: a trial is yielded as soon as
    the first window of the next trial (one window of lookahead) or the end of
    the input shows that it is complete. The reconstruction retains at most
    one active trial/chunk plus one window/lookahead. At a suspended final or
    chunk yield, the last loop window may remain live; emitted arrays are owned
    by the caller, not accumulated by this iterator.

    Parameters
    ----------
    preds : iterable of array-like
        Window predictions in chronological order, each at least 2d with time
        in axis 1, e.g. ``(n_classes, n_preds_per_window)``. Each window is
        copied with ``np.array`` when it is read.
    i_window_in_trials : iterable of int
        Index of each window in its trial; a trial starts at a window index
        that is not one more than the previous one and must be 0.
    i_stop_in_trials : iterable of int
        Stop position of each window in its trial.
    windows_per_chunk : int | None
        If given, emit at most this many source windows per chunk, never
        crossing a trial boundary. Overlap state carries across chunks.
        None (default) retains the existing whole-trial output contract.

    Yields
    ------
    trial_preds : np.ndarray
        Predictions of one trial (or chunk), ``(n_classes, n_times)``, in input order.
        Each is a new array concatenated along axis 1; it does not share memory
        with the inputs.

    Raises
    ------
    AssertionError
        If a new trial does not start at window index 0. Boundary validation
        occurs before yielding the preceding accumulated trial.
    ValueError
        If the input is empty (on the first ``next``), if windows of one trial
        cannot be concatenated, or if the three inputs have different lengths
        (detected when the shortest one is exhausted).
    """
    chunks = _iter_trial_pred_chunks(
        preds, i_window_in_trials, i_stop_in_trials, windows_per_chunk
    )
    try:
        while True:
            try:
                chunk = next(chunks)
            except StopIteration:
                return
            # Pop from a temporary list inside the yield expression: the
            # suspended adapter must not retain its emitted prediction array.
            pending = [chunk[2]]
            del chunk
            yield pending.pop()
    finally:
        chunks.close()


def _iter_trial_pred_chunks(preds, i_window_in_trials, i_stop_in_trials, limit):
    """Overlap owner retaining one active trial/chunk plus one window/lookahead."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0
    ):
        raise ValueError("windows_per_chunk must be a positive integer")
    # Algorithm for assigning window predictions to trials
    # while removing duplicate predictions:
    # Loop through windows:
    # In each iteration you have predictions (assumed: #classes x #timesteps,
    # or at least #timesteps must be in axis=1)
    # and you have i_window_in_trial, i_stop_in_trial
    # (i_trial removed from variable names for brevity)
    # You first check if the i_window_in_trial is 1 larger
    # than in last iteration, then you are still in the same trial
    # Otherwise you are in a new trial, and the previous trial is complete
    # If you are in the same trial, you check for duplicate predictions
    # Only take predictions that are after (inclusive)
    # the stop of the last iteration (i.e., the index of final prediction
    # in the last iteration)
    # Then add the duplicate-removed predictions from this window
    # to predictions for current trial
    cur_trial_preds = []
    trial_index = prediction_start = chunk_windows = 0
    seen = False
    i_last_stop = None
    i_last_window = -1
    for window_preds, i_window, i_stop in zip(
        preds, i_window_in_trials, i_stop_in_trials, strict=True
    ):
        window_preds = np.array(window_preds)
        if i_window != (i_last_window + 1):
            assert i_window == 0, "window numbers in new trial should start from 0"
            if cur_trial_preds:
                yield (
                    trial_index,
                    prediction_start,
                    _concatenate_and_release(cur_trial_preds),
                )
            trial_index += 1
            prediction_start = chunk_windows = 0
            i_last_stop = None

        if i_last_stop is not None:
            # Remove duplicates
            n_needed_preds = i_stop - i_last_stop
            window_preds = window_preds[:, -n_needed_preds:]
        cur_trial_preds.append(window_preds)
        i_last_window = i_window
        i_last_stop = i_stop
        chunk_windows += 1
        seen = True
        if limit is not None and chunk_windows == limit:
            width = sum(p.shape[1] for p in cur_trial_preds)
            yield (
                trial_index,
                prediction_start,
                _concatenate_and_release(cur_trial_preds),
            )
            prediction_start += width
            chunk_windows = 0
    # Preserve the existing empty-input error, but never emit an empty tail
    # after an exactly full chunk. Zero-width windows still count as windows.
    if cur_trial_preds or not seen:
        yield trial_index, prediction_start, _concatenate_and_release(cur_trial_preds)


def _concatenate_and_release(cur_trial_preds: list) -> np.ndarray:
    """Concatenate one trial's windows along time and empty the list in place.

    Called inside the ``yield`` expression so the suspended generator does not
    retain the emitted concatenation or list entries. Its last loop window (or
    boundary lookahead) can remain live until the generator advances.
    """
    trial_preds = np.concatenate(cur_trial_preds, axis=1)
    cur_trial_preds.clear()
    return trial_preds

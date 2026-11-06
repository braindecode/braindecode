# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#
# License: BSD-3
"""Streaming overlap removal behind ``trial_preds_from_window_preds``."""

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
    """Lazily yield ``trial_preds_from_window_preds`` outputs, one trial at a time.

    The inputs are consumed in step, with one window of lookahead to detect the
    end of a trial. With ``windows_per_chunk``, a trial is emitted in chunks of
    at most that many windows; overlap removal carries across chunks. The
    iterator does not keep the arrays it yields.
    """
    windows = zip(preds, i_window_in_trials, i_stop_in_trials, strict=True)
    for chunk in _iter_trial_pred_chunks(windows, windows_per_chunk):
        # Yield from a temporary list so the suspended frame keeps no reference.
        pending = [chunk[2]]
        del chunk
        yield pending.pop()


def _iter_trial_pred_chunks(windows, limit):
    """Yield ``(trial_index, prediction_start, preds)`` from ``(preds, i_window, i_stop)``.

    Holds one active trial (or chunk) plus one window of lookahead.
    """
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
    for window_preds, i_window, i_stop in windows:
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
    # Empty input still raises numpy's error; a full last chunk has no empty tail.
    if cur_trial_preds or not seen:
        yield trial_index, prediction_start, _concatenate_and_release(cur_trial_preds)


def _concatenate_and_release(cur_trial_preds: list) -> np.ndarray:
    """Concatenate along time and empty the list, inside the ``yield`` expression."""
    trial_preds = np.concatenate(cur_trial_preds, axis=1)
    cur_trial_preds.clear()
    return trial_preds

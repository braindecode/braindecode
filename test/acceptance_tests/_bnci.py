"""Left- vs right-hand trials of BNCI2014_001 subject 1 for acceptance tests.

The CI test job already downloads this subject before running the unit tests,
so the acceptance job reuses the same files and ``~/mne_data`` cache.
"""

from functools import lru_cache

import numpy as np

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

SFREQ = 100
MAPPING = {"left_hand": 0, "right_hand": 1}


@lru_cache(maxsize=1)
def _preprocessed_subject_1():
    dataset = MOABBDataset("BNCI2014_001", subject_ids=[1])
    preprocess(
        dataset,
        [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: data * 1e6),  # V -> uV
            Preprocessor("filter", l_freq=4.0, h_freq=38.0),
            Preprocessor("resample", sfreq=SFREQ),
        ],
        n_jobs=1,
    )
    return dataset


def load_left_right_trials(tmin, tmax):
    """Return ``X`` (trials, channels, times), ``y`` and the session of each trial.

    ``tmin`` and ``tmax`` are in seconds relative to the cue; the motor-imagery
    period of BNCI2014_001 lasts 4 s after the cue.
    """
    windows = create_windows_from_events(
        _preprocessed_subject_1(),
        trial_start_offset_samples=round(tmin * SFREQ),
        trial_stop_offset_samples=round((tmax - 4.0) * SFREQ),
        mapping=MAPPING,
        preload=True,
    )
    X = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
    y = np.array([windows[i][1] for i in range(len(windows))], dtype=np.int64)
    session = np.concatenate(
        [[ds.description["session"]] * len(ds) for ds in windows.datasets]
    )
    return X, y, session


def shuffled_labels(y, seed):
    """Balanced labels that are independent of the true class.

    Half of the trials of each true class get each label, so a model can only
    beat chance on these labels by chance.
    """
    rng = np.random.RandomState(seed)
    y_shuffled = np.empty_like(y)
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        y_shuffled[idx] = np.arange(len(idx)) % 2
    return y_shuffled

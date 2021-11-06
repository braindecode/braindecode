# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset, WindowsDataset


def create_from_mne_raw(
        raws, trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples, window_stride_samples, drop_last_window,
        descriptions=None, mapping=None, preload=False, drop_bad_windows=True,
        accepted_bads_ratio=0.0):
    """Create WindowsDatasets from mne.RawArrays

    Parameters
    ----------
    raws: array-like
        list of mne.RawArrays
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial stop in samples
    window_size_samples: int
        window size
    window_stride_samples: int
        stride between windows
    drop_last_window: bool
        whether or not have a last overlapping window, when
        windows do not equally divide the continuous signal
    descriptions: array-like
        list of dicts or pandas.Series with additional information about the raws
    mapping: dict(str: int)
        mapping from event description to target value
    preload: bool
        if True, preload the data of the Epochs objects.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    accepted_bads_ratio: float, optional
        Acceptable proportion of trials withinconsistent length in a raw. If
        the number of trials whose length is exceeded by the window size is
        smaller than this, then only the corresponding trials are dropped, but
        the computation continues. Otherwise, an error is raised. Defaults to
        0.0 (raise an error).

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compativle with skorch
        and braindecode
    """
    # Prevent circular import
    from ..preprocessing.windowers import create_windows_from_events
    if descriptions is not None:
        if len(descriptions) != len(raws):
            raise ValueError(
                f"length of 'raws' ({len(raws)}) and 'description' "
                f"({len(descriptions)}) has to match")
        base_datasets = [BaseDataset(raw, desc) for raw, desc in
                         zip(raws, descriptions)]
    else:
        base_datasets = [BaseDataset(raw) for raw in raws]

    base_datasets = BaseConcatDataset(base_datasets)
    windows_datasets = create_windows_from_events(
        base_datasets,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=drop_last_window,
        mapping=mapping,
        drop_bad_windows=drop_bad_windows,
        preload=preload,
        accepted_bads_ratio=accepted_bads_ratio,
    )
    return windows_datasets


def create_from_mne_epochs(list_of_epochs, window_size_samples,
                           window_stride_samples, drop_last_window):
    """Create WindowsDatasets from mne.Epochs

    Parameters
    ----------
    list_of_epochs: array-like
        list of mne.Epochs
    window_size_samples: int
        window size
    window_stride_samples: int
        stride between windows
    drop_last_window: bool
        whether or not have a last overlapping window, when
        windows do not equally divide the continuous signal

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compativle with skorch
        and braindecode
    """
    # Prevent circular import
    from ..preprocessing.windowers import _check_windowing_arguments
    _check_windowing_arguments(0, 0, window_size_samples,
                               window_stride_samples)

    list_of_windows_ds = []
    for epochs in list_of_epochs:
        event_descriptions = epochs.events[:, 2]
        original_trial_starts = epochs.events[:, 0]
        stop = len(epochs.times) - window_size_samples

        # already includes last incomplete window start
        starts = np.arange(0, stop + 1, window_stride_samples)

        if not drop_last_window and starts[-1] < stop:
            # if last window does not end at trial stop, make it stop there
            starts = np.append(starts, stop)

        fake_events = [[start, window_size_samples, -1] for start in
                       starts]

        for trial_i, trial in enumerate(epochs):
            metadata = pd.DataFrame({
                'i_window_in_trial': np.arange(len(fake_events)),
                'i_start_in_trial': starts + original_trial_starts[trial_i],
                'i_stop_in_trial': starts + original_trial_starts[
                    trial_i] + window_size_samples,
                'target': len(fake_events) * [event_descriptions[trial_i]]
            })
            # window size - 1, since tmax is inclusive
            mne_epochs = mne.Epochs(
                mne.io.RawArray(trial, epochs.info), fake_events,
                baseline=None,
                tmin=0,
                tmax=(window_size_samples - 1) / epochs.info["sfreq"],
                metadata=metadata)

            mne_epochs.drop_bad(reject=None, flat=None)

            windows_ds = WindowsDataset(mne_epochs)
            list_of_windows_ds.append(windows_ds)

    return BaseConcatDataset(list_of_windows_ds)

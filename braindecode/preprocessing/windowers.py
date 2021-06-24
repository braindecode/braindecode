"""Get epochs from mne.Raw
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import warnings

import numpy as np
import mne
import pandas as pd
from joblib import Parallel, delayed

from ..datasets.base import WindowsDataset, BaseConcatDataset


# XXX it's called concat_ds...
def create_windows_from_events(
        concat_ds, start_offset_samples, stop_offset_samples,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False, mapping=None, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error', n_jobs=1):
    """Create windows based on events in mne.Raw.

    This function extracts windows of size window_size_samples in the interval
    [trial onset + start_offset_samples, trial onset + trial duration +
    stop_offset_samples] around each trial, with a separation of
    window_stride_samples between consecutive windows. If the last window
    around an event does not end at stop_offset_samples and
    drop_last_window is set to False, an additional overlapping window that
    ends at stop_offset_samples is created.

    Windows are extracted from the interval defined by the following::

                                                  trial onset +
                    trial onset                     duration
    |--------------------|------------------------------|---------------------|
    trial onset -                                                 trial onset +
    start_offset_samples                                       duration +
                                                      stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        A concat of base datasets each holding raw and description.
    start_offset_samples: int
        Start offset from original trial onsets, in samples.
    stop_offset_samples: int
        Stop offset from original trial stop, in samples.
    window_size_samples: int | None
        Window size. If None, the window size is inferred from the original
        trial size of the first trial and start_offset_samples and
        stop_offset_samples.
    window_stride_samples: int | None
        Stride between windows, in samples. If None, the window stride is
        inferred from the original trial size of the first trial and
        start_offset_samples and stop_offset_samples.
    drop_last_window: bool
        If False, an additional overlapping window that ends at
        stop_offset_samples will be extracted around each event when the
        last window does not end exactly at stop_offset_samples.
    mapping: dict(str: int)
        Mapping from event description to numerical target value.
    preload: bool
        If True, preload the data of the Epochs objects. This is useful to
        reduce disk reading overhead when returning windows in a training
        scenario, however very large data might not fit into memory.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    picks: str | list | slice | None
        Channels to include. If None, all available channels are used. See
        mne.Epochs.
    reject: dict | None
        Epoch rejection parameters based on peak-to-peak amplitude. If None, no
        rejection is done based on peak-to-peak amplitude. See mne.Epochs.
    flat: dict | None
        Epoch rejection parameters based on flatness of signals. If None, no
        rejection based on flatness is done. See mne.Epochs.
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples)

    # If user did not specify mapping, we extract all events from all datasets
    # and map them to increasing integers starting from 0
    infer_mapping = mapping is None
    mapping = dict() if infer_mapping else mapping
    infer_window_size_stride = window_size_samples is None

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_events)(
            ds, infer_mapping, infer_window_size_stride,
            start_offset_samples, stop_offset_samples,
            window_size_samples, window_stride_samples, drop_last_window,
            mapping, preload, drop_bad_windows, picks, reject, flat,
            'error') for ds in concat_ds.datasets)

    return BaseConcatDataset(list_of_windows_ds)


def create_fixed_length_windows(
        concat_ds, start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples, drop_last_window,
        mapping=None, preload=False, drop_bad_windows=True, picks=None,
        reject=None, flat=None, on_missing='error', n_jobs=1):
    """Windower that creates sliding windows.

    Parameters
    ----------
    concat_ds: ConcatDataset
        A concat of base datasets each holding raw and descpription.
    start_offset_samples: int
        Start offset from beginning of recording in samples.
    stop_offset_samples: int | None
        Stop offset from beginning of recording in samples. If None, set to be
        the end of the recording.
    window_size_samples: int
        Window size.
    window_stride_samples: int
        Stride between windows.
    drop_last_window: bool
        Whether or not have a last overlapping window, when windows do not
        equally divide the continuous signal.
    mapping: dict(str: int)
        Mapping from event description to target value.
    preload: bool
        If True, preload the data of the Epochs objects.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as
        otherwise the BaseConcatDataset has to be updated as well.
    picks: str | list | slice | None
        Channels to include. If None, all available channels are used. See
        mne.Epochs.
    reject: dict | None
        Epoch rejection parameters based on peak-to-peak amplitude. If None, no
        rejection is done based on peak-to-peak amplitude. See mne.Epochs.
    flat: dict | None
        Epoch rejection parameters based on flatness of signals. If None, no
        rejection based on flatness is done. See mne.Epochs.
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples)
    if stop_offset_samples == 0:
        warnings.warn(
            'Meaning of `stop_offset_samples`=0 has changed, use `None` '
            'to indicate end of trial/recording. Using `None`.')
        stop_offset_samples = None

    if start_offset_samples != 0 or stop_offset_samples not in [0, None]:
        warnings.warn('Usage of offset_sample args in create_fixed_length_windows is deprecated and'
                      ' will be removed in future versions. Please use '
                      'braindecode.preprocessing.preprocess.Preprocessor("crop", tmin, tmax)'
                      ' instead.')

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_fixed_length_windows)(
            ds, start_offset_samples, stop_offset_samples, window_size_samples,
            window_stride_samples, drop_last_window, mapping, preload,
            drop_bad_windows, picks, reject, flat, on_missing)
        for ds in concat_ds.datasets)

    return BaseConcatDataset(list_of_windows_ds)


def _create_windows_from_events(
        ds, infer_mapping, infer_window_size_stride,
        start_offset_samples, stop_offset_samples,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False, mapping=None, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error'):
    """Create WindowsDataset from BaseDataset based on events.

    Parameters
    ----------
    ds : BaseDataset
        Dataset containing continuous data and description.
    infer_mapping : bool
        If True, extract all events from all datasets and map them to
        increasing integers starting from 0.
    infer_window_size_stride : bool
        If True, infer the stride from the original trial size of the first
        trial and start_offset_samples and stop_offset_samples.

    See `create_windows_from_events` for description of other parameters.

    Returns
    -------
    WindowsDataset :
        Windowed dataset.
    """
    if infer_mapping:
        unique_events = np.unique(ds.raw.annotations.description)
        new_unique_events = [x for x in unique_events if x not in mapping]
        # mapping event descriptions to integers from 0 on
        max_id_mapping = len(mapping)
        mapping.update(
            {v: k + max_id_mapping for k, v in enumerate(new_unique_events)}
        )

    events, events_id = mne.events_from_annotations(ds.raw, mapping)
    onsets = events[:, 0]
    # Onsets are relative to the beginning of the recording
    filtered_durations = np.array(
        [a['duration'] for a in ds.raw.annotations
            if a['description'] in events_id]
    )
    stops = onsets + (filtered_durations * ds.raw.info['sfreq']).astype(int)
    # XXX This could probably be simplified by using chunk_duration in
    #     `events_from_annotations`

    last_samp = ds.raw.first_samp + ds.raw.n_times
    if stops[-1] + stop_offset_samples > last_samp:
        raise ValueError(
            '"stop_offset_samples" too large. Stop of last trial '
            f'({stops[-1]}) + "stop_offset_samples" '
            f'({stop_offset_samples}) must be smaller than length of'
            f' recording ({len(ds)}).')

    if infer_window_size_stride:
        # window size is trial size
        if window_size_samples is None:
            window_size_samples = stops[0] + stop_offset_samples - (
                    onsets[0] + start_offset_samples)
            window_stride_samples = window_size_samples
        this_trial_sizes = (stops + stop_offset_samples) - (
                onsets + start_offset_samples)
        # Maybe actually this is not necessary?
        # We could also just say we just assume window size=trial size
        # in case not given, without this condition...
        # but then would have to change functions overall
        # to deal with varying window sizes hmmhmh
        assert np.all(this_trial_sizes == window_size_samples), (
            'All trial sizes should be the same if you do not supply a window '
            'size.')

    description = events[:, -1]

    i_trials, i_window_in_trials, starts, stops = _compute_window_inds(
        onsets, stops, start_offset_samples,
        stop_offset_samples, window_size_samples,
        window_stride_samples, drop_last_window)

    events = [[start, window_size_samples, description[i_trials[i_start]]]
              for i_start, start in enumerate(starts)]
    events = np.array(events)

    if any(np.diff(events[:, 0]) <= 0):
        raise NotImplementedError('Trial overlap not implemented.')

    description = events[:, -1]

    metadata = pd.DataFrame({
        'i_window_in_trial': i_window_in_trials,
        'i_start_in_trial': starts,
        'i_stop_in_trial': stops,
        'target': description})

    # window size - 1, since tmax is inclusive
    mne_epochs = mne.Epochs(
        ds.raw, events, events_id, baseline=None, tmin=0,
        tmax=(window_size_samples - 1) / ds.raw.info['sfreq'],
        metadata=metadata, preload=preload, picks=picks, reject=reject,
        flat=flat, on_missing=on_missing)

    if drop_bad_windows:
        mne_epochs.drop_bad()

    return WindowsDataset(mne_epochs, ds.description)


def _create_fixed_length_windows(
        ds, start_offset_samples, stop_offset_samples, window_size_samples,
        window_stride_samples, drop_last_window, mapping=None, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error'):
    """Create WindowsDataset from BaseDataset with sliding windows.

    Parameters
    ----------
    ds : BaseDataset
        Dataset containing continuous data and description.

    See `create_fixed_length_windows` for description of other parameters.

    Returns
    -------
    WindowsDataset :
        Windowed dataset.
    """
    stop = ds.raw.n_times \
        if stop_offset_samples is None else stop_offset_samples
    stop = stop - window_size_samples + ds.raw.first_samp
    # already includes last incomplete window start
    starts = np.arange(
        ds.raw.first_samp + start_offset_samples, stop + 1,
        window_stride_samples)

    if not drop_last_window and starts[-1] < stop:
        # if last window does not end at trial stop, make it stop there
        starts = np.append(starts, stop)

    # TODO: handle multi-target case / non-integer target case
    target = -1 if ds.target is None else ds.target
    if mapping is not None:
        target = mapping[target]

    fake_events = [[start, window_size_samples, -1] for start in starts]
    metadata = pd.DataFrame({
        'i_window_in_trial': np.arange(len(fake_events)),
        'i_start_in_trial': starts,
        'i_stop_in_trial': starts + window_size_samples,
        'target': len(fake_events) * [target]
    })

    # window size - 1, since tmax is inclusive
    mne_epochs = mne.Epochs(
        ds.raw, fake_events, baseline=None, tmin=0,
        tmax=(window_size_samples - 1) / ds.raw.info['sfreq'],
        metadata=metadata, preload=preload, picks=picks, reject=reject,
        flat=flat, on_missing=on_missing)

    if drop_bad_windows:
        mne_epochs.drop_bad()

    return WindowsDataset(mne_epochs, ds.description)


def _compute_window_inds(
        starts, stops, start_offset, stop_offset, size, stride,
        drop_last_window):
    """Compute window start and stop indices.

    Create window starts from trial onsets (shifted by start_offset) to trial
    end (shifted by stop_offset) separated by stride, as long as window size
    fits into trial.

    Parameters
    ----------
    starts: array-like
        Trial starts in samples.
    stops: array-like
        Trial stops in samples.
    start_offset: int
        Start offset from original trial onsets in samples.
    stop_offset: int
        Stop offset from original trial stop in samples.
    size: int
        Window size.
    stride: int
        Stride between windows.
    drop_last_window: bool
        Toggles of shifting last window within range or dropping last samples.

    Returns
    -------
    result_lists: (list, list, list, list)
        Trial, i_window_in_trial, start sample and stop sample of windows.
    """
    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts += start_offset
    stops += stop_offset

    i_window_in_trials, i_trials, window_starts = [], [], []
    for start_i, (start, stop) in enumerate(zip(starts, stops)):
        # Generate possible window starts with given stride between original
        # trial onsets (shifted by start_offset) and stops
        possible_starts = np.arange(start, stop, stride)

        # Possible window start is actually a start, if window size fits in
        # trial start and stop
        for i_window, s in enumerate(possible_starts):
            if (s + size) <= stop:
                window_starts.append(s)
                i_window_in_trials.append(i_window)
                i_trials.append(start_i)

        # If the last window start + window size is not the same as
        # stop + stop_offset, create another window that overlaps and stops
        # at onset + stop_offset
        if not drop_last_window:
            if window_starts[-1] + size != stop:
                window_starts.append(stop - size)
                i_window_in_trials.append(i_window_in_trials[-1] + 1)
                i_trials.append(start_i)

    # Update stops to now be event stops instead of trial stops
    window_stops = np.array(window_starts) + size
    if not (len(i_window_in_trials) == len(window_starts) == len(window_stops)):
        raise ValueError(f'{len(i_window_in_trials)} == '
                         f'{len(window_starts)} == {len(window_stops)}')

    return i_trials, i_window_in_trials, window_starts, window_stops


def _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples):
    assert isinstance(start_offset_samples, (int, np.integer))
    assert (isinstance(stop_offset_samples, (int, np.integer)) or
            (stop_offset_samples is None))
    assert isinstance(window_size_samples, (int, np.integer, type(None)))
    assert isinstance(window_stride_samples, (int, np.integer, type(None)))
    assert (window_size_samples is None) == (window_stride_samples is None)
    if window_size_samples is not None:
        assert window_size_samples > 0, (
            "window size has to be larger than 0")
        assert window_stride_samples > 0, (
            "window stride has to be larger than 0")

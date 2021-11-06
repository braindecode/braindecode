"""Get epochs from mne.Raw
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Henrik Bonsmann <henrikbons@gmail.com>
#          Ann-Kathrin Kiessner <ann-kathrin.kiessner@gmx.de>
#          Vytautas Jankauskas <vytauto.jankausko@gmail.com>
#          Dan Wilson <dan.c.wil@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
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
        concat_ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False, mapping=None, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error', accepted_bads_ratio=0.0, n_jobs=1):
    """Create windows based on events in mne.Raw.

    This function extracts windows of size window_size_samples in the interval
    [trial onset + trial_start_offset_samples, trial onset + trial duration +
    trial_stop_offset_samples] around each trial, with a separation of
    window_stride_samples between consecutive windows. If the last window
    around an event does not end at trial_stop_offset_samples and
    drop_last_window is set to False, an additional overlapping window that
    ends at trial_stop_offset_samples is created.

    Windows are extracted from the interval defined by the following::

                                                trial onset +
                        trial onset                duration
        |--------------------|------------------------|-----------------------|
        trial onset -                                             trial onset +
        trial_start_offset_samples                                   duration +
                                                    trial_stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        A concat of base datasets each holding raw and description.
    trial_start_offset_samples: int
        Start offset from original trial onsets, in samples. Defaults to zero.
    trial_stop_offset_samples: int
        Stop offset from original trial stop, in samples. Defaults to zero.
    window_size_samples: int | None
        Window size. If None, the window size is inferred from the original
        trial size of the first trial and trial_start_offset_samples and
        trial_stop_offset_samples.
    window_stride_samples: int | None
        Stride between windows, in samples. If None, the window stride is
        inferred from the original trial size of the first trial and
        trial_start_offset_samples and trial_stop_offset_samples.
    drop_last_window: bool
        If False, an additional overlapping window that ends at
        trial_stop_offset_samples will be extracted around each event when the
        last window does not end exactly at trial_stop_offset_samples.
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
    accepted_bads_ratio: float, optional
        Acceptable proportion of trials withinconsistent length in a raw. If
        the number of trials whose length is exceeded by the window size is
        smaller than this, then only the corresponding trials are dropped, but
        the computation continues. Otherwise, an error is raised. Defaults to
        0.0 (raise an error).
    n_jobs: int
        Number of jobs to use to parallelize the windowing.

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples, window_stride_samples)

    # If user did not specify mapping, we extract all events from all datasets
    # and map them to increasing integers starting from 0
    infer_mapping = mapping is None
    mapping = dict() if infer_mapping else mapping
    infer_window_size_stride = window_size_samples is None

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_events)(
            ds, infer_mapping, infer_window_size_stride,
            trial_start_offset_samples, trial_stop_offset_samples,
            window_size_samples, window_stride_samples, drop_last_window,
            mapping, preload, drop_bad_windows, picks, reject, flat,
            on_missing, accepted_bads_ratio) for ds in concat_ds.datasets)
    return BaseConcatDataset(list_of_windows_ds)


def create_fixed_length_windows(
        concat_ds, start_offset_samples=0, stop_offset_samples=None,
        window_size_samples=None, window_stride_samples=None, drop_last_window=None,
        mapping=None, preload=False, drop_bad_windows=True, picks=None,
        reject=None, flat=None, targets_from='metadata', last_target_only=True,
        on_missing='error', n_jobs=1):
    """Windower that creates sliding windows.

    Parameters
    ----------
    concat_ds: ConcatDataset
        A concat of base datasets each holding raw and description.
    start_offset_samples: int
        Start offset from beginning of recording in samples.
    stop_offset_samples: int | None
        Stop offset from beginning of recording in samples. If None, set to be
        the end of the recording.
    window_size_samples: int | None
        Window size in samples. If None, set to be the maximum possible window size, ie length of
        the recording, once offsets are accounted for.
    window_stride_samples: int | None
        Stride between windows in samples. If None, set to be equal to winddow_size_samples, so
        windows will not overlap.
    drop_last_window: bool | None
        Whether or not have a last overlapping window, when windows do not
        equally divide the continuous signal. Must be set to a bool if window size and stride are
        not None.
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
    stop_offset_samples, drop_last_window = _check_and_set_fixed_length_window_arguments(
        start_offset_samples, stop_offset_samples, window_size_samples, window_stride_samples,
        drop_last_window)

    # check if recordings are of different lengths
    lengths = np.array([ds.raw.n_times for ds in concat_ds.datasets])
    if (np.diff(lengths) != 0).any() and window_size_samples is None:
        warnings.warn('Recordings have different lengths, they will not be batch-able!')
    if any(window_size_samples > lengths):
        raise ValueError(f'Window size {window_size_samples} exceeds trial '
                         f'duration {lengths.min()}.')

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_fixed_length_windows)(
            ds, start_offset_samples, stop_offset_samples, window_size_samples,
            window_stride_samples, drop_last_window, mapping, preload,
            drop_bad_windows, picks, reject, flat, targets_from, last_target_only,
            on_missing) for ds in concat_ds.datasets)

    return BaseConcatDataset(list_of_windows_ds)


def _create_windows_from_events(
        ds, infer_mapping, infer_window_size_stride,
        trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False, mapping=None, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None,
        on_missing='error', accepted_bads_ratio=0.0):
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
        trial and trial_start_offset_samples and trial_stop_offset_samples.

    See `create_windows_from_events` for description of other parameters.

    Returns
    -------
    WindowsDataset :
        Windowed dataset.
    """
    # catch window_kwargs to store to dataset
    window_kwargs = [
        (create_windows_from_events.__name__, _get_windowing_kwargs(locals())),
    ]
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
    if stops[-1] + trial_stop_offset_samples > last_samp:
        raise ValueError(
            '"trial_stop_offset_samples" too large. Stop of last trial '
            f'({stops[-1]}) + "trial_stop_offset_samples" '
            f'({trial_stop_offset_samples}) must be smaller than length of'
            f' recording ({len(ds)}).')

    if infer_window_size_stride:
        # window size is trial size
        if window_size_samples is None:
            window_size_samples = stops[0] + trial_stop_offset_samples - (
                onsets[0] + trial_start_offset_samples)
            window_stride_samples = window_size_samples
        this_trial_sizes = (stops + trial_stop_offset_samples) - (
            onsets + trial_start_offset_samples)
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
        onsets, stops, trial_start_offset_samples,
        trial_stop_offset_samples, window_size_samples,
        window_stride_samples, drop_last_window, accepted_bads_ratio)

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

    windows_ds = WindowsDataset(mne_epochs, ds.description)
    # add window_kwargs and raw_preproc_kwargs to windows dataset
    setattr(windows_ds, 'window_kwargs', window_kwargs)
    kwargs_name = 'raw_preproc_kwargs'
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def _create_fixed_length_windows(
        ds, start_offset_samples, stop_offset_samples, window_size_samples,
        window_stride_samples, drop_last_window, mapping=None, preload=False,
        drop_bad_windows=True, picks=None, reject=None, flat=None, targets_from='metadata',
        last_target_only=True, on_missing='error'):
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
    # catch window_kwargs to store to dataset
    window_kwargs = [
        (create_fixed_length_windows.__name__, _get_windowing_kwargs(locals())),
    ]
    stop = ds.raw.n_times \
        if stop_offset_samples is None else stop_offset_samples

    # assume window should be whole recording
    if window_size_samples is None:
        window_size_samples = stop - start_offset_samples
    if window_stride_samples is None:
        window_stride_samples = window_size_samples

    stop = stop - window_size_samples + ds.raw.first_samp
    # already includes last incomplete window start
    starts = np.arange(
        ds.raw.first_samp + start_offset_samples,
        stop + 1,
        window_stride_samples)

    if not drop_last_window and starts[-1] < stop:
        # if last window does not end at trial stop, make it stop there
        starts = np.append(starts, stop)

    # get targets from dataset description if they exist
    target = -1 if ds.target_name is None else ds.description[ds.target_name]
    if mapping is not None:
        # in case of multiple targets
        if isinstance(target, pd.Series):
            target = target.replace(mapping).to_list()
        # in case of single value target
        else:
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

    window_kwargs.append(
        (WindowsDataset.__name__, {'targets_from': targets_from,
                                   'last_target_only': last_target_only})
    )
    windows_ds = WindowsDataset(mne_epochs, ds.description, targets_from=targets_from,
                                last_target_only=last_target_only)
    # add window_kwargs and raw_preproc_kwargs to windows dataset
    setattr(windows_ds, 'window_kwargs', window_kwargs)
    kwargs_name = 'raw_preproc_kwargs'
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def create_windows_from_target_channels(
        concat_ds, window_size_samples=None, preload=False, drop_bad_windows=True,
        picks=None, reject=None, flat=None, n_jobs=1, last_target_only=True):
    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_target_channels)(
            ds, window_size_samples, preload, drop_bad_windows, picks, reject,
            flat, last_target_only, 'error') for ds in concat_ds.datasets)
    return BaseConcatDataset(list_of_windows_ds)


def _create_windows_from_target_channels(
        ds, window_size_samples, preload=False, drop_bad_windows=True, picks=None,
        reject=None, flat=None, last_target_only=True, on_missing='error'):
    """Create WindowsDataset from BaseDataset using targets `misc` channels from mne.Raw.

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
    window_kwargs = [
        (create_windows_from_target_channels.__name__, _get_windowing_kwargs(locals())),
    ]
    stop = ds.raw.n_times + ds.raw.first_samp

    target = ds.raw.get_data(picks='misc')
    # TODO: handle multi targets present only for some events
    stops = np.nonzero((~np.isnan(target[0, :])))[0]
    stops = stops[(stops < stop) & (stops >= window_size_samples)]
    stops = stops.astype(int)
    # TODO: Make sure that indices are correct
    fake_events = [[stop, window_size_samples, -1] for stop in stops]
    metadata = pd.DataFrame({
        'i_window_in_trial': np.arange(len(fake_events)),
        'i_start_in_trial': stops - window_size_samples,
        'i_stop_in_trial': stops,
        'target': len(fake_events) * [target]
    })

    # window size - 1, since tmax is inclusive
    mne_epochs = mne.Epochs(
        ds.raw, fake_events, baseline=None,
        tmin=-(window_size_samples - 1) / ds.raw.info['sfreq'],
        tmax=0., metadata=metadata, preload=preload, picks=picks,
        reject=reject, flat=flat, on_missing=on_missing)

    if drop_bad_windows:
        mne_epochs.drop_bad()

    window_kwargs.append(
        (WindowsDataset.__name__, {'targets_from': 'channels',
                                   'last_target_only': last_target_only})
    )
    windows_ds = WindowsDataset(mne_epochs, ds.description, targets_from='channels',
                                last_target_only=last_target_only)
    setattr(windows_ds, 'window_kwargs', window_kwargs)
    kwargs_name = 'raw_preproc_kwargs'
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def _compute_window_inds(
        starts, stops, start_offset, stop_offset, size, stride,
        drop_last_window, accepted_bads_ratio):
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
    accepted_bads_ratio: float
        Acceptable proportion of bad trials within a raw. If the number of
        trials whose length is exceeded by the window size is smaller than
        this, then only the corresponding trials are dropped, but the
        computation continues. Otherwise, an error is raised.

    Returns
    -------
    result_lists: (list, list, list, list)
        Trial, i_window_in_trial, start sample and stop sample of windows.
    """
    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts += start_offset
    stops += stop_offset
    if any(size > (stops-starts)):
        bads_mask = size > (stops-starts)
        min_duration = (stops-starts).min()
        if sum(bads_mask) <= accepted_bads_ratio * len(starts):
            starts = starts[np.logical_not(bads_mask)]
            stops = stops[np.logical_not(bads_mask)]
            warnings.warn(
                f'Trials {np.where(bads_mask)[0]} are being dropped as the '
                f'window size ({size}) exceeds their duration {min_duration}.')
        else:
            current_ratio = sum(bads_mask) / len(starts)
            raise ValueError(f'Window size {size} exceeds trial duration '
                             f'({min_duration}) for too many trials '
                             f'({current_ratio * 100}%). Set '
                             f'accepted_bads_ratio to at least {current_ratio}'
                             'and restart training to be able to continue.')

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
        trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples, window_stride_samples):
    assert isinstance(trial_start_offset_samples, (int, np.integer))
    assert (isinstance(trial_stop_offset_samples, (int, np.integer)) or
           (trial_stop_offset_samples is None))
    assert isinstance(window_size_samples, (int, np.integer, type(None)))
    assert isinstance(window_stride_samples, (int, np.integer, type(None)))
    assert (window_size_samples is None) == (window_stride_samples is None)
    if window_size_samples is not None:
        assert window_size_samples > 0, (
            "window size has to be larger than 0")
        assert window_stride_samples > 0, (
            "window stride has to be larger than 0")


def _check_and_set_fixed_length_window_arguments(start_offset_samples, stop_offset_samples,
                                                 window_size_samples, window_stride_samples,
                                                 drop_last_window):
    """Raises warnings for incorrect input arguments and will set correct default values for
    stop_offset_samples & drop_last_window, if necessary.
    """
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples)

    if stop_offset_samples == 0:
        warnings.warn(
            'Meaning of `trial_stop_offset_samples`=0 has changed, use `None` '
            'to indicate end of trial/recording. Using `None`.')
        stop_offset_samples = None

    if start_offset_samples != 0 or stop_offset_samples is not None:
        warnings.warn('Usage of offset_sample args in create_fixed_length_windows is deprecated and'
                      ' will be removed in future versions. Please use '
                      'braindecode.preprocessing.preprocess.Preprocessor("crop", tmin, tmax)'
                      ' instead.')

    if window_size_samples is not None and window_stride_samples is not None and \
            drop_last_window is None:
        raise ValueError('drop_last_window must be set if both window_size_samples &'
                         ' window_stride_samples have also been set')
    elif window_size_samples is None and\
            window_stride_samples is None and\
            drop_last_window is False:
        # necessary for following assertion
        drop_last_window = None

    assert (window_size_samples is None) == \
           (window_stride_samples is None) == \
           (drop_last_window is None)

    return stop_offset_samples, drop_last_window


def _get_windowing_kwargs(windowing_func_locals):
    input_kwargs = windowing_func_locals
    input_kwargs.pop('ds')
    windowing_kwargs = {k: v for k, v in input_kwargs.items()}
    return windowing_kwargs

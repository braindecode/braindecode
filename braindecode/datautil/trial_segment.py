import logging
from collections import OrderedDict, Counter
from copy import deepcopy

import numpy as np

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.util import ms_to_samples

log = logging.getLogger(__name__)

marker_def = OrderedDict(
    (['Right', 1], ['Left', [2]], ['Rest', 3], ['Feet', 4]))


def create_signal_target_from_raw_mne(raw, name_to_start_codes, epoch_ival_ms,
                                      name_to_stop_codes=None,
                                      pad_to_n_samples=None):
    """
    Create SignalTarget set from given `mne.io.RawArray`.
    
    Parameters
    ----------
    raw: `mne.io.RawArray`
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    pad_to_n_samples: int
        Pad trials that would be too short with the signal before it (only
        valid name_to_stop_codes is not None.

    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.
    """
    data = raw.get_data()
    events = np.array([raw.info['events'][:,0],
                      raw.info['events'][:,2]]).T
    fs = raw.info['sfreq']
    return create_signal_target(data, events, fs, name_to_start_codes,
                                epoch_ival_ms,
                                name_to_stop_codes=name_to_stop_codes,
                                pad_to_n_samples=pad_to_n_samples)


def create_signal_target(data, events, fs, name_to_start_codes, epoch_ival_ms,
                         name_to_stop_codes=None, pad_to_n_samples=None):
    """
    Create SignalTarget set given continuous data.
    
    Parameters
    ----------
    data: 2d-array of number
        The continuous recorded data. Channels x times order.
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    pad_to_n_samples: int
        Pad trials that would be too short with the signal before it (only
        valid name_to_stop_codes is not None.


    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.

    """
    if name_to_stop_codes is None:
        return _create_signal_target_from_start_and_ival(
            data, events, fs, name_to_start_codes, epoch_ival_ms)
    else:
        return _create_signal_target_from_start_and_stop(
            data, events, fs, name_to_start_codes, epoch_ival_ms,
            name_to_stop_codes, pad_to_n_samples)


def _to_mrk_code_to_name_and_y(name_to_codes):
    # Create mapping from marker code to class name and y=classindex
    mrk_code_to_name_and_y = {}
    for i_class, class_name in enumerate(name_to_codes):
        codes = name_to_codes[class_name]
        if hasattr(codes, '__len__'):
            for code in codes:
                assert code not in mrk_code_to_name_and_y
                mrk_code_to_name_and_y[code] = (class_name, i_class)
        else:
            assert codes not in mrk_code_to_name_and_y
            mrk_code_to_name_and_y[codes] = (class_name, i_class)
    return mrk_code_to_name_and_y


def _create_signal_target_from_start_and_ival(
        data, events, fs, name_to_codes, epoch_ival_ms):
    ival_in_samples = ms_to_samples(np.array(epoch_ival_ms), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    mrk_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_codes)

    class_to_n_trials = Counter()
    X = []
    y = []

    for i_sample, mrk_code in zip(events[:, 0], events[:, 1]):
        start_sample = int(i_sample) + start_offset
        stop_sample = int(i_sample) + stop_offset
        if mrk_code in mrk_code_to_name_and_y:
            if start_sample < 0:
                log.warning("Ignore trial with marker code {:d}, would start at "
                            "sample {:d}".format(mrk_code, start_sample))
                continue
            if stop_sample > data.shape[1]:
                log.warning("Ignore trial with marker code {:d}, would end at "
                            "sample {:d} of {:d}".format(mrk_code, stop_sample-1,
                                                         data.shape[1]-1))
                continue

            name, this_y = mrk_code_to_name_and_y[mrk_code]
            X.append(data[:, start_sample:stop_sample].astype(np.float32))
            y.append(np.int64(this_y))
            class_to_n_trials[name] += 1
    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return SignalAndTarget(np.array(X), np.array(y))


def _create_signal_target_from_start_and_stop(
        data, events, fs, name_to_start_codes, epoch_ival_ms,
        name_to_stop_codes, pad_to_n_samples=None):
    assert np.array_equal(list(name_to_start_codes.keys()),
                          list(name_to_stop_codes.keys()))
    ival_in_samples = ms_to_samples(np.array(epoch_ival_ms), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    start_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    # Ensure all stop marker codes are iterables
    for name in name_to_stop_codes:
        codes = name_to_stop_codes[name]
        if not hasattr(codes, '__len__'):
            name_to_stop_codes[name] = [codes]
    all_stop_codes = np.concatenate(list(name_to_stop_codes.values()))
    class_to_n_trials = Counter()
    X = []
    y = []

    event_samples = events[:, 0]
    event_codes = events[:, 1]

    i_event = 0
    first_start_code_found = False
    while i_event < len(events):
        while i_event < len(events) and (
                    event_codes[i_event] not in start_code_to_name_and_y):
            i_event += 1
        if i_event < len(events):
            start_sample = event_samples[i_event]
            start_code = event_codes[i_event]
            start_name = start_code_to_name_and_y[start_code][0]
            start_y = start_code_to_name_and_y[start_code][1]
            i_event += 1
            first_start_code_found = True
            waiting_for_end_code = True

            while i_event < len(events) and (
                        event_codes[i_event] not in all_stop_codes):
                if event_codes[i_event] in start_code_to_name_and_y:
                    log.warning(
                        "New start marker  {:.0f} at {:.0f} samples found, "
                        "no end marker for earlier start marker {:.0f} "
                        "at {:.0f} samples found.".format(
                            event_codes[i_event], event_samples[i_event],
                            start_code, start_sample))
                    start_sample = event_samples[i_event]
                    start_name = start_code_to_name_and_y[start_code][0]
                    start_code = event_codes[i_event]
                    start_y = start_code_to_name_and_y[start_code][1]
                i_event += 1
        if i_event == len(events):
            if waiting_for_end_code:
                log.warning(("No end marker for start marker code {:.0f} "
                             "at sample {:.0f} found.").format(start_code,
                                                               start_sample))
            elif (not first_start_code_found):
                log.warning("No markers found at all.")
            break
        stop_sample = event_samples[i_event]
        stop_code = event_codes[i_event]
        assert stop_code in name_to_stop_codes[start_name]
        i_start = int(start_sample) + start_offset
        i_stop = int(stop_sample) + stop_offset
        waiting_for_end_code = False
        if (pad_to_n_samples is not None) and (
                    (i_stop - i_start) < pad_to_n_samples):
            if i_stop < pad_to_n_samples:
                log.warning("Could not pad trial enough, therefore not "
                            "not using trial from {:d} to {:d}".format(
                    i_start, i_stop
                ))
                continue
            i_start = i_stop - pad_to_n_samples
        if i_start < 0:
            log.warning("Ignore trial with start code {:d}, would start at "
                        "sample {:d}".format(start_code, i_start))
            continue
        if i_stop > data.shape[1]:
            log.warning("Ignore trial with stop code {:d}, would end at "
                        "sample {:d} of {:d}".format(stop_code, i_stop - 1,
                                                     data.shape[1] - 1))
            continue

        X.append(data[:, i_start:i_stop].astype(np.float32))
        y.append(np.int64(start_y))
        class_to_n_trials[start_name] += 1

    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return SignalAndTarget(X, np.array(y))


def add_breaks(
        events, fs, break_start_code, break_stop_code, name_to_start_codes,
        name_to_stop_codes, min_break_length_ms=None,
        max_break_length_ms=None, break_start_offset_ms=None,
        break_stop_offset_ms=None):
    """
    Add break events to given events.
    
    Parameters
    ----------
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    break_start_code: int
        Marker code that will be used for break start markers.
    break_stop_code: int
        Marker code that will be used for break stop markers.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to start marker code or 
        start marker codes.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
    min_break_length_ms: number, optional
        Minimum length in milliseconds a break should have to be included.
    max_break_length_ms: number, optional
        Maximum length in milliseconds a break can have to be included.

    Returns
    -------
    events: 2d-array
        Events with break start and stop markers.
    """
    min_samples = (None if min_break_length_ms is None
                   else ms_to_samples(min_break_length_ms, fs))
    max_samples = (None if max_break_length_ms is None
                   else ms_to_samples(max_break_length_ms, fs))
    orig_events = events
    break_starts, break_stops = _extract_break_start_stop_ms(
        events, name_to_start_codes, name_to_stop_codes)

    break_durations = break_stops - break_starts
    valid_mask = np.array([True] * len(break_starts))
    if min_samples is not None:
        valid_mask[break_durations < min_samples] = False
    if max_samples is not None:
        valid_mask[break_durations > max_samples] = False
    if sum(valid_mask) == 0:
        return deepcopy(events)
    break_starts = break_starts[valid_mask]
    break_stops = break_stops[valid_mask]
    if break_start_offset_ms is not None:
        break_starts += int(round(ms_to_samples(break_start_offset_ms, fs)))
    if break_stop_offset_ms is not None:
        break_stops += int(round(ms_to_samples(break_stop_offset_ms, fs)))
    break_events = np.zeros((len(break_starts) * 2, 2))
    break_events[0::2,0] = break_starts
    break_events[1::2,0] = break_stops
    break_events[0::2,1] = break_start_code
    break_events[1::2,1] = break_stop_code

    new_events = np.concatenate((orig_events, break_events))
    # sort events
    sort_order = np.argsort(new_events[:,0], kind='mergesort')
    new_events = new_events[sort_order]
    return new_events


def _extract_break_start_stop_ms(events, name_to_start_codes,
                                 name_to_stop_codes):
    assert len(events[0]) == 2, "expect only 2dimensional event array here"
    start_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    # Ensure all stop marker codes are iterables
    for name in name_to_stop_codes:
        codes = name_to_stop_codes[name]
        if not hasattr(codes, '__len__'):
            name_to_stop_codes[name] = [codes]
    all_stop_codes = np.concatenate(list(name_to_stop_codes.values())).astype(np.int32)
    event_samples = events[:, 0]
    event_codes = events[:, 1]

    break_starts = []
    break_stops = []
    i_event = 0
    while i_event < len(events):
        while (i_event < len(events)) and (
                event_codes[i_event] not in all_stop_codes):
            i_event += 1
        if i_event < len(events):
            # one sample after start
            stop_sample = event_samples[i_event]
            stop_code = event_codes[i_event]
            i_event += 1
            while (i_event < len(events)) and (
                event_codes[i_event] not in start_code_to_name_and_y):
                if event_codes[i_event] in all_stop_codes:
                    log.warning(
                        "New end marker  {:.0f} at {:.0f} samples found, "
                        "no start marker for earlier end marker {:.0f} "
                        "at {:.0f} samples found.".format(
                            event_codes[i_event],
                            event_samples[i_event],
                            stop_code, stop_sample))
                stop_sample = event_samples[i_event] + 1
                stop_code = event_codes[i_event]
                i_event += 1

            if i_event == len(events):
                break

            start_sample = event_samples[i_event]
            start_code = event_codes[i_event]
            assert start_code in start_code_to_name_and_y
            # let's start one after stop of the trial and stop one efore
            # start of the trial to ensure that markers will be
            # in right order
            break_starts.append(stop_sample + 1)
            break_stops.append(start_sample - 1)
    return np.array(break_starts), np.array(break_stops)


def create_cnt_y_and_start_stop_samples(
        n_samples, events, fs, name_to_start_codes, epoch_ival_ms,
        name_to_stop_codes):
    """
    Create a one-hot-encoded continuous marker array (cnt_y).
    
    Parameters
    ----------
    n_samples: int
        Number of samples=timesteps in the recorded data.
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.


    """
    assert np.array_equal(list(name_to_start_codes.keys()),
                          list(name_to_stop_codes.keys()))
    events = np.asarray(events)
    ival_in_samples = ms_to_samples(np.array(epoch_ival_ms), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    start_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    # Ensure all stop marker codes are iterables
    for name in name_to_stop_codes:
        codes = name_to_stop_codes[name]
        if not hasattr(codes, '__len__'):
            name_to_stop_codes[name] = [codes]
    all_stop_codes = np.concatenate(list(name_to_stop_codes.values())).astype(np.int64)
    class_to_n_trials = Counter()
    n_classes = len(name_to_start_codes)
    cnt_y = np.zeros((n_samples, n_classes), dtype=np.int64)

    event_samples = events[:, 0]
    event_codes = events[:, 1]
    i_start_stops = []
    i_event = 0
    first_start_code_found = False
    while i_event < len(events):
        while i_event < len(events) and (
                    event_codes[i_event] not in start_code_to_name_and_y):
            i_event += 1
        if i_event < len(events):
            start_sample = event_samples[i_event]
            start_code = event_codes[i_event]
            start_name = start_code_to_name_and_y[start_code][0]
            start_y = start_code_to_name_and_y[start_code][1]
            i_event += 1
            first_start_code_found = True
            waiting_for_end_code = True

            while i_event < len(events) and (
                        event_codes[i_event] not in all_stop_codes):
                if event_codes[i_event] in start_code_to_name_and_y:
                    log.warning(
                        "New start marker  {:.0f} at {:.0f} samples found, "
                        "no end marker for earlier start marker {:.0f} "
                        "at {:.0f} samples found.".format(
                            event_codes[i_event], event_samples[i_event],
                            start_code, start_sample))
                    start_sample = event_samples[i_event]
                    start_name = start_code_to_name_and_y[start_code][0]
                    start_code = event_codes[i_event]
                    start_y = start_code_to_name_and_y[start_code][1]
                i_event += 1
        if i_event == len(events):
            if waiting_for_end_code:
                log.warning(("No end marker for start marker code {:.0f} "
                             "at sample {:.0f} found.").format(start_code,
                                                               start_sample))
            elif (not first_start_code_found):
                log.warning("No markers found at all.")
            break
        stop_sample = event_samples[i_event]
        stop_code = event_codes[i_event]
        assert stop_code in name_to_stop_codes[start_name]
        i_start = int(start_sample) + start_offset
        i_stop = int(stop_sample) + stop_offset
        cnt_y[i_start:i_stop, start_y] = 1
        i_start_stops.append((i_start, i_stop))
        class_to_n_trials[start_name] += 1
        waiting_for_end_code = False

    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return cnt_y, i_start_stops


def create_signal_target_with_breaks_from_mne(
        cnt, name_to_start_codes,
        trial_segment_ival_ms,
        name_to_stop_codes,
        min_break_length_ms, max_break_length_ms,
        break_segment_ival_ms,
        pad_trials_to_n_samples=None):
    assert 'Break' not in name_to_start_codes
    # Create new marker codes for start and stop of breaks
    # Use marker codes that did not exist in the given marker codes...
    all_start_codes = np.concatenate(
        [np.atleast_1d(vals) for vals in name_to_start_codes.values()])
    all_stop_codes = np.concatenate(
        [np.atleast_1d(vals) for vals in name_to_stop_codes.values()])
    break_start_code = -1
    while break_start_code in np.concatenate((all_start_codes, all_stop_codes)):
        break_start_code -= 1
    break_stop_code = break_start_code - 1
    while break_stop_code in np.concatenate((all_start_codes, all_stop_codes)):
        break_stop_code -= 1

    events = cnt.info['events'][:, [0, 2]]
    # later trial segment ival will be added when creating set
    # so remove it here
    break_segment_ival_ms = np.array(break_segment_ival_ms) - (
        np.array(trial_segment_ival_ms))
    events_with_breaks = add_breaks(events, cnt.info['sfreq'],
                                    break_start_code, break_stop_code,
                                    name_to_start_codes, name_to_stop_codes,
                                    min_break_length_ms=min_break_length_ms,
                                    max_break_length_ms=max_break_length_ms,
                                    break_start_offset_ms=break_segment_ival_ms[
                                        0],
                                    break_stop_offset_ms=break_segment_ival_ms[
                                        1])

    name_to_start_codes_with_breaks = deepcopy(name_to_start_codes)
    name_to_start_codes_with_breaks['Break'] = break_start_code
    name_to_stop_codes_with_breaks = deepcopy(name_to_stop_codes)
    name_to_stop_codes_with_breaks['Break'] = break_stop_code


    data = cnt.get_data()
    fs = cnt.info['sfreq']
    signal_target = create_signal_target_with_cnt_y(
        data, events_with_breaks, fs,
        name_to_start_codes_with_breaks, trial_segment_ival_ms,
        name_to_stop_codes_with_breaks,
        pad_to_n_samples=pad_trials_to_n_samples)

    return signal_target


def create_signal_target_with_cnt_y_from_raw_mne(
        raw, name_to_start_codes, epoch_ival_ms,
          name_to_stop_codes,
          pad_to_n_samples=None):
    data = raw.get_data()
    events = raw.info['events'][:,[0,2]]
    fs = raw.info['sfreq']
    return create_signal_target_with_cnt_y(data, events, fs,
                                    name_to_start_codes, epoch_ival_ms,
                                    name_to_stop_codes,
                                    pad_to_n_samples=pad_to_n_samples
                                    )


def create_signal_target_with_cnt_y(data, events, fs,
                                    name_to_start_codes, epoch_ival_ms,
                                    name_to_stop_codes,
                                    pad_to_n_samples=None
                                    ):
    """
    Create a signal

    Parameters
    ----------
    data: 2d-array of number
        The continuous recorded data. Channels x times order.
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    pad_to_n_samples: int, optional
        Use signal before trial start to pad trials that are otherwise too small.

    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels,
        one array per trial, as labels can be different within one trial.
    """
    cnt_y, i_start_stops = create_cnt_y_and_start_stop_samples(
        data.shape[1], events, fs,
        name_to_start_codes,
        epoch_ival_ms, name_to_stop_codes, )
    return create_signal_target_from_cnt_y_start_stops(
        data, cnt_y, i_start_stops, pad_to_n_samples, one_hot_y=False,
        one_label_per_trial=True)


def create_signal_target_from_cnt_y_start_stops(
        data,
        cnt_y,
        i_start_stops,
        pad_to_n_samples,
        one_hot_y,
        one_label_per_trial):
    if pad_to_n_samples is not None:
        new_i_start_stops = []
        for i_start, i_stop in i_start_stops:
            if (i_stop - i_start) > pad_to_n_samples:
                new_i_start_stops.append((i_start, i_stop))
            elif i_stop >= pad_to_n_samples:
                new_i_start_stops.append(
                    (i_stop - pad_to_n_samples, i_stop))
            else:
                log.warning("Could not pad trial enough, therefore not "
                            "not using trial from {:d} to {:d}".format(
                    i_start, i_stop
                ))
                continue

    else:
        new_i_start_stops = i_start_stops

    X = []
    y = []
    for i_start, i_stop in new_i_start_stops:
        if i_start < 0:
            log.warning("Trial start too early, therefore not "
                        "not using trial from {:d} to {:d}".format(
                i_start, i_stop
            ))
            continue
        if i_stop > data.shape[1]:
            log.warning("Trial stop too late, therefore not "
                        "not using trial from {:d} to {:d}".format(
                i_start, i_stop
            ))
            continue
        X.append(data[:, i_start:i_stop].astype(np.float32))
        y.append(cnt_y[i_start:i_stop])

    if not one_hot_y:
        # change from one-hot-encoding to regular encoding
        # with -1 as indication none of the classes are present
        new_y = []
        for this_y in y:
            this_new_y = np.argmax(this_y, axis=1)
            this_new_y[np.sum(this_y, axis=1) == 0] = -1
            new_y.append(this_new_y)
        y = new_y
    # take last label always
    if one_label_per_trial:
        y = [this_y[-1] for this_y in y]
    return SignalAndTarget(X, y)







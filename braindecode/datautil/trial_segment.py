import logging
from collections import OrderedDict, Counter

import numpy as np

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.util import ms_to_samples

log = logging.getLogger(__name__)

marker_def = OrderedDict(
    (['Right', 1], ['Left', [2]], ['Rest', 3], ['Feet', 4]))


def create_signal_target_from_raw_mne(raw, name_to_codes, epoch_ival,
                         name_to_stop_codes=None):
    """ Create SignalTarget set from X and y given raw mne object."""
    data = raw.get_data()
    events = np.array([raw.info['events'][:,0],
                      raw.info['events'][:,2] - raw.info['events'][:,1]]).T
    fs = raw.info['sfreq']
    return create_signal_target(data, events, fs, name_to_codes, epoch_ival,
                         name_to_stop_codes=name_to_stop_codes)

def create_signal_target(data, events, fs, name_to_codes, epoch_ival,
                         name_to_stop_codes=None):
    if name_to_stop_codes is None:
        return create_signal_target_from_start_and_ival(
            data, events, fs, name_to_codes, epoch_ival)
    else:
        return create_signal_target_from_start_and_stop(
            data, events, fs, name_to_codes, epoch_ival, name_to_stop_codes)


def to_mrk_code_to_name_and_y(name_to_codes):
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


def create_signal_target_from_start_and_ival(
        data, events, fs, name_to_codes, epoch_ival):
    ival_in_samples = ms_to_samples(np.array(epoch_ival), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    mrk_code_to_name_and_y = to_mrk_code_to_name_and_y(name_to_codes)

    class_to_n_trials = Counter()
    X = []
    y = []

    for i_sample, mrk_code in zip(events[:, 0], events[:, 1]):
        start_sample = int(i_sample) + start_offset
        stop_sample = int(i_sample) + stop_offset
        if mrk_code in mrk_code_to_name_and_y:
            name, this_y = mrk_code_to_name_and_y[mrk_code]
            X.append(data[:, start_sample:stop_sample].astype(np.float32))
            y.append(np.int64(this_y))
            class_to_n_trials[name] += 1
    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return SignalAndTarget(np.array(X), np.array(y))


def create_signal_target_from_start_and_stop(
        data, events, fs, name_to_start_codes, epoch_ival, name_to_stop_codes):
    assert np.array_equal(list(name_to_start_codes.keys()),
                          list(name_to_stop_codes.keys()))
    ival_in_samples = ms_to_samples(np.array(epoch_ival), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    start_code_to_name_and_y = to_mrk_code_to_name_and_y(name_to_start_codes)
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
            log.warning(("No end marker for start marker code {:.0f} "
                         "at sample {:.0f} found.").format(start_code,
                                                           start_sample))
            break
        stop_sample = event_samples[i_event]
        stop_code = event_codes[i_event]
        assert stop_code in name_to_stop_codes[start_name]
        i_start = int(start_sample) + start_offset
        i_stop = int(stop_sample) + stop_offset
        X.append(data[:, i_start:i_stop].astype(np.float32))
        y.append(np.int64(start_y))
        class_to_n_trials[start_name] += 1

    log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return SignalAndTarget(X, np.array(y))


def add_breaks(
        events, fs, break_start_code, break_stop_code, name_to_start_codes=None,
        name_to_stop_codes=None, min_break_length_ms=None,
        max_break_length_ms=None):
    min_samples = (None if min_break_length_ms is None
                   else ms_to_samples(min_break_length_ms, fs))
    max_samples = (None if min_break_length_ms is None
                   else ms_to_samples(max_break_length_ms, fs))
    orig_events = events
    events = np.array([events[:,0], events[:,2] - events[:,1]]).T
    break_starts, break_stops = extract_break_start_stop_ms(
        events, name_to_start_codes, name_to_stop_codes)

    break_durations = break_stops - break_starts
    valid_mask = np.array([True] * len(break_starts))
    if min_samples is not None:
        valid_mask[break_durations < min_samples] = False
    if max_samples is not None:
        valid_mask[break_durations > max_samples] = False
    break_starts = break_starts[valid_mask]
    break_stops = break_stops[valid_mask]

    break_events = np.zeros((len(break_starts) * 2, 3))
    break_events[0::2,0] = break_starts
    break_events[1::2,0] = break_stops
    break_events[0::2,2] = break_start_code
    break_events[1::2,2] = break_stop_code

    new_events = np.concatenate((orig_events, break_events))
    # sort events
    sort_order = np.argsort(new_events[:,0], kind='mergesort')
    new_events = new_events[sort_order]
    return new_events


def extract_break_start_stop_ms(events, name_to_start_codes,
                                name_to_stop_codes):
    start_code_to_name_and_y = to_mrk_code_to_name_and_y(name_to_start_codes)
    # Ensure all stop marker codes are iterables
    for name in name_to_stop_codes:
        codes = name_to_stop_codes[name]
        if not hasattr(codes, '__len__'):
            name_to_stop_codes[name] = [codes]
    all_stop_codes = np.concatenate(list(name_to_stop_codes.values()))
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
            stop_sample = event_samples[i_event] + 1
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
            break_starts.append(stop_sample)
            break_stops.append(start_sample - 1)
    return np.array(break_starts), np.array(break_stops)

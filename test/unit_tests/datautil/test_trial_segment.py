from collections import OrderedDict

import numpy as np
import pytest

from braindecode.datautil.trial_segment import (
    _create_cnt_y_and_trial_bounds_from_start_stop)
from braindecode.datautil.trial_segment import (
    _create_signal_target_from_start_and_ival)
from braindecode.datautil.trial_segment import (
    _create_signal_target_from_start_and_stop)
from braindecode.datautil.trial_segment import add_breaks




def check_cnt_y_start_stop_samples(n_samples, events, fs, epoch_ival_ms,
                                   name_to_start_codes,
                                   name_to_stop_codes, cnt_y, start_stop):

    cnt_y = np.array(cnt_y).T
    real_cnt_y, real_start_stop = _create_cnt_y_and_trial_bounds_from_start_stop(
        n_samples, events ,fs, name_to_start_codes, epoch_ival_ms,
        name_to_stop_codes)
    np.testing.assert_array_equal(cnt_y, real_cnt_y)
    np.testing.assert_array_equal(start_stop, real_start_stop)


def test_cnt_y_start_stop_samples():
    check_cnt_y_start_stop_samples(n_samples=5, events=[(0, 1), (3, 2), ],
                                   fs=10, epoch_ival_ms=[0, 0],
                                   name_to_start_codes=OrderedDict(
                                       [('Event', 1)]),
                                   name_to_stop_codes=OrderedDict(
                                       [('Event', 2)]),
                                   cnt_y=[[1, 1, 1, 0, 0]],
                                   start_stop=[(0, 3)])


def test_cnt_y_start_stop_samples_epoch_ival():
    check_cnt_y_start_stop_samples(n_samples=5, events=[(0, 1), (3, 2)], fs=10,
                                   epoch_ival_ms=[0, -100],
                                   name_to_start_codes=OrderedDict(
                                       [('Event', 1)]),
                                   name_to_stop_codes=OrderedDict(
                                       [('Event', 2)]),
                                   cnt_y=[[1, 1, 0, 0, 0]],
                                   start_stop=[(0, 2)])


def test_cnt_y_start_stop_samples_two_class():
    check_cnt_y_start_stop_samples(n_samples=5, events=[(0, 1), (3, -1)], fs=10,
                                   epoch_ival_ms=[0, 0],
                                   name_to_start_codes=OrderedDict(
                                       [('Event1', 1), ('Event2', 2)]),
                                   name_to_stop_codes=OrderedDict(
                                       [('Event1', -1), ('Event2', -2)]),
                                   cnt_y=[[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
                                   start_stop=[(0, 3)])


def test_cnt_y_start_stop_samples_two_class_with_both_appearing():
    check_cnt_y_start_stop_samples(n_samples=8,
                                   events=[(0, 1), (2, -1), (3, 2), (6, -2)],
                                   fs=10, epoch_ival_ms=[0, 0],
                                   name_to_start_codes=OrderedDict(
                                       [('Event1', 1), ('Event2', 2)]),
                                   name_to_stop_codes=OrderedDict(
                                       [('Event1', -1), ('Event2', -2)]),
                                    cnt_y=[[1, 1, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 1, 0, 0]],
                                   start_stop=[(0, 2), (3, 6)])


def check_signal_target_from_start_and_ival(data, events, fs, name_to_codes,
                                           epoch_ival_ms, expected_X, expected_y):
    data = np.array(data)
    events = np.array(events)
    name_to_codes = OrderedDict(name_to_codes)
    out_set = _create_signal_target_from_start_and_ival(
        data, events, fs, name_to_codes, epoch_ival_ms,
        one_hot_labels=False, one_label_per_trial=True)
    np.testing.assert_array_equal(out_set.y, expected_y)
    np.testing.assert_allclose(out_set.X, expected_X)


def test_signal_target_from_start_and_ival():
    check_signal_target_from_start_and_ival(
        data=[np.arange(10)], events=[(0, 1)], fs=100, name_to_codes=[('A', 1)],
        epoch_ival_ms=[0, 30],
        expected_X=[[[0, 1, 2]]], expected_y=[0])


def test_signal_target_from_start_and_ival_ignored_marker():
    check_signal_target_from_start_and_ival(
        data=[np.arange(10)], events=[(0, 1), (5, 2)], fs=100,
        name_to_codes=[('A', 1)], epoch_ival_ms=[0, 30],
        expected_X=[[[0, 1, 2]]], expected_y=[0])


def test_signal_target_from_start_and_ival_two_class():
    check_signal_target_from_start_and_ival(
        data=[np.arange(10)], events=[(0,1), (5,2)], fs=100,
        name_to_codes=[('A', 1), ('B', 2)], epoch_ival_ms=[0, 30],
        expected_X=[[[0,1,2]], [[5,6,7]]], expected_y=[0,1])


def test_signal_target_from_start_and_ival_too_early():
    check_signal_target_from_start_and_ival(
        data=[np.arange(10)], events=[(0, 1)], fs=100, name_to_codes=[('A', 1)],
        epoch_ival_ms=[-10, 30],
        expected_X=[], expected_y=[])


def test_signal_target_from_start_and_ival_too_late():
    check_signal_target_from_start_and_ival(
        data=[np.arange(10)], events=[(8, 1)], fs=100, name_to_codes=[('A', 1)],
        epoch_ival_ms=[0, 30],
        expected_X=[], expected_y=[])


def test_signal_target_from_start_and_ival_overlapping():
    check_signal_target_from_start_and_ival(
        data=[np.arange(10)], events=[(0, 1), (1, 2)], fs=100,
        name_to_codes=[('A', 1), ('B', 2)], epoch_ival_ms=[0, 30],
        expected_X=[[[0, 1, 2]], [[1, 2, 3]]], expected_y=[0, 1])


def check_signal_target_from_start_and_stop(data, events, fs, name_to_codes,
                                           epoch_ival_ms, name_to_stop_codes,
                                            pad_to_n_samples,
                                            expected_X, expected_y, ):
    data = np.array(data)
    events = np.array(events)
    name_to_codes = OrderedDict(name_to_codes)
    name_to_stop_codes = OrderedDict(name_to_stop_codes)
    out_set = _create_signal_target_from_start_and_stop(
        data, events, fs, name_to_codes, epoch_ival_ms, name_to_stop_codes,
        pad_to_n_samples, one_hot_labels=False, one_label_per_trial=True)
    np.testing.assert_array_equal(out_set.y, expected_y)
    assert len(out_set.X) == len(expected_X)
    for x_out, x_expected in zip(out_set.X, expected_X):
        np.testing.assert_allclose(x_out, x_expected)


def test_signal_target_from_start_and_stop():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)], events=[(0, 1), (2, -1)], fs=100,
        name_to_codes=[('A', 1)], epoch_ival_ms=[0, 20],
        name_to_stop_codes=[('A', -1)],
        pad_to_n_samples=None,
        expected_X=[[[0, 1, 2, 3]]], expected_y=[0])


def test_signal_target_from_start_and_stop_different_ival():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)], events=[(0, 1), (2, -1)], fs=100,
        name_to_codes=[('A', 1)], epoch_ival_ms=[0, -10],
        name_to_stop_codes=[('A', -1)],
        pad_to_n_samples=None,
        expected_X=[[[0, ]]], expected_y=[0])


def test_signal_target_from_start_and_stop_ignored_marker():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)], events=[(0, 1), (1, 3), (2, -1)], fs=100,
        name_to_codes=[('A', 1)], epoch_ival_ms=[0, -10],
        name_to_stop_codes=[('A', -1)],
        pad_to_n_samples=None,
        expected_X=[[[0, ]]], expected_y=[0])


def test_signal_target_from_start_and_stop_too_early():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)], events=[(0, 1), (2, -1)], fs=100,
        name_to_codes=[('A', 1)], epoch_ival_ms=[-10, 20],
        name_to_stop_codes=[('A', -1)],
        pad_to_n_samples=None,
        expected_X=[], expected_y=[])


def test_signal_target_from_start_and_stop_too_late():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)], events=[(0, 1), (2, -1), (8, 1), (9, -1)], fs=100,
        name_to_codes=[('A', 1)], epoch_ival_ms=[0, 20],
        name_to_stop_codes=[('A', -1)],
        pad_to_n_samples=None,
        expected_X=[[[0, 1, 2, 3]]], expected_y=[0])


def test_signal_target_from_start_and_stop_overlapping():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)],
        events=[(0, 1), (2, -1), (2, 2), (5, -2)],
        fs=100,
        name_to_codes=[('A', 1), ('B', 2)],
        epoch_ival_ms=[0, 20],
        name_to_stop_codes=[('A', -1), ('B', -2)],
        pad_to_n_samples=None,
        expected_X=[[[0, 1, 2, 3]], [[2, 3, 4, 5, 6]]],
        expected_y=[0, 1])


def test_signal_target_from_start_and_stop_stop_missing():
    check_signal_target_from_start_and_stop(
        data=[np.arange(10)],
        events=[(0, 1), (2, -1), (2, 2), ],
        fs=100,
        name_to_codes=[('A', 1), ('B', 2)],
        epoch_ival_ms=[0, 20],
        name_to_stop_codes=[('A', -1), ('B', -2)],
        pad_to_n_samples=None,
        expected_X=[[[0, 1, 2, 3]], ],
        expected_y=[0, ])


def test_signal_target_from_start_and_stop_wrong_stop_for_start():
    # wrong stop for start
    # expect assertion raised
    with pytest.raises(AssertionError):
        check_signal_target_from_start_and_stop(
            data=[np.arange(10)],
            events=[(0, 1), (2, -1), (2, 2), (3, -1)],
            fs=100,
            name_to_codes=[('A', 1), ('B', 2)],
            epoch_ival_ms=[0, 20],
            name_to_stop_codes=[('A', -1), ('B', -2)],
            pad_to_n_samples=None,
            expected_X=[[[0, 1, 2, 3]], ],
            expected_y=[0, ])

def check_add_breaks(
        events,
        fs,
        break_start_code,
        break_stop_code,
        name_to_start_codes,
        name_to_stop_codes,
        min_break_length_ms,
        max_break_length_ms,
        break_start_offset_ms,
        break_stop_offset_ms,
        expected_events,
        ):
    events = np.array(events)
    name_to_start_codes = OrderedDict(name_to_start_codes)
    name_to_stop_codes = OrderedDict(name_to_stop_codes)
    events_with_breaks = add_breaks(
        events, fs, break_start_code, break_stop_code, name_to_start_codes,
        name_to_stop_codes, min_break_length_ms=min_break_length_ms,
        max_break_length_ms=max_break_length_ms,
        break_start_offset_ms=break_start_offset_ms, break_stop_offset_ms=break_stop_offset_ms)
    np.testing.assert_array_equal(events_with_breaks,
                                 expected_events)


def test_add_breaks_no_break():
    check_add_breaks(
        events=np.array([(0, 1), (2, -1)]),
        fs=100,
        break_start_code=-3,
        break_stop_code=-4,
        name_to_start_codes=[('A', 1), ],
        name_to_stop_codes=[('A', -1), ],
        min_break_length_ms=None,
        max_break_length_ms=None,
        break_start_offset_ms=None,
        break_stop_offset_ms=None,
        expected_events=[(0, 1), (2, -1), ])


def test_add_breaks_one_break():
    # a break added!
    check_add_breaks(
        events=np.array([(0, 1), (2, -1), (5, 2)]),
        fs=100,
        break_start_code=-3,
        break_stop_code=-4,
        name_to_start_codes=[('A', 1), ('B', 2), ],
        name_to_stop_codes=[('A', -1), ],
        min_break_length_ms=None,
        max_break_length_ms=None,
        break_start_offset_ms=None,
        break_stop_offset_ms=None,
        expected_events=[(0, 1), (2, -1), (3, -3), (4, -4), (5, 2), ])


def test_add_breaks_break_within_bound():
    # a break added within bounds!
    check_add_breaks(
        events=np.array([(0, 1), (2, -1), (5, 2)]),
        fs=100,
        break_start_code=-3,
        break_stop_code=-4,
        name_to_start_codes=[('A', 1), ('B', 2), ],
        name_to_stop_codes=[('A', -1), ],
        min_break_length_ms=10,
        max_break_length_ms=None,
        break_start_offset_ms=None,
        break_stop_offset_ms=None,
        expected_events=[(0, 1), (2, -1), (3, -3), (4, -4), (5, 2), ])


def test_add_breaks_too_short():
    # not added, too short!
    check_add_breaks(
        events=np.array([(0, 1), (2, -1), (5, 2)]),
        fs=100,
        break_start_code=-3,
        break_stop_code=-4,
        name_to_start_codes=[('A', 1), ('B', 2), ],
        name_to_stop_codes=[('A', -1), ],
        min_break_length_ms=20,
        max_break_length_ms=None,
        break_start_offset_ms=None,
        break_stop_offset_ms=None,
        expected_events=[(0, 1), (2, -1), (5, 2), ])

def test_add_breaks_too_long():
    # a break added within both upper and lower bound!
    check_add_breaks(
        events=np.array([(0, 1), (2, -1), (6, 2)]),
        fs=100,
        break_start_code=-3,
        break_stop_code=-4,
        name_to_start_codes=[('A', 1), ('B', 2), ],
        name_to_stop_codes=[('A', -1), ],
        min_break_length_ms=None,
        max_break_length_ms=10,
        break_start_offset_ms=None,
        break_stop_offset_ms=None,
        expected_events=[(0, 1), (2, -1), (6, 2), ])


def test_add_breaks_within_both_bounds():
    # a break added within both upper and lower bound!
    check_add_breaks(
        events=np.array([(0, 1), (2, -1), (5, 2)]),
        fs=100,
        break_start_code=-3,
        break_stop_code=-4,
        name_to_start_codes=[('A', 1), ('B', 2), ],
        name_to_stop_codes=[('A', -1), ],
        min_break_length_ms=10,
        max_break_length_ms=30,
        break_start_offset_ms=None,
        break_stop_offset_ms=None,
        expected_events=[(0, 1), (2, -1), (3, -3), (4, -4), (5, 2), ])

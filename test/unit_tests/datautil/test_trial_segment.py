from collections import OrderedDict

import numpy as np

from braindecode.datautil.trial_segment import (
    create_cnt_y_and_start_stop_samples)


def check_cnt_y_start_stop_samples(n_samples, events, fs, epoch_ival_ms,
                                   name_to_start_codes,
                                   name_to_stop_codes, cnt_y, start_stop):

    cnt_y = np.array(cnt_y).T
    real_cnt_y, real_start_stop = create_cnt_y_and_start_stop_samples(
        n_samples, events ,fs, name_to_start_codes, epoch_ival_ms, name_to_stop_codes)
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

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

from collections import OrderedDict

import numpy as np
import pytest

from braindecode.datasets import MOABBDataset
from braindecode.datautil.preprocess import preprocess, zscore, scale, \
    MNEPreproc, NumpyPreproc
from braindecode.datautil.preprocess import (
    exponential_moving_demean, exponential_moving_standardize)
from braindecode.datautil.windowers import create_fixed_length_windows


@pytest.fixture(scope="module")
def base_concat_ds():
    return MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1, 2])


@pytest.fixture(scope='module')
def windows_concat_ds(base_concat_ds):
    return create_fixed_length_windows(
        base_concat_ds, start_offset_samples=100, stop_offset_samples=0,
        window_size_samples=1000, window_stride_samples=1000,
        drop_last_window=True, mapping=None, preload=True)


def test_not_list():
    with pytest.raises(AssertionError):
        preprocess(None, {'test': 1})


def test_no_raw_or_epochs():
    class EmptyDataset(object):
        def __init__(self):
            self.datasets = [1, 2, 3]

    ds = EmptyDataset()
    with pytest.raises(AssertionError):
        preprocess(ds, ["dummy", "dummy"])


def test_method_not_available(base_concat_ds):
    preprocessors = [MNEPreproc('this_method_is_not_real', )]
    with pytest.raises(AttributeError):
        preprocess(base_concat_ds, preprocessors)


def test_transform_base_method(base_concat_ds):
    preprocessors = [MNEPreproc("resample", sfreq=50)]
    preprocess(base_concat_ds, preprocessors)
    assert base_concat_ds.datasets[0].raw.info['sfreq'] == 50


def test_transform_windows_method(windows_concat_ds):
    preprocessors = [MNEPreproc("filter", l_freq=7, h_freq=13)]
    raw_window = windows_concat_ds[0][0]
    preprocess(windows_concat_ds, preprocessors)
    assert not np.array_equal(raw_window, windows_concat_ds[0][0])


def test_zscore_continuous(base_concat_ds):
    preprocessors = [
        MNEPreproc('pick_types', eeg=True, meg=False, stim=False),
        MNEPreproc('apply_function', fun=zscore, channel_wise=True)
    ]
    preprocess(base_concat_ds, preprocessors)
    for ds in base_concat_ds.datasets:
        raw_data = ds.raw.get_data()
        shape = raw_data.shape
        # zero mean
        expected = np.zeros(shape[:-1])
        np.testing.assert_allclose(
            raw_data.mean(axis=-1), expected, rtol=1e-4, atol=1e-4)
        # unit variance
        expected = np.ones(shape[:-1])
        np.testing.assert_allclose(
            raw_data.std(axis=-1), expected, rtol=1e-4, atol=1e-4)


def test_zscore_windows(windows_concat_ds):
    preprocessors = [
        MNEPreproc('pick_types', eeg=True, meg=False, stim=False),
        MNEPreproc(zscore, )
    ]
    preprocess(windows_concat_ds, preprocessors)
    for ds in windows_concat_ds.datasets:
        windowed_data = ds.windows.get_data()
        shape = windowed_data.shape
        # zero mean
        expected = np.zeros(shape[:-1])
        np.testing.assert_allclose(
            windowed_data.mean(axis=-1), expected, rtol=1e-4, atol=1e-4)
        # unit variance
        expected = np.ones(shape[:-1])
        np.testing.assert_allclose(
            windowed_data.std(axis=-1), expected, rtol=1e-4, atol=1e-4)


def test_scale_continuous(base_concat_ds):
    factor = 1e6
    preprocessors = [
        MNEPreproc('pick_types', eeg=True, meg=False, stim=False),
        NumpyPreproc(scale, factor=factor)
    ]
    raw_timepoint = base_concat_ds[0][0]
    preprocess(base_concat_ds, preprocessors)
    expected = np.ones_like(raw_timepoint) * factor
    np.testing.assert_allclose(base_concat_ds[0][0] / raw_timepoint, expected,
                               rtol=1e-4, atol=1e-4)


def test_scale_windows(windows_concat_ds):
    factor = 1e6
    preprocessors = [
        MNEPreproc('pick_types', eeg=True, meg=False, stim=False),
        MNEPreproc(scale, factor=factor)
    ]
    raw_window = windows_concat_ds[0][0]
    preprocess(windows_concat_ds, preprocessors)
    expected = np.ones_like(raw_window) * factor
    np.testing.assert_allclose(windows_concat_ds[0][0] / raw_window, expected,
                               rtol=1e-4, atol=1e-4)



@pytest.fixture(scope="module")
def mock_data():
    np.random.seed(20200217)
    mock_input = np.random.rand(2, 10).reshape(2, 10)
    expected_standardized = np.array(
        [[ 0.        , -1.41385996, -1.67770482,  1.95328935,  0.61618697,
          -0.55294099, -1.08890304,  1.04546089, -1.368485  , -1.08669994],
         [ 0.        , -1.41385996, -0.41117774,  1.65212819, -0.5392431 ,
          -0.23009334,  0.15087203, -1.45238971,  1.88407553, -0.38583499]])
    expected_demeaned = np.array(
        [[ 0.        , -0.02547392, -0.10004415,  0.47681459,  0.1399319 ,
          -0.11764405, -0.23535964,  0.22749205, -0.3155749 , -0.25316515],
         [ 0.        , -0.29211105, -0.07138808,  0.44137798, -0.13274718,
          -0.0519248 ,  0.03156507, -0.33137195,  0.52134583, -0.1020266 ]])
    return mock_input, expected_standardized, expected_demeaned


def test_exponential_running_standardize(mock_data):
    mock_input, expected_data, _ = mock_data
    standardized_data = exponential_moving_standardize(mock_input)
    assert mock_input.shape == standardized_data.shape == expected_data.shape
    np.testing.assert_allclose(
        standardized_data, expected_data, rtol=1e-4, atol=1e-4)


def test_exponential_running_demean(mock_data):
    mock_input, _, expected_data = mock_data
    demeaned_data = exponential_moving_demean(mock_input)
    assert mock_input.shape == demeaned_data.shape == expected_data.shape
    np.testing.assert_allclose(
        demeaned_data, expected_data, rtol=1e-4, atol=1e-4)


def test_exponential_running_init_block_size(mock_data):
    mock_input, _, _ = mock_data
    init_block_size = 3
    standardized_data = exponential_moving_standardize(
        mock_input, init_block_size=init_block_size)
    np.testing.assert_allclose(
        standardized_data[:, :init_block_size].sum(), [0], rtol=1e-4, atol=1e-4)

    demeaned_data = exponential_moving_demean(
        mock_input, init_block_size=init_block_size)
    np.testing.assert_allclose(
        demeaned_data[:, :init_block_size].sum(), [0], rtol=1e-4, atol=1e-4)

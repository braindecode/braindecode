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

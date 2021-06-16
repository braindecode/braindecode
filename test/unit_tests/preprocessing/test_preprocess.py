# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import copy

import pytest
import numpy as np

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing.preprocess import preprocess, zscore, scale, \
    Preprocessor, filterbank, exponential_moving_demean, \
    exponential_moving_standardize, MNEPreproc, NumpyPreproc
from braindecode.preprocessing.windowers import create_fixed_length_windows


# We can't use fixtures with scope='module' as the dataset objects are modified
# inplace during preprocessing. To avoid the long setup time caused by calling
# the dataset/windowing functions multiple times, we instantiate the dataset
# objects once and deep-copy them in fixture.
raw_ds = MOABBDataset(dataset_name='BNCI2014001', subject_ids=[1, 2])
windows_ds = create_fixed_length_windows(
    raw_ds, start_offset_samples=100, stop_offset_samples=None,
    window_size_samples=1000, window_stride_samples=1000,
    drop_last_window=True, mapping=None, preload=True)


@pytest.fixture
def base_concat_ds():
    return copy.deepcopy(raw_ds)


@pytest.fixture
def windows_concat_ds():
    return copy.deepcopy(windows_ds)


def modify_windows_object(epochs, factor=1):
    epochs._data *= factor


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


def test_deprecated_preprocs(base_concat_ds):
    msg1 = 'MNEPreproc is deprecated. Use Preprocessor with ' \
           '`apply_on_array=False` instead.'
    msg2 = 'NumpyPreproc is deprecated. Use Preprocessor with ' \
           '`apply_on_array=True` instead.'
    with pytest.warns(UserWarning, match=msg1):
        mne_preproc = MNEPreproc('pick_types', eeg=True, meg=False, stim=False)
    factor = 1e6
    with pytest.warns(UserWarning, match=msg2):
        np_preproc = NumpyPreproc(scale, factor=factor)

    raw_timepoint = base_concat_ds[0][0][:22]  # only keep EEG channels
    preprocess(base_concat_ds, [mne_preproc, np_preproc])
    np.testing.assert_allclose(base_concat_ds[0][0], raw_timepoint * factor,
                               rtol=1e-4, atol=1e-4)


def test_method_not_available(base_concat_ds):
    preprocessors = [Preprocessor('this_method_is_not_real', )]
    with pytest.raises(AttributeError):
        preprocess(base_concat_ds, preprocessors)


def test_preprocess_raw_str(base_concat_ds):
    preprocessors = [Preprocessor('crop', tmax=10, include_tmax=False)]
    preprocess(base_concat_ds, preprocessors)
    assert len(base_concat_ds.datasets[0].raw.times) == 2500


def test_preprocess_windows_str(windows_concat_ds):
    preprocessors = [
        Preprocessor('crop', tmin=0, tmax=0.1, include_tmax=False)]
    preprocess(windows_concat_ds, preprocessors)
    assert windows_concat_ds[0][0].shape[1] == 25


def test_preprocess_raw_callable_on_array(base_concat_ds):
    # Case tested in test_zscore_continuous
    pass


def test_preprocess_windows_callable_on_array(windows_concat_ds):
    # Case tested in test_zscore_windows
    pass


def test_preprocess_raw_callable_on_object(base_concat_ds):
    # Case tested in test_filterbank
    pass


def test_preprocess_windows_callable_on_object(windows_concat_ds):
    factor = 10
    preprocessors = [Preprocessor(modify_windows_object, apply_on_array=False,
                                  factor=factor)]
    raw_window = windows_concat_ds[0][0]
    preprocess(windows_concat_ds, preprocessors)
    np.testing.assert_allclose(windows_concat_ds[0][0], raw_window * factor,
                               rtol=1e-4, atol=1e-4)


def test_zscore_continuous(base_concat_ds):
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(zscore, channel_wise=True)
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
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(zscore)
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
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(scale, factor=factor)
    ]
    raw_timepoint = base_concat_ds[0][0][:22]  # only keep EEG channels
    preprocess(base_concat_ds, preprocessors)
    np.testing.assert_allclose(base_concat_ds[0][0], raw_timepoint * factor,
                               rtol=1e-4, atol=1e-4)


def test_scale_windows(windows_concat_ds):
    factor = 1e6
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(scale, factor=factor)
    ]
    raw_window = windows_concat_ds[0][0][:22]  # only keep EEG channels
    preprocess(windows_concat_ds, preprocessors)
    np.testing.assert_allclose(windows_concat_ds[0][0], raw_window * factor,
                               rtol=1e-4, atol=1e-4)


@pytest.fixture(scope='module')
def mock_data():
    mock_input = np.random.RandomState(20200217).rand(2, 10).reshape(2, 10)
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
    # mean over time axis (1!) should give 0 per channel
    np.testing.assert_allclose(
        standardized_data[:, :init_block_size].mean(axis=1), 0,
        rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        standardized_data[:, :init_block_size].std(axis=1), 1,
        rtol=1e-4, atol=1e-4)

    # mean over time axis (1!) should give 0 per channel
    demeaned_data = exponential_moving_demean(
        mock_input, init_block_size=init_block_size)
    np.testing.assert_allclose(
        demeaned_data[:, :init_block_size].mean(axis=1), 0, rtol=1e-4,
        atol=1e-4)


def test_filterbank(base_concat_ds):
    base_concat_ds = base_concat_ds.split([[0]])['0']
    preprocessors = [
        Preprocessor('pick_channels', ch_names=sorted(['C4', 'Cz']),
                     ordered=True),
        Preprocessor(filterbank, frequency_bands=[(0, 4), (4, 8), (8, 13)],
                     drop_original_signals=False, apply_on_array=False)
    ]
    preprocess(base_concat_ds, preprocessors)
    for x, y in base_concat_ds:
        break
    assert x.shape[0] == 8
    freq_band_annots = [
        ch.split('_')[-1]
        for ch in base_concat_ds.datasets[0].raw.ch_names
        if '_' in ch]
    assert len(np.unique(freq_band_annots)) == 3
    np.testing.assert_array_equal(base_concat_ds.datasets[0].raw.ch_names, [
        'C4', 'C4_0-4', 'C4_4-8', 'C4_8-13',
        'Cz', 'Cz_0-4', 'Cz_4-8', 'Cz_8-13',
    ])


def test_filterbank_order_channels_by_freq(base_concat_ds):
    base_concat_ds = base_concat_ds.split([[0]])['0']
    preprocessors = [
        Preprocessor(
            'pick_channels', ch_names=sorted(['C4', 'Cz']), ordered=True),
        Preprocessor(
            filterbank, frequency_bands=[(0, 4), (4, 8), (8, 13)],
            drop_original_signals=False, order_by_frequency_band=True,
            apply_on_array=False)]
    preprocess(base_concat_ds, preprocessors)
    np.testing.assert_array_equal(base_concat_ds.datasets[0].raw.ch_names, [
        'C4', 'Cz', 'C4_0-4', 'Cz_0-4',
        'C4_4-8', 'Cz_4-8', 'C4_8-13', 'Cz_8-13'
    ])

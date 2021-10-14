# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import os
import copy
from glob import glob

import mne
import pandas as pd
import pytest
import numpy as np

from braindecode.datasets import MOABBDataset, BaseConcatDataset, BaseDataset
from braindecode.preprocessing.preprocess import (
    preprocess, zscore, Preprocessor, filterbank, exponential_moving_demean,
    exponential_moving_standardize, MNEPreproc, NumpyPreproc, _replace_inplace,
    _set_preproc_kwargs)
from braindecode.preprocessing.preprocess import scale as deprecated_scale
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.datautil.serialization import load_concat_dataset


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
    msg1 = 'Class MNEPreproc is deprecated; will be removed in 0.7.0. Use ' \
           'Preprocessor with `apply_on_array=False` instead.'
    msg2 = 'NumpyPreproc is deprecated; will be removed in 0.7.0. Use ' \
           'Preprocessor with `apply_on_array=True` instead.'
    with pytest.warns(FutureWarning, match=msg1):
        mne_preproc = MNEPreproc('pick_types', eeg=True, meg=False, stim=False)
    factor = 1e6
    with pytest.warns(FutureWarning, match=msg2):
        np_preproc = NumpyPreproc(deprecated_scale, factor=factor)

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
    assert all([ds.raw_preproc_kwargs == [
        ('crop', {'tmax': 10, 'include_tmax': False}),
    ] for ds in base_concat_ds.datasets])


def test_preprocess_windows_str(windows_concat_ds):
    preprocessors = [
        Preprocessor('crop', tmin=0, tmax=0.1, include_tmax=False)]
    preprocess(windows_concat_ds, preprocessors)
    assert windows_concat_ds[0][0].shape[1] == 25
    assert all([ds.window_preproc_kwargs == [
        ('crop', {'tmin': 0, 'tmax': 0.1, 'include_tmax': False}),
    ] for ds in windows_concat_ds.datasets])


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


def test_zscore_deprecated():
    msg = 'Function zscore is deprecated; will be removed in 0.7.0. Use ' \
          'sklearn.preprocessing.scale instead.'
    with pytest.warns(FutureWarning, match=msg):
        zscore(np.random.rand(2, 2))


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
    assert all([ds.raw_preproc_kwargs == [
        ('pick_types', {'eeg': True, 'meg': False, 'stim': False}),
        ('zscore', {}),
    ] for ds in base_concat_ds.datasets])


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
    assert all([ds.window_preproc_kwargs == [
        ('pick_types', {'eeg': True, 'meg': False, 'stim': False}),
        ('zscore', {}),
    ] for ds in windows_concat_ds.datasets])


def test_scale_deprecated():
    msg = 'Function scale is deprecated; will be removed in 0.7.0. Use ' \
          'numpy.multiply instead.'
    with pytest.warns(FutureWarning, match=msg):
        deprecated_scale(np.random.rand(2, 2), factor=2)


def test_scale_continuous(base_concat_ds):
    factor = 1e6
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(deprecated_scale, factor=factor)
    ]
    raw_timepoint = base_concat_ds[0][0][:22]  # only keep EEG channels
    preprocess(base_concat_ds, preprocessors)
    np.testing.assert_allclose(base_concat_ds[0][0], raw_timepoint * factor,
                               rtol=1e-4, atol=1e-4)
    assert all([ds.raw_preproc_kwargs == [
        ('pick_types', {'eeg': True, 'meg': False, 'stim': False}),
        ('scale', {'factor': 1e6}),
    ] for ds in base_concat_ds.datasets])


def test_scale_windows(windows_concat_ds):
    factor = 1e6
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(deprecated_scale, factor=factor)
    ]
    raw_window = windows_concat_ds[0][0][:22]  # only keep EEG channels
    preprocess(windows_concat_ds, preprocessors)
    np.testing.assert_allclose(windows_concat_ds[0][0], raw_window * factor,
                               rtol=1e-4, atol=1e-4)
    assert all([ds.window_preproc_kwargs == [
        ('pick_types', {'eeg': True, 'meg': False, 'stim': False}),
        ('scale', {'factor': 1e6}),
    ] for ds in windows_concat_ds.datasets])


@pytest.fixture(scope='module')
def mock_data():
    mock_input = np.random.RandomState(20200217).rand(2, 10).reshape(2, 10)
    expected_standardized = np.array(
        [[ 0.        , -1.41385996, -1.67770482,  1.95328935,  0.61618697,    # noqa: E201,E203,E241,E501
          -0.55294099, -1.08890304,  1.04546089, -1.368485  , -1.08669994],   # noqa: E201,E203,E241,E128,E501
         [ 0.        , -1.41385996, -0.41117774,  1.65212819, -0.5392431 ,    # noqa: E201,E203,E241,E128,E501
          -0.23009334,  0.15087203, -1.45238971,  1.88407553, -0.38583499]])  # noqa: E201,E203,E241,E128,E501
    expected_demeaned = np.array(
        [[ 0.        , -0.02547392, -0.10004415,  0.47681459,  0.1399319 ,    # noqa: E201,E203,E241,E501
          -0.11764405, -0.23535964,  0.22749205, -0.3155749 , -0.25316515],   # noqa: E201,E203,E241,E128,E501
         [ 0.        , -0.29211105, -0.07138808,  0.44137798, -0.13274718,    # noqa: E201,E203,E241,E128,E501
          -0.0519248 ,  0.03156507, -0.33137195,  0.52134583, -0.1020266 ]])  # noqa: E201,E202,E203,E241,E128,E501
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
    assert all([ds.raw_preproc_kwargs == [
        ('pick_channels', {'ch_names': ['C4', 'Cz'], 'ordered': True}),
        ('filterbank', {'frequency_bands': [(0, 4), (4, 8), (8, 13)],
                        'drop_original_signals': False}),
    ] for ds in base_concat_ds.datasets])


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
    assert all([ds.raw_preproc_kwargs == [
        ('pick_channels', {'ch_names': ['C4', 'Cz'], 'ordered': True}),
        ('filterbank', {'frequency_bands': [(0, 4), (4, 8), (8, 13)],
                        'drop_original_signals': False,
                        'order_by_frequency_band': True}),
    ] for ds in base_concat_ds.datasets])


def test_replace_inplace(base_concat_ds):
    base_concat_ds2 = copy.deepcopy(base_concat_ds)
    for i in range(len(base_concat_ds2.datasets)):
        base_concat_ds2.datasets[i].raw.crop(0, 10, include_tmax=False)
    _replace_inplace(base_concat_ds, base_concat_ds2)

    assert all([len(ds.raw.times) == 2500 for ds in base_concat_ds.datasets])


def test_set_raw_preproc_kwargs(base_concat_ds):
    raw_preproc_kwargs = [('crop', {'tmax': 10, 'include_tmax': False})]
    preprocessors = [Preprocessor('crop', tmax=10, include_tmax=False)]
    ds = base_concat_ds.datasets[0]
    _set_preproc_kwargs(ds, preprocessors)

    assert hasattr(ds, 'raw_preproc_kwargs')
    assert ds.raw_preproc_kwargs == raw_preproc_kwargs


def test_set_window_preproc_kwargs(windows_concat_ds):
    window_preproc_kwargs = [('crop', {'tmax': 10, 'include_tmax': False})]
    preprocessors = [Preprocessor('crop', tmax=10, include_tmax=False)]
    ds = windows_concat_ds.datasets[0]
    _set_preproc_kwargs(ds, preprocessors)

    assert hasattr(ds, 'window_preproc_kwargs')
    assert ds.window_preproc_kwargs == window_preproc_kwargs


def test_set_preproc_kwargs_wrong_type(base_concat_ds):
    preprocessors = [Preprocessor('crop', tmax=10, include_tmax=False)]
    with pytest.raises(TypeError):
        _set_preproc_kwargs(base_concat_ds, preprocessors)


@pytest.mark.parametrize('kind', ['raw', 'windows'])
@pytest.mark.parametrize('save', [True, False])
@pytest.mark.parametrize('overwrite', [True, False])
@pytest.mark.parametrize('n_jobs', [1, 2, None])
def test_preprocess_save_dir(base_concat_ds, windows_concat_ds, tmp_path,
                             kind, save, overwrite, n_jobs):
    preproc_kwargs = [
        ('crop', {'tmin': 0, 'tmax': 0.1, 'include_tmax': False})]
    preprocessors = [
        Preprocessor('crop', tmin=0, tmax=0.1, include_tmax=False)]

    save_dir = str(tmp_path) if save else None
    if kind == 'raw':
        concat_ds = base_concat_ds
        preproc_kwargs_name = 'raw_preproc_kwargs'
    elif kind == 'windows':
        concat_ds = windows_concat_ds
        preproc_kwargs_name = 'window_preproc_kwargs'

    concat_ds = preprocess(
        concat_ds, preprocessors, save_dir, overwrite=overwrite, n_jobs=n_jobs)

    assert all([hasattr(ds, preproc_kwargs_name) for ds in concat_ds.datasets])
    assert all([getattr(ds, preproc_kwargs_name) == preproc_kwargs
                for ds in concat_ds.datasets])
    assert all([len(getattr(ds, kind).times) == 25
                for ds in concat_ds.datasets])
    if kind == 'raw':
        assert all([hasattr(ds, 'target_name') for ds in concat_ds.datasets])

    if save_dir is None:
        assert all([getattr(ds, kind).preload
                    for ds in concat_ds.datasets])
    else:
        assert all([not getattr(ds, kind).preload
                    for ds in concat_ds.datasets])
        save_dirs = [os.path.join(save_dir, str(i))
                     for i in range(len(concat_ds.datasets))]
        assert set(glob(save_dir + '/*')) == set(save_dirs)


@pytest.mark.parametrize('overwrite', [True, False])
def test_preprocess_overwrite(base_concat_ds, tmp_path, overwrite):
    preprocessors = [Preprocessor('crop', tmax=10, include_tmax=False)]

    # Create temporary directory with preexisting files
    save_dir = str(tmp_path)
    for i, ds in enumerate(base_concat_ds.datasets):
        concat_ds = BaseConcatDataset([ds])
        save_subdir = os.path.join(save_dir, str(i))
        os.makedirs(save_subdir)
        concat_ds.save(save_subdir, overwrite=True)

    if overwrite:
        preprocess(base_concat_ds, preprocessors, save_dir, overwrite=True)
        # Make sure the serialized data is preprocessed
        preproc_concat_ds = load_concat_dataset(save_dir, True)
        assert all([len(ds.raw.times) == 2500
                    for ds in preproc_concat_ds.datasets])
    else:
        with pytest.raises(FileExistsError):
            preprocess(base_concat_ds, preprocessors, save_dir,
                       overwrite=False)


def test_preprocessors_with_misc_channels():
    rng = np.random.RandomState(42)
    signal_sfreq = 50
    info = mne.create_info(ch_names=['0', '1', 'target_0', 'target_1'],
                           sfreq=signal_sfreq,
                           ch_types=['eeg', 'eeg', 'misc', 'misc'])
    signal = rng.randn(2, 1000)
    targets = rng.randn(2, 1000)
    raw = mne.io.RawArray(np.concatenate([signal, targets]), info=info)
    desc = pd.Series({'pathological': True, 'gender': 'M', 'age': 48})
    base_dataset = BaseDataset(raw, desc, target_name=None)
    concat_ds = BaseConcatDataset([base_dataset])
    preprocessors = [
        Preprocessor('pick_types', eeg=True, misc=True),
        Preprocessor(lambda x: x / 1e6),
    ]

    preprocess(concat_ds, preprocessors)

    # Check whether preprocessing has not affected the targets
    # This is only valid for preprocessors that use mne functions which do not modify
    # `misc` channels.
    np.testing.assert_array_equal(
        concat_ds.datasets[0].raw.get_data()[-2:, :],
        targets
    )

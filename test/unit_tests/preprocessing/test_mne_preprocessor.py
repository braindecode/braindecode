# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Bruna Lopes <brunajaflopes@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
# Adapting some tests from test_preprocess file
#
# License: BSD-3


import os
import copy
import platform
from glob import glob

import mne
import pandas as pd
import pytest
import numpy as np

from pytest_cases import parametrize_with_cases

from braindecode.datasets import MOABBDataset, BaseConcatDataset, BaseDataset
from braindecode.preprocessing.preprocess import (
    preprocess, Preprocessor, filterbank, exponential_moving_standardize,
    _replace_inplace, _set_preproc_kwargs)
from braindecode.preprocessing.preprocess import scale as deprecated_scale
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.preprocessing import (
    Pick, Crop, Filter, Resample, DropChannels, SetEEGReference)

# We can't use fixtures with scope='module' as the dataset objects are modified
# inplace during preprocessing. To avoid the long setup time caused by calling
# the dataset/windowing functions multiple times, we instantiate the dataset
# objects once and deep-copy them in fixture.
raw_ds = MOABBDataset(dataset_name='BNCI2014001', subject_ids=[1, 2])
windows_ds = create_fixed_length_windows(
    raw_ds, start_offset_samples=100, stop_offset_samples=None,
    window_size_samples=1000, window_stride_samples=1000,
    drop_last_window=True, mapping=None, preload=True)


# Get the raw data in fixture
@pytest.fixture
def base_concat_ds():
    return copy.deepcopy(raw_ds)


@pytest.fixture
def windows_concat_ds():
    return copy.deepcopy(windows_ds)


def test_preprocess_raw_kwargs(base_concat_ds):
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    preprocess(base_concat_ds, preprocessors)
    assert len(base_concat_ds.datasets[0].raw.times) == 2500
    assert all([ds.raw_preproc_kwargs == [
        ('crop', {'tmax': 10, 'include_tmax': False}),
    ] for ds in base_concat_ds.datasets])


def test_preprocess_windows_kwargs(windows_concat_ds):
    preprocessors = [
        Crop(tmin=0, tmax=0.1, include_tmax=False)]
    preprocess(windows_concat_ds, preprocessors)
    assert windows_concat_ds[0][0].shape[1] == 25
    assert all([ds.window_preproc_kwargs == [
        ('crop', {'tmin': 0, 'tmax': 0.1, 'include_tmax': False}),
    ] for ds in windows_concat_ds.datasets])


def test_scale_deprecated():
    msg = 'Function scale is deprecated; will be removed in 0.8.0. ' \
          'Use numpy.multiply inside a lambda function instead.'
    with pytest.warns(FutureWarning, match=msg):
        deprecated_scale(np.random.rand(2, 2), factor=2)


# To test one preprocessor at each time, using this fixture structure
class PrepClasses:
    @pytest.mark.parametrize("sfreq", [100, 300])
    def prep_resample(self, sfreq):
        return Resample(sfreq=sfreq)

    @pytest.mark.parametrize("picks", ['eeg'])
    def prep_picktype(self, picks):
        return Pick(picks=picks)

    @pytest.mark.parametrize("picks", [['Cz'], ['C4', 'FC3']])
    def prep_pickchannels(self, picks):
        return Pick(picks=picks)

    @pytest.mark.parametrize("l_freq,h_freq", [(4, 30), (7, None), (None, 35)])
    def prep_filter(self, l_freq, h_freq):
        return Filter(l_freq=l_freq, h_freq=h_freq)

    @pytest.mark.parametrize("ref_channels", ['average', ['C4'], ['C4', 'Cz']])
    def prep_setref(self, ref_channels):
        return SetEEGReference(ref_channels=ref_channels)

    @pytest.mark.parametrize("tmin,tmax", [(0, .1), (.1, 1.2),
                                           (0.1, None)])
    def prep_crop(self, tmin, tmax):
        return Crop(tmin=tmin, tmax=tmax)

    @pytest.mark.parametrize("ch_names", ["Pz", "P2", "P1", "POz"])
    def prep_drop(self, ch_names):
        return DropChannels(ch_names=ch_names)


@parametrize_with_cases("prep", cases=PrepClasses, prefix="prep_")
def test_preprocessings(prep, base_concat_ds):
    preprocessors = [prep]
    preprocess(base_concat_ds, preprocessors, n_jobs=1)


def test_new_filterbank(base_concat_ds):
    base_concat_ds = base_concat_ds.split([[0]])['0']
    preprocessors = [
        Pick(picks=sorted(['C4', 'Cz'])),
        Preprocessor(fn=filterbank, frequency_bands=[(0, 4), (4, 8), (8, 13)],
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
        ('pick', {'picks': ['C4', 'Cz']}),
        ('filterbank', {'frequency_bands': [(0, 4), (4, 8), (8, 13)],
                        'drop_original_signals': False}),
    ] for ds in base_concat_ds.datasets])


def test_replace_inplace(base_concat_ds):
    base_concat_ds2 = copy.deepcopy(base_concat_ds)
    for i in range(len(base_concat_ds2.datasets)):
        base_concat_ds2.datasets[i].raw.crop(0, 10, include_tmax=False)
    _replace_inplace(base_concat_ds, base_concat_ds2)

    assert all([len(ds.raw.times) == 2500 for ds in base_concat_ds.datasets])


def test_set_raw_preproc_kwargs(base_concat_ds):
    raw_preproc_kwargs = [('crop', {'tmax': 10, 'include_tmax': False})]
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    ds = base_concat_ds.datasets[0]
    _set_preproc_kwargs(ds, preprocessors)

    assert hasattr(ds, 'raw_preproc_kwargs')
    assert ds.raw_preproc_kwargs == raw_preproc_kwargs


def test_set_window_preproc_kwargs(windows_concat_ds):
    window_preproc_kwargs = [('crop', {'tmax': 10, 'include_tmax': False})]
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    ds = windows_concat_ds.datasets[0]
    _set_preproc_kwargs(ds, preprocessors)

    assert hasattr(ds, 'window_preproc_kwargs')
    assert ds.window_preproc_kwargs == window_preproc_kwargs


def test_set_preproc_kwargs_wrong_type(base_concat_ds):
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    with pytest.raises(TypeError):
        _set_preproc_kwargs(base_concat_ds, preprocessors)


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Not supported on Windows")
@pytest.mark.parametrize('kind', ['raw', 'windows'])
@pytest.mark.parametrize('save', [True, False])
@pytest.mark.parametrize('overwrite', [True, False])
@pytest.mark.parametrize('n_jobs', [1, 2, None])
def test_preprocess_save_dir(base_concat_ds, windows_concat_ds, tmp_path,
                             kind, save, overwrite, n_jobs):
    preproc_kwargs = [
        ('crop', {'tmin': 0, 'tmax': 0.1, 'include_tmax': False})]
    preprocessors = [
        Crop(tmin=0, tmax=0.1, include_tmax=False)]

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


def test_mne_preprocessor(base_concat_ds):
    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 38.0  # high cut frequency for filtering

    preprocessors = [
        Resample(sfreq=100),
        Pick(picks=['eeg']),  # Keep EEG sensors
        Filter(l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize)
    ]

    preprocess(base_concat_ds, preprocessors, n_jobs=-1)


def test_new_eegref(base_concat_ds):
    preprocessors = [SetEEGReference(ref_channels='average')]
    preprocess(base_concat_ds, preprocessors, n_jobs=1)


def test_new_filterbank_order_channels_by_freq(base_concat_ds):
    base_concat_ds = base_concat_ds.split([[0]])['0']
    preprocessors = [
        # DropChannels(ch_names=["P2", "P1"]),
        Pick(picks=sorted(['C4', 'Cz'])),
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
        ('pick', {'picks': ['C4', 'Cz']}),
        ('filterbank', {'frequency_bands': [(0, 4), (4, 8), (8, 13)],
                        'drop_original_signals': False,
                        'order_by_frequency_band': True}),
    ] for ds in base_concat_ds.datasets])


# Test overwriting
@pytest.mark.parametrize('overwrite', [True, False])
def test_new_overwrite(base_concat_ds, tmp_path, overwrite):
    preprocessors = [
        Crop(tmax=10, include_tmax=False)]

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


def test_new_misc_channels():
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
        Pick(picks=['eeg', 'misc']),
        Preprocessor(lambda x: x / 1e6),
    ]

    preprocess(concat_ds, preprocessors)

    # Check whether preprocessing has not affected the targets
    # This is only valid for preprocessors that use mne functions which do not modify
    # `misc` channels.
    np.testing.assert_array_equal(
        concat_ds.datasets[0].raw.get_data()[-2:, :],
        targets)

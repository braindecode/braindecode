# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import os

import pytest
import numpy as np
import pandas as pd

from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing import (
    create_windows_from_events, Preprocessor, preprocess)
from braindecode.datautil.serialization import load_concat_dataset


@pytest.fixture()
def setup_concat_raw_dataset():
    return MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])


@pytest.fixture()
def setup_concat_windows_dataset(setup_concat_raw_dataset):
    moabb_dataset = setup_concat_raw_dataset
    return create_windows_from_events(
        concat_ds=moabb_dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0)


def test_outdated_save_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    with pytest.warns(
            UserWarning, match='This function only exists for '
                               'backwards compatibility purposes. DO NOT USE!'):
        concat_raw_dataset._outdated_save(path=tmpdir, overwrite=False)
    assert os.path.exists(tmpdir.join("description.json"))
    for raw_i in range(n_raw_datasets):
        assert os.path.exists(tmpdir.join(f"{raw_i}-raw.fif"))
    assert not os.path.exists(tmpdir.join(f"{n_raw_datasets}-raw.fif"))


def test_outdated_save_concat_windows_dataset(
        setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    with pytest.warns(
            UserWarning, match='This function only exists for '
                               'backwards compatibility purposes. DO NOT USE!'):
        concat_windows_dataset._outdated_save(path=tmpdir, overwrite=False)
    assert os.path.exists(tmpdir.join("description.json"))
    for windows_i in range(n_windows_datasets):
        assert os.path.exists(tmpdir.join(f"{windows_i}-epo.fif"))
    assert not os.path.exists(tmpdir.join(f"{n_windows_datasets}-epo.fif"))


def test_load_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    with pytest.warns(
            UserWarning, match='This function only exists for '
                               'backwards compatibility purposes. DO NOT USE!'):
        concat_raw_dataset._outdated_save(path=tmpdir, overwrite=False)
    with pytest.warns(
            UserWarning, match="The way your dataset was saved is deprecated by"
                               " now. Please save it again using dataset.save()"
                               "."):
        loaded_concat_raw_dataset = load_concat_dataset(
            path=tmpdir, preload=False)
    assert len(concat_raw_dataset) == len(loaded_concat_raw_dataset)
    assert (len(concat_raw_dataset.datasets) ==
            len(loaded_concat_raw_dataset.datasets))
    assert (len(concat_raw_dataset.description) ==
            len(loaded_concat_raw_dataset.description))
    for raw_i in range(n_raw_datasets):
        actual_x, actual_y = concat_raw_dataset[raw_i]
        x, y = loaded_concat_raw_dataset[raw_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
    pd.testing.assert_frame_equal(
        concat_raw_dataset.description, loaded_concat_raw_dataset.description)


def test_load_concat_windows_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    with pytest.warns(
            UserWarning, match='This function only exists for '
                               'backwards compatibility purposes. DO NOT USE!'):
        concat_windows_dataset._outdated_save(path=tmpdir, overwrite=False)
    with pytest.warns(
            UserWarning, match="The way your dataset was saved is deprecated by"
                               " now. Please save it again using dataset.save()"
                               "."):
        loaded_concat_windows_dataset = load_concat_dataset(
            path=tmpdir, preload=False)
    assert len(concat_windows_dataset) == len(loaded_concat_windows_dataset)
    assert (len(concat_windows_dataset.datasets) ==
            len(loaded_concat_windows_dataset.datasets))
    assert (len(concat_windows_dataset.description) ==
            len(loaded_concat_windows_dataset.description))
    for windows_i in range(n_windows_datasets):
        actual_x, actual_y, actual_crop_inds = concat_windows_dataset[windows_i]
        x, y, crop_inds = loaded_concat_windows_dataset[windows_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y, actual_y, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(crop_inds, actual_crop_inds)
    pd.testing.assert_frame_equal(concat_windows_dataset.description,
                                  loaded_concat_windows_dataset.description)


def test_load_multiple_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    for i in range(2):
        path = os.path.join(tmpdir, str(i))
        os.makedirs(path)
        with pytest.warns(
                UserWarning, match='This function only exists for '
                                   'backwards compatibility purposes. DO NOT '
                                   'USE!'):
            concat_raw_dataset._outdated_save(path=path, overwrite=False)
        with pytest.warns(
                UserWarning, match="The way your dataset was saved is "
                                   "deprecated by now. Please save it again "
                                   "using dataset.save()."):
            loaded_concat_raw_datasets = load_concat_dataset(
                path=tmpdir, preload=False)
    assert 2 * len(concat_raw_dataset) == len(loaded_concat_raw_datasets)
    assert (2 * len(concat_raw_dataset.datasets) ==
            len(loaded_concat_raw_datasets.datasets))
    assert (2 * len(concat_raw_dataset.description) ==
            len(loaded_concat_raw_datasets.description))


def test_load_multiple_concat_windows_dataset(setup_concat_windows_dataset,
                                              tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    for i in range(2):
        path = os.path.join(tmpdir, str(i))
        os.makedirs(path)
        with pytest.warns(
                UserWarning, match='This function only exists for '
                                   'backwards compatibility purposes. DO NOT '
                                   'USE!'):
            concat_windows_dataset._outdated_save(path=path, overwrite=False)
        with pytest.warns(
                UserWarning, match="The way your dataset was saved is "
                                   "deprecated by now. Please save it again "
                                   "using dataset.save()."):
            loaded_concat_windows_datasets = load_concat_dataset(
                path=tmpdir, preload=False)
    assert 2 * len(concat_windows_dataset) == len(loaded_concat_windows_datasets)
    assert (2 * len(concat_windows_dataset.datasets) ==
            len(loaded_concat_windows_datasets.datasets))
    assert (2 * len(concat_windows_dataset.description) ==
            len(loaded_concat_windows_datasets.description))


def test_load_save_raw_preproc_kwargs(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    preprocess(concat_raw_dataset, [
        Preprocessor('pick_channels', ch_names=['C3']),
    ])
    concat_raw_dataset.save(tmpdir, overwrite=False)
    assert os.path.exists(os.path.join(tmpdir, 'raw_preproc_kwargs.json'))
    loaded_concat_raw_dataset = load_concat_dataset(tmpdir, preload=False)
    assert loaded_concat_raw_dataset.raw_preproc_kwargs == [
        ('pick_channels', {'ch_names': ['C3']}),
    ]


def test_load_save_window_preproc_kwargs(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    concat_windows_dataset.save(tmpdir, overwrite=False)
    assert os.path.exists(os.path.join(tmpdir, 'window_kwargs.json'))

    preprocess(concat_windows_dataset, [
        Preprocessor('pick_channels', ch_names=['Cz']),
    ])
    concat_windows_dataset.save(tmpdir, overwrite=True)
    assert os.path.exists(os.path.join(tmpdir, 'window_kwargs.json'))
    assert os.path.exists(os.path.join(tmpdir, 'window_preproc_kwargs.json'))
    loaded_concat_windows_dataset = load_concat_dataset(tmpdir, preload=False)
    assert loaded_concat_windows_dataset.window_kwargs == [
        ('create_windows_from_events', {
            'trial_start_offset_samples': 0, 'trial_stop_offset_samples': 0,
            'window_size_samples': None, 'window_stride_samples': None,
            'drop_last_window': False, 'mapping': None, 'preload': False,
            'drop_bad_windows': True, 'picks': None, 'reject': None,
            'flat': None, 'on_missing': 'error', 'n_jobs': 1})
    ]
    assert loaded_concat_windows_dataset.window_preproc_kwargs == [
        ('pick_channels', {'ch_names': ['Cz']}),
    ]


def test_save_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns(None) as raised_warnings:
        concat_raw_dataset.save(path=tmpdir, overwrite=False)
        assert len(raised_warnings) == 0
    for raw_i in range(n_raw_datasets):
        assert os.path.exists(os.path.join(tmpdir, f"{raw_i}", "description.json"))
        assert os.path.exists(os.path.join(tmpdir, f"{raw_i}", f"{raw_i}-raw.fif"))
    assert not os.path.exists(os.path.join(tmpdir, f"{n_raw_datasets}-raw.fif"))


def test_save_concat_windows_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns(None) as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=False)
        assert len(raised_warnings) == 0
    for windows_i in range(n_windows_datasets):
        assert os.path.exists(os.path.join(tmpdir, f"{windows_i}", "description.json"))
        assert os.path.exists(os.path.join(tmpdir, f"{windows_i}", f"{windows_i}-epo.fif"))
    assert not os.path.exists(os.path.join(tmpdir, f"{n_windows_datasets}-epo.fif"))


def test_load_concat_raw_dataset_parallel(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns(None) as raised_warnings:
        concat_raw_dataset.save(path=tmpdir, overwrite=False)
        assert len(raised_warnings) == 0
    # assert no warning raised with loading dataset saved in 'new' way
    with pytest.warns(None) as raised_warnings:
        loaded_concat_raw_dataset = load_concat_dataset(
            path=tmpdir, preload=False, n_jobs=2)
        assert len(raised_warnings) == 0
    assert len(concat_raw_dataset) == len(loaded_concat_raw_dataset)
    assert (len(concat_raw_dataset.datasets) ==
            len(loaded_concat_raw_dataset.datasets))
    assert (len(concat_raw_dataset.description) ==
            len(loaded_concat_raw_dataset.description))
    for raw_i in range(n_raw_datasets):
        actual_x, actual_y = concat_raw_dataset[raw_i]
        x, y = loaded_concat_raw_dataset[raw_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
    pd.testing.assert_frame_equal(
        concat_raw_dataset.description, loaded_concat_raw_dataset.description)


def test_load_concat_windows_dataset_parallel(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns(None) as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=False)
        assert len(raised_warnings) == 0
    # assert warning raised because of n_jobs not supported with mne.Epochs
    with pytest.warns(UserWarning, match='Parallelized reading with '
                                         '`preload=False` is not supported for '
                                         'windowed data. Will use `n_jobs=1`.'):
        loaded_concat_windows_dataset = load_concat_dataset(
            path=tmpdir, preload=False, n_jobs=2)
        assert len(raised_warnings) == 0
    assert len(concat_windows_dataset) == len(loaded_concat_windows_dataset)
    assert (len(concat_windows_dataset.datasets) ==
            len(loaded_concat_windows_dataset.datasets))
    assert (len(concat_windows_dataset.description) ==
            len(loaded_concat_windows_dataset.description))
    for windows_i in range(n_windows_datasets):
        actual_x, actual_y, actual_crop_inds = concat_windows_dataset[windows_i]
        x, y, crop_inds = loaded_concat_windows_dataset[windows_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y, actual_y, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(crop_inds, actual_crop_inds)
    pd.testing.assert_frame_equal(concat_windows_dataset.description,
                                  loaded_concat_windows_dataset.description)

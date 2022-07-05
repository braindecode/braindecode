# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import os

import pytest
import numpy as np
import pandas as pd

from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.preprocessing import (
    create_windows_from_events, Preprocessor, preprocess)
from braindecode.datautil.serialization import (
    load_concat_dataset, _check_save_dir_empty)


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
    for i in range(len(concat_raw_dataset.datasets)):
        assert os.path.exists(os.path.join(tmpdir, str(i), 'raw_preproc_kwargs.json'))
    loaded_concat_raw_dataset = load_concat_dataset(tmpdir, preload=False)
    for ds in loaded_concat_raw_dataset.datasets:
        assert ds.raw_preproc_kwargs == [
            ('pick_channels', {'ch_names': ['C3']}),
        ]


def test_load_save_window_preproc_kwargs(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    concat_windows_dataset.save(tmpdir, overwrite=False)
    for i in range(len(concat_windows_dataset.datasets)):
        subdir = os.path.join(tmpdir, str(i))
        assert os.path.exists(os.path.join(subdir, 'window_kwargs.json'))

    preprocess(concat_windows_dataset, [
        Preprocessor('pick_channels', ch_names=['Cz']),
    ])
    concat_windows_dataset.save(tmpdir, overwrite=True)
    for i in range(len(concat_windows_dataset.datasets)):
        subdir = os.path.join(tmpdir, str(i))
        assert os.path.exists(os.path.join(subdir, 'window_kwargs.json'))
        assert os.path.exists(os.path.join(subdir, 'window_preproc_kwargs.json'))
    loaded_concat_windows_dataset = load_concat_dataset(tmpdir, preload=False)

    for ds in loaded_concat_windows_dataset.datasets:
        assert ds.window_kwargs == [
            ('create_windows_from_events', {
                'infer_mapping': True, 'infer_window_size_stride': True,
                'trial_start_offset_samples': 0, 'trial_stop_offset_samples': 0,
                'window_size_samples': None, 'window_stride_samples': None,
                'drop_last_window': False, 'mapping':  {
                    'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3},
                'preload': False, 'drop_bad_windows': True, 'picks': None,
                'reject': None, 'flat': None, 'on_missing': 'error',
                'accepted_bads_ratio': 0.0})
        ]
        assert ds.window_preproc_kwargs == [
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
        subdir = os.path.join(tmpdir, str(raw_i))
        assert os.path.exists(os.path.join(subdir, "description.json"))
        assert os.path.exists(os.path.join(subdir, f"{raw_i}-raw.fif"))
    assert not os.path.exists(os.path.join(tmpdir, f"{n_raw_datasets}"))


def test_save_concat_windows_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns(None) as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=False)
        assert len(raised_warnings) == 0
    for windows_i in range(n_windows_datasets):
        subdir = os.path.join(tmpdir, str(windows_i))
        assert os.path.exists(os.path.join(subdir, "description.json"))
        assert os.path.exists(os.path.join(subdir, f"{windows_i}-epo.fif"))
    assert not os.path.exists(os.path.join(tmpdir, f"{n_windows_datasets}"))


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


def test_save_varying_number_of_datasets_with_overwrite(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    concat_windows_dataset.save(path=tmpdir, overwrite=False)
    subset = concat_windows_dataset.split([0])['0']
    with pytest.warns(UserWarning, match='The number of saved datasets'):
        subset.save(path=tmpdir, overwrite=True)

    # assert no warning raised when there are as many subdirectories than before
    with pytest.warns(None) as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=True)
        assert len(raised_warnings) == 0

    # assert no warning raised when there are more subdirectories than before
    double_concat_windows_dataset = BaseConcatDataset(
        [concat_windows_dataset, concat_windows_dataset])
    with pytest.warns(None) as raised_warnings:
        double_concat_windows_dataset.save(path=tmpdir, overwrite=True)
        assert len(raised_warnings) == 0


def test_directory_contains_file(setup_concat_windows_dataset, tmpdir):
    with open(os.path.join(tmpdir, 'test.txt'), 'w') as f:
        f.write('test')
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.warns(UserWarning, match='Chosen directory'):
        concat_windows_dataset.save(tmpdir)


def test_other_subdirectories_exist(setup_concat_windows_dataset, tmpdir):
    os.mkdir(os.path.join(tmpdir, '999'))
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.warns(UserWarning, match='Chosen directory'):
        concat_windows_dataset.save(tmpdir)


def test_subdirectory_already_exist(setup_concat_windows_dataset, tmpdir):
    os.mkdir(os.path.join(tmpdir, '0'))
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.raises(FileExistsError, match='Subdirectory'):
        concat_windows_dataset.save(tmpdir)


def test_check_save_dir_empty(setup_concat_raw_dataset, tmpdir):
    _check_save_dir_empty(tmpdir)
    setup_concat_raw_dataset.save(tmpdir)
    with pytest.raises(FileExistsError):
        _check_save_dir_empty(tmpdir)

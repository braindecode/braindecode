# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3

import json
import os
import platform
import shutil
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import BaseConcatDataset, MOABBDataset, RawDataset
from braindecode.datautil.serialization import (
    _check_save_dir_empty,
    load_concat_dataset,
    save_concat_dataset,
)
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    preprocess,
)

bnci_kwargs = {
    "n_sessions": 2,
    "n_runs": 3,
    "n_subjects": 9,
    "paradigm": "imagery",
    "duration": 386.9,
    "sfreq": 250,
    "event_list": ("feet", "left_hand", "right_hand", "tongue"),
    "channels": ("C3", "Cz", "C2"),
}


@pytest.fixture()
def setup_concat_raw_dataset():
    return MOABBDataset(
        dataset_name="FakeDataset", subject_ids=[1], dataset_kwargs=bnci_kwargs
    )


@pytest.fixture()
def setup_concat_windows_dataset(setup_concat_raw_dataset):
    moabb_dataset = setup_concat_raw_dataset
    return create_windows_from_events(
        concat_ds=moabb_dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
    )


@pytest.fixture()
def setup_concat_windows_mne_dataset(setup_concat_raw_dataset):
    moabb_dataset = setup_concat_raw_dataset
    return create_windows_from_events(
        concat_ds=moabb_dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        picks=moabb_dataset.datasets[0].raw.ch_names,
    )


def test_outdated_save_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    with pytest.warns(
        UserWarning,
        match="This function only exists for "
        "backwards compatibility purposes. DO NOT USE!",
    ):
        concat_raw_dataset._outdated_save(path=tmpdir, overwrite=False)
    assert os.path.exists(tmpdir.join("description.json"))
    for raw_i in range(n_raw_datasets):
        assert os.path.exists(tmpdir.join(f"{raw_i}-raw.fif"))
    assert not os.path.exists(tmpdir.join(f"{n_raw_datasets}-raw.fif"))


def test_outdated_save_concat_windows_dataset(setup_concat_windows_mne_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_mne_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    with pytest.warns(
        UserWarning,
        match="This function only exists for "
        "backwards compatibility purposes. DO NOT USE!",
    ):
        concat_windows_dataset._outdated_save(path=tmpdir, overwrite=False)
    assert os.path.exists(tmpdir.join("description.json"))
    for windows_i in range(n_windows_datasets):
        assert os.path.exists(tmpdir.join(f"{windows_i}-epo.fif"))
    assert not os.path.exists(tmpdir.join(f"{n_windows_datasets}-epo.fif"))


def test_load_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    concat_raw_dataset.save(path=tmpdir, overwrite=False)
    loaded_concat_raw_dataset = load_concat_dataset(path=tmpdir, preload=False)
    assert len(concat_raw_dataset) == len(loaded_concat_raw_dataset)
    assert len(concat_raw_dataset.datasets) == len(loaded_concat_raw_dataset.datasets)
    assert len(concat_raw_dataset.description) == len(
        loaded_concat_raw_dataset.description
    )
    for raw_i in range(n_raw_datasets):
        actual_x, actual_y = concat_raw_dataset[raw_i]
        x, y = loaded_concat_raw_dataset[raw_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
    pd.testing.assert_frame_equal(
        concat_raw_dataset.description.astype(str),
        loaded_concat_raw_dataset.description.astype(str),
    )


def test_fail_outdated_save_on_new_window_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.raises(
        NotImplementedError,
        match="Outdated save not implemented for new window datasets.",
    ):
        concat_windows_dataset._outdated_save(path=tmpdir, overwrite=False)


def test_load_concat_windows_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    concat_windows_dataset.save(path=tmpdir, overwrite=False)
    loaded_concat_windows_dataset = load_concat_dataset(path=tmpdir, preload=False)
    assert len(concat_windows_dataset) == len(loaded_concat_windows_dataset)
    assert len(concat_windows_dataset.datasets) == len(
        loaded_concat_windows_dataset.datasets
    )
    assert len(concat_windows_dataset.description) == len(
        loaded_concat_windows_dataset.description
    )
    for windows_i in range(n_windows_datasets):
        actual_x, actual_y, actual_crop_inds = concat_windows_dataset[windows_i]
        x, y, crop_inds = loaded_concat_windows_dataset[windows_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y, actual_y, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(crop_inds, actual_crop_inds)
    pd.testing.assert_frame_equal(
        concat_windows_dataset.description.astype(str),
        loaded_concat_windows_dataset.description.astype(str),
    )


@pytest.mark.parametrize("use_mne_epochs", [False, True])
def test_event_source_metadata_roundtrip(tmp_path, use_mne_epochs):
    raws = []
    for _ in range(2):
        raw = mne.io.RawArray(
            np.zeros((1, 1100)),
            mne.create_info(["Cz"], sfreq=100, ch_types="eeg"),
            verbose=False,
        )
        raw.set_annotations(
            mne.Annotations(
                onset=[7, 1, 4],
                duration=[2, 2, 2],
                description=["B", "A", "OMIT"],
                extras=[
                    {"trial_uid": 30, "phase": "b"},
                    {"trial_uid": 10, "phase": "a"},
                    {"trial_uid": 20, "phase": "omit"},
                ],
            )
        )
        raws.append(raw)

    windows = create_windows_from_events(
        BaseConcatDataset(
            [
                RawDataset(raw, description={"recording": i})
                for i, raw in enumerate(raws)
            ]
        ),
        window_size_samples=100,
        window_stride_samples=100,
        on_last_window="drop",
        mapping={"A": 7, "B": 9},
        use_mne_epochs=use_mne_epochs,
        drop_bad_windows=False,
        preload=True,
    )
    expected = pd.DataFrame(
        {
            "i_window_in_trial": [0, 1, 0, 1],
            "i_trial_in_dataset": [0, 0, 2, 2],
            "i_start_in_trial": [100, 200, 700, 800],
            "i_stop_in_trial": [200, 300, 800, 900],
            "target": [7, 7, 9, 9],
            "trial_uid": [10, 10, 30, 30],
            "phase": ["a", "a", "b", "b"],
        }
    )
    for ds in windows.datasets:
        pd.testing.assert_frame_equal(
            ds.metadata.reset_index(drop=True), expected, check_dtype=False
        )
        assert pd.api.types.is_integer_dtype(ds.metadata["i_trial_in_dataset"])

    save_dir = tmp_path / "event-source-windows"
    windows.save(save_dir, overwrite=False)
    loaded = load_concat_dataset(save_dir, preload=False)

    for before, after in zip(windows.datasets, loaded.datasets):
        # MNE reindexes metadata rows to follow the Epochs selection, so only
        # pandas row-index identity is deliberately excluded from this roundtrip.
        pd.testing.assert_frame_equal(
            after.metadata.reset_index(drop=True),
            before.metadata.reset_index(drop=True),
        )
        assert pd.api.types.is_integer_dtype(after.metadata["i_trial_in_dataset"])
        normalized_kwargs = json.loads(json.dumps(before.window_kwargs))
        assert after.window_kwargs == normalized_kwargs
        assert "i_trial_in_dataset" not in json.dumps(after.window_kwargs)


def test_load_multiple_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    for i in range(2):
        i_offset = i * len(concat_raw_dataset.datasets)
        concat_raw_dataset.save(path=tmpdir, overwrite=False, offset=i_offset)
    loaded_concat_raw_datasets = load_concat_dataset(path=tmpdir, preload=False)
    assert 2 * len(concat_raw_dataset) == len(loaded_concat_raw_datasets)
    assert 2 * len(concat_raw_dataset.datasets) == len(
        loaded_concat_raw_datasets.datasets
    )
    assert 2 * len(concat_raw_dataset.description) == len(
        loaded_concat_raw_datasets.description
    )


def test_load_multiple_concat_windows_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    for i in range(2):
        i_offset = i * len(concat_windows_dataset.datasets)
        concat_windows_dataset.save(path=tmpdir, overwrite=False, offset=i_offset)
    loaded_concat_windows_datasets = load_concat_dataset(path=tmpdir, preload=False)
    assert 2 * len(concat_windows_dataset) == len(loaded_concat_windows_datasets)
    assert 2 * len(concat_windows_dataset.datasets) == len(
        loaded_concat_windows_datasets.datasets
    )
    assert 2 * len(concat_windows_dataset.description) == len(
        loaded_concat_windows_datasets.description
    )


def test_load_save_raw_preproc_kwargs(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    preprocess(
        concat_raw_dataset,
        [
            Preprocessor("pick_channels", ch_names=["C3"]),
        ],
    )
    concat_raw_dataset.save(tmpdir, overwrite=False)
    for i in range(len(concat_raw_dataset.datasets)):
        assert os.path.exists(os.path.join(tmpdir, str(i), "raw_preproc_kwargs.json"))
    loaded_concat_raw_dataset = load_concat_dataset(tmpdir, preload=False)
    for ds in loaded_concat_raw_dataset.datasets:
        assert ds.raw_preproc_kwargs == [
            {
                '__class_path__': 'braindecode.preprocessing.preprocess.Preprocessor',
                "fn": "pick_channels",
                "kwargs": {"ch_names": ["C3"]},
                "apply_on_array": False,
            },
        ]


def test_load_save_window_preproc_kwargs(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    concat_windows_dataset.save(tmpdir, overwrite=False)
    for i in range(len(concat_windows_dataset.datasets)):
        subdir = os.path.join(tmpdir, str(i))
        assert os.path.exists(os.path.join(subdir, "window_kwargs.json"))

    preprocess(
        concat_windows_dataset,
        [
            Preprocessor("pick_channels", ch_names=["Cz"]),
        ],
    )
    concat_windows_dataset.save(tmpdir, overwrite=True)
    for i in range(len(concat_windows_dataset.datasets)):
        subdir = os.path.join(tmpdir, str(i))
        assert os.path.exists(os.path.join(subdir, "window_kwargs.json"))
        assert os.path.exists(os.path.join(subdir, "raw_preproc_kwargs.json"))
    loaded_concat_windows_dataset = load_concat_dataset(tmpdir, preload=False)

    for ds in loaded_concat_windows_dataset.datasets:
        assert ds.window_kwargs == [
            [
                "create_windows_from_events",
                {
                    "infer_mapping": True,
                    "infer_window_size_stride": True,
                    "trial_start_offset_samples": 0,
                    "trial_stop_offset_samples": 0,
                    "window_size_samples": None,
                    "window_stride_samples": None,
                    "drop_last_window": False,
                    "mapping": {
                        "feet": 0,
                        "left_hand": 1,
                        "right_hand": 2,
                        "tongue": 3,
                    },
                    "preload": False,
                    "drop_bad_windows": None,
                    "picks": None,
                    "reject": None,
                    "flat": None,
                    "on_missing": "error",
                    "accepted_bads_ratio": 0.0,
                    'on_overlapping_events': 'raise',
                    "verbose": "error",
                    "use_mne_epochs": False,
                },
            ]
        ]
        assert ds.raw_preproc_kwargs == [
            {
                '__class_path__': 'braindecode.preprocessing.preprocess.Preprocessor',
                "fn": "pick_channels",
                "kwargs": {"ch_names": ["Cz"]},
                "apply_on_array": False,
            },
        ]


def test_save_concat_raw_dataset(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns() as raised_warnings:
        concat_raw_dataset.save(path=tmpdir, overwrite=False)
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1
    for raw_i in range(n_raw_datasets):
        subdir = os.path.join(tmpdir, str(raw_i))
        assert os.path.exists(os.path.join(subdir, "description.json"))
        assert os.path.exists(os.path.join(subdir, f"{raw_i}-raw.fif"))
    assert not os.path.exists(os.path.join(tmpdir, f"{n_raw_datasets}"))


def test_save_concat_windows_dataset(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns() as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=False)
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1
    for windows_i in range(n_windows_datasets):
        subdir = os.path.join(tmpdir, str(windows_i))
        assert os.path.exists(os.path.join(subdir, "description.json"))
        assert os.path.exists(os.path.join(subdir, f"{windows_i}-raw.fif"))
    assert not os.path.exists(os.path.join(tmpdir, f"{n_windows_datasets}"))


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_load_concat_raw_dataset_parallel(setup_concat_raw_dataset, tmpdir):
    concat_raw_dataset = setup_concat_raw_dataset
    n_raw_datasets = len(concat_raw_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns() as raised_warnings:
        concat_raw_dataset.save(path=tmpdir, overwrite=False)
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1

    # assert no warning raised with loading dataset saved in 'new' way
    with pytest.warns() as raised_warnings:
        loaded_concat_raw_dataset = load_concat_dataset(
            path=tmpdir, preload=False, n_jobs=2
        )
        warnings.warn("", UserWarning)
        assert len(raised_warnings) >= 0
    assert len(concat_raw_dataset) == len(loaded_concat_raw_dataset)
    assert len(concat_raw_dataset.datasets) == len(loaded_concat_raw_dataset.datasets)
    assert len(concat_raw_dataset.description) == len(
        loaded_concat_raw_dataset.description
    )
    for raw_i in range(n_raw_datasets):
        actual_x, actual_y = concat_raw_dataset[raw_i]
        x, y = loaded_concat_raw_dataset[raw_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
    pd.testing.assert_frame_equal(
        concat_raw_dataset.description.astype(str),
        loaded_concat_raw_dataset.description.astype(str),
    )


def test_load_concat_windows_dataset_parallel(setup_concat_windows_dataset, tmpdir):
    concat_windows_dataset = setup_concat_windows_dataset
    n_windows_datasets = len(concat_windows_dataset.datasets)
    # assert no warning raised with 'new' saving function
    with pytest.warns() as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=False)
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1
    # assert no warning raised anymore as underlying data is raw now
    with pytest.warns() as raised_warnings:
        loaded_concat_windows_dataset = load_concat_dataset(
            path=tmpdir, preload=False, n_jobs=2
        )
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1
    assert len(concat_windows_dataset) == len(loaded_concat_windows_dataset)
    assert len(concat_windows_dataset.datasets) == len(
        loaded_concat_windows_dataset.datasets
    )
    assert len(concat_windows_dataset.description) == len(
        loaded_concat_windows_dataset.description
    )
    for windows_i in range(n_windows_datasets):
        actual_x, actual_y, actual_crop_inds = concat_windows_dataset[windows_i]
        x, y, crop_inds = loaded_concat_windows_dataset[windows_i]
        np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y, actual_y, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(crop_inds, actual_crop_inds)
    pd.testing.assert_frame_equal(
        concat_windows_dataset.description.astype(str),
        loaded_concat_windows_dataset.description.astype(str),
    )


def test_save_varying_number_of_datasets_with_overwrite(
    setup_concat_windows_dataset, tmpdir
):
    concat_windows_dataset = setup_concat_windows_dataset
    concat_windows_dataset.save(path=tmpdir, overwrite=False)
    subset = concat_windows_dataset.split([0])["0"]
    with pytest.warns(UserWarning, match="The number of saved datasets"):
        subset.save(path=tmpdir, overwrite=True)

    # assert no warning raised when there are as many subdirectories than before
    with pytest.warns() as raised_warnings:
        concat_windows_dataset.save(path=tmpdir, overwrite=True)
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1

    # assert no warning raised when there are more subdirectories than before
    double_concat_windows_dataset = BaseConcatDataset(
        [concat_windows_dataset, concat_windows_dataset]
    )
    with pytest.warns() as raised_warnings:
        double_concat_windows_dataset.save(path=tmpdir, overwrite=True)
        warnings.warn("", UserWarning)
        assert len(raised_warnings) == 1


def _save_two_recordings(path, kind):
    rng = np.random.RandomState(0)
    info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types="eeg")
    raws = [mne.io.RawArray(rng.randn(2, 1000), info, verbose=False) for _ in range(2)]
    for raw in raws:
        raw.set_annotations(mne.Annotations([1, 5], [1, 1], ["a", "b"]))
    concat_ds = BaseConcatDataset(
        [RawDataset(raw, {"subject": i}) for i, raw in enumerate(raws)]
    )
    if kind == "windows":
        concat_ds = create_fixed_length_windows(
            concat_ds,
            window_size_samples=100,
            window_stride_samples=100,
            on_last_window="drop",
        )
    elif kind == "epochs":
        concat_ds = create_windows_from_events(concat_ds, use_mne_epochs=True)
    concat_ds.save(str(path))


def _signals(concat_ds):
    return [
        ds.raw.get_data() if hasattr(ds, "raw") else ds.windows.get_data()
        for ds in concat_ds.datasets
    ]


@pytest.mark.parametrize("kind", ["raw", "windows", "epochs", "epochs_of_lazy_raw"])
def test_save_overwrite_keeps_the_files_a_lazy_dataset_reads(tmp_path, kind):
    # Saving a lazily loaded dataset over the directory it reads from deleted
    # each recording before mne refused (or failed) to write it: data lost.
    _save_two_recordings(tmp_path, "raw" if kind == "epochs_of_lazy_raw" else kind)
    expected = _signals(load_concat_dataset(tmp_path, preload=True))
    lazy = load_concat_dataset(tmp_path, preload=False)
    if kind == "epochs_of_lazy_raw":
        lazy = create_windows_from_events(lazy, use_mne_epochs=True, preload=False)
    with pytest.raises(ValueError):
        lazy.save(str(tmp_path), overwrite=True)
    reloaded = load_concat_dataset(tmp_path, preload=True)
    for x_expected, x in zip(expected, _signals(reloaded)):
        np.testing.assert_array_equal(x, x_expected)


@pytest.mark.parametrize("spelling", ["other_case", "file_symlink"])
def test_save_overwrite_recognizes_the_source_directory(tmp_path, spelling):
    """The directory may be named with other letter case (case-insensitive file
    systems) or hold symlinks to the data (DataLad, git-annex)."""
    source = tmp_path / "data"
    _save_two_recordings(source, "raw")
    expected = _signals(load_concat_dataset(source, preload=True))
    target = source
    if spelling == "other_case":
        target = tmp_path / "DATA"
        if not target.exists():
            pytest.skip("case-sensitive file system")
    else:
        (tmp_path / "store").mkdir()
        for i in range(2):
            fif = source / str(i) / f"{i}-raw.fif"
            stored = fif.rename(tmp_path / "store" / fif.name)
            try:
                fif.symlink_to(stored)
            except OSError:
                pytest.skip("symlinks not supported")
    lazy = load_concat_dataset(source, preload=False)
    with pytest.raises(ValueError):
        lazy.save(str(target), overwrite=True)
    reloaded = load_concat_dataset(source, preload=True)
    for x_expected, x in zip(expected, _signals(reloaded)):
        np.testing.assert_array_equal(x, x_expected)


def test_save_overwrite_in_place_once_the_data_is_loaded(tmp_path):
    """Data in memory may still be saved back where it was loaded from."""
    _save_two_recordings(tmp_path, "raw")
    loaded = load_concat_dataset(tmp_path, preload=True)
    loaded.set_description({"group": ["a", "b"]})
    loaded.save(str(tmp_path), overwrite=True)
    reloaded = load_concat_dataset(tmp_path, preload=False)
    assert list(reloaded.description["group"]) == ["a", "b"]


def test_directory_contains_file(setup_concat_windows_dataset, tmpdir):
    with open(os.path.join(tmpdir, "test.txt"), "w") as f:
        f.write("test")
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.warns(UserWarning, match="Chosen directory"):
        concat_windows_dataset.save(tmpdir)


def test_other_subdirectories_exist(setup_concat_windows_dataset, tmpdir):
    os.mkdir(os.path.join(tmpdir, "999"))
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.warns(UserWarning, match="Chosen directory"):
        concat_windows_dataset.save(tmpdir)


def test_subdirectory_already_exist(setup_concat_windows_dataset, tmpdir):
    os.mkdir(os.path.join(tmpdir, "0"))
    concat_windows_dataset = setup_concat_windows_dataset
    with pytest.raises(FileExistsError, match="Subdirectory"):
        concat_windows_dataset.save(tmpdir)


def test_check_save_dir_empty(setup_concat_raw_dataset, tmpdir):
    _check_save_dir_empty(tmpdir)
    setup_concat_raw_dataset.save(tmpdir)
    with pytest.raises(FileExistsError):
        _check_save_dir_empty(tmpdir)


def test_save_concat_dataset(tmpdir, setup_concat_raw_dataset):
    # Call the save_concat_dataset function
    with pytest.warns(UserWarning):
        save_concat_dataset(tmpdir, setup_concat_raw_dataset, overwrite=False)


def test_load_concat_windows_dataset_channel_targets(tmp_path):
    info = mne.create_info(["Cz", "target"], sfreq=100, ch_types=["eeg", "misc"])
    raw = mne.io.RawArray(np.random.RandomState(0).randn(2, 1000), info, verbose=False)
    windows = create_fixed_length_windows(
        BaseConcatDataset([RawDataset(raw, description={"recording": 0})]),
        window_size_samples=100,
        window_stride_samples=100,
        on_last_window="overlap",
        targets_from="channels",
        last_target_only=False,
    )
    windows.save(tmp_path, overwrite=False)
    loaded = load_concat_dataset(tmp_path, preload=False)

    for before, after in zip(windows.datasets, loaded.datasets):
        assert after.targets_from == before.targets_from
        assert after.last_target_only == before.last_target_only
    for i in range(len(windows)):
        x_before, y_before, _ = windows[i]
        x_after, y_after, _ = loaded[i]
        np.testing.assert_allclose(x_after, x_before, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(y_after, y_before, rtol=1e-4, atol=1e-5)


def test_save_with_offset_expects_sibling_subdirectories(
    setup_concat_raw_dataset, tmpdir
):
    """preprocess(save_dir) saves recording i at offset i into one directory:
    the siblings are expected, so no scan of them and no warning listing them."""
    concat_raw_dataset = setup_concat_raw_dataset
    n = len(concat_raw_dataset.datasets)
    concat_raw_dataset.save(path=tmpdir, offset=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        concat_raw_dataset.save(path=tmpdir, offset=n)
    assert len(load_concat_dataset(tmpdir, preload=False).datasets) == 2 * n


def test_load_concat_dataset_closes_json_files(setup_concat_raw_dataset, tmpdir):
    """Reloading reads target_name.json and the kwargs JSONs without leaking file handles."""
    concat_raw_dataset = setup_concat_raw_dataset
    preprocess(concat_raw_dataset, [Preprocessor("pick_channels", ch_names=["C3"])])
    concat_raw_dataset.save(tmpdir, overwrite=False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        load_concat_dataset(tmpdir, preload=False)
    assert not [w for w in record if issubclass(w.category, ResourceWarning)]


def test_load_concat_dataset_keeps_fif_tags_as_one_array(tmp_path):
    """Reloaded lazy FIF raws hold their buffer tags as one array; reads unchanged."""
    import pickle

    from braindecode.datasets.base import _FifTagTable, _read_fif_window

    info = mne.create_info(["a", "b", "c"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.RandomState(0).randn(3, 1234) * 1e-5, info)
    BaseConcatDataset([RawDataset(raw, description={"recording": 0})]).save(tmp_path)
    fif = tmp_path / "0" / "0-raw.fif"
    mne_raw = mne.io.read_raw_fif(fif, preload=False)
    for _ in range(2):  # reads the fif and writes the pickle, then reads the pickle
        lazy = load_concat_dataset(tmp_path, preload=False).datasets[0].raw
        table = lazy._raw_extras[0]["ent"]
        assert isinstance(table, _FifTagTable)
        assert len(table) == 13 and list(table) == mne_raw._raw_extras[0]["ent"]
    with open(fif.with_suffix(".pkl"), "rb") as f:  # mne's own layout on disk
        assert isinstance(pickle.load(f)._raw_extras[0]["ent"], list)
    expected = mne_raw.get_data()
    for copy in (lazy, lazy.copy(), pickle.loads(pickle.dumps(lazy))):
        np.testing.assert_array_equal(copy.get_data(), expected)  # mne's own read
        for start, stop in [(0, 1), (90, 310), (99, 100), (1100, 1234)]:
            np.testing.assert_array_equal(
                _read_fif_window(copy, start, stop), expected[:, start:stop]
            )


def test_load_copied_dataset_reads_the_copy(setup_concat_raw_dataset, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    setup_concat_raw_dataset.save(path=src, overwrite=False)
    load_concat_dataset(path=src, preload=False)  # writes the N-raw.pkl caches
    shutil.copytree(src, dst)  # copies the caches, which point into src
    with pytest.warns(UserWarning, match="not next to it"):
        loaded = load_concat_dataset(path=dst, preload=False)
    for i, ds in enumerate(loaded.datasets):
        assert Path(ds.raw.filenames[0]).parent.resolve() == (dst / str(i)).resolve()
    # the re-created caches point into dst: loading again does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        load_concat_dataset(path=dst, preload=False)


def test_load_concat_dataset_ignores_stray_entries(setup_concat_raw_dataset, tmp_path):
    setup_concat_raw_dataset.save(path=tmp_path, overwrite=False)
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "README.md").write_text("notes")
    (tmp_path / "plots").mkdir()
    loaded = load_concat_dataset(path=tmp_path, preload=False)
    assert len(loaded.datasets) == len(setup_concat_raw_dataset.datasets)


def _string_description_raw_dataset():
    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((2, 200)), info)
    description = {"subject": "01", "session": "02", "age": 30, "sex": "F"}
    return BaseConcatDataset([RawDataset(raw, description, target_name="age")])


def test_load_concat_dataset_keeps_description_strings(tmp_path):
    concat_ds = _string_description_raw_dataset()
    # windows keep only part of the description: all of it numeric strings
    concat_ds.datasets[0]._description = concat_ds.datasets[0].description[
        ["subject", "session"]
    ]
    concat_ds.save(tmp_path)
    loaded = load_concat_dataset(tmp_path, preload=False)
    assert loaded.datasets[0].description.to_dict() == {
        "subject": "01",
        "session": "02",
    }


def test_load_concat_dataset_target_name(tmp_path):
    concat_ds = _string_description_raw_dataset()
    concat_ds.save(tmp_path)
    loaded = load_concat_dataset(tmp_path, preload=False, target_name="sex")
    assert loaded.datasets[0].target_name == "sex"
    assert loaded[0][1] == "F"
    loaded = load_concat_dataset(tmp_path, preload=False)
    assert loaded.datasets[0].target_name == "age"


def test_load_concat_dataset_target_name_windows_raises(
    setup_concat_windows_dataset, tmp_path
):
    setup_concat_windows_dataset.save(tmp_path)
    with pytest.raises(ValueError, match="set_target"):
        load_concat_dataset(tmp_path, preload=False, target_name="subject")

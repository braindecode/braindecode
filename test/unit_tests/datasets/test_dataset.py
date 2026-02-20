# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import BaseConcatDataset, RawDataset, WindowsDataset
from braindecode.datasets.moabb import fetch_data_with_moabb
from braindecode.preprocessing.windowers import (
    create_fixed_length_windows,
    create_windows_from_events,
)

bnci_kwargs = {
    "n_sessions": 2,
    "n_runs": 3,
    "n_subjects": 9,
    "paradigm": "imagery",
    "duration": 386.9,
    "sfreq": 250,
    "event_list": ("left", "right"),
    "channels": (
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ),
}


# TODO: split file up into files with proper matching names
@pytest.fixture(scope="module")
# TODO: add test for transformers and case when subject_info is used
def set_up():
    rng = np.random.RandomState(42)
    info = mne.create_info(ch_names=["0", "1"], sfreq=50, ch_types="eeg")
    raw = mne.io.RawArray(data=rng.randn(2, 1000), info=info)
    desc = pd.Series({"pathological": True, "gender": "M", "age": 48})
    base_dataset = RawDataset(raw, desc, target_name="age")

    events = np.array([[100, 0, 1], [200, 0, 2], [300, 0, 1], [400, 0, 4], [500, 0, 3]])
    window_idxs = [(0, 0, 100), (0, 100, 200), (1, 0, 100), (2, 0, 100), (2, 50, 150)]
    i_window_in_trial, i_start_in_trial, i_stop_in_trial = list(zip(*window_idxs))
    metadata = pd.DataFrame(
        {
            "sample": events[:, 0],
            "x": events[:, 1],
            "target": events[:, 2],
            "i_window_in_trial": i_window_in_trial,
            "i_start_in_trial": i_start_in_trial,
            "i_stop_in_trial": i_stop_in_trial,
        }
    )

    mne_epochs = mne.Epochs(raw=raw, events=events, metadata=metadata)
    windows_dataset = WindowsDataset(mne_epochs, desc)

    return raw, base_dataset, mne_epochs, windows_dataset, events, window_idxs


@pytest.fixture(scope="module")
def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="FakeDataset", subject_ids=1, dataset_kwargs=bnci_kwargs
    )

    events, _ = mne.events_from_annotations(raws[0])
    targets = events[:, -1] - 1
    ds = [RawDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)
    return concat_ds, targets


@pytest.fixture(scope="module")
def concat_windows_dataset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows_ds = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=100,
        window_stride_samples=100,
        drop_last_window=False,
    )

    return windows_ds


def test_get_item(set_up):
    _, _, mne_epochs, windows_dataset, events, window_idxs = set_up
    for i, epoch in enumerate(mne_epochs.get_data()):
        x, y, inds = windows_dataset[i]
        np.testing.assert_allclose(epoch, x)
        assert events[i, 2] == y, f"Y not equal for epoch {i}"
        np.testing.assert_array_equal(
            window_idxs[i], inds, f"window inds not equal for epoch {i}"
        )


def test_len_windows_dataset(set_up):
    _, _, mne_epochs, windows_dataset, _, _ = set_up
    assert len(mne_epochs.events) == len(windows_dataset)


def test_len_base_dataset(set_up):
    raw, base_dataset, _, _, _, _ = set_up
    assert len(raw) == len(base_dataset)


def test_len_concat_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    assert len(concat_ds) == sum([len(c) for c in concat_ds.datasets])


def test_target_in_subject_info(set_up):
    raw, _, _, _, _, _ = set_up
    desc = pd.Series({"pathological": True, "gender": "M", "age": 48})
    with pytest.warns(UserWarning, match="'does_not_exist' not in description"):
        RawDataset(raw, desc, target_name="does_not_exist")


def test_description_concat_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    assert isinstance(concat_ds.description, pd.DataFrame)
    assert concat_ds.description.shape[0] == len(concat_ds.datasets)


def test_split_concat_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    splits = concat_ds.split("run")

    for k, v in splits.items():
        assert k == v.description["run"].values
        assert isinstance(v, BaseConcatDataset)

    assert len(concat_ds) == sum([len(v) for v in splits.values()])


def test_concat_concat_dataset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    concat_ds1 = BaseConcatDataset(concat_ds.datasets[:2])
    concat_ds2 = BaseConcatDataset(concat_ds.datasets[2:])
    list_of_concat_ds = [concat_ds1, concat_ds2]
    descriptions = pd.concat([ds.description for ds in list_of_concat_ds])
    descriptions.reset_index(inplace=True, drop=True)
    lens = [0] + [len(ds) for ds in list_of_concat_ds]
    cumsums = [ds.cumulative_sizes for ds in list_of_concat_ds]
    cumsums = [
        ls for i, cumsum in enumerate(cumsums) for ls in np.array(cumsum) + lens[i]
    ]
    concat_concat_ds = BaseConcatDataset(list_of_concat_ds)
    assert len(concat_concat_ds) == sum(lens)
    assert len(concat_concat_ds) == concat_concat_ds.cumulative_sizes[-1]
    assert len(concat_concat_ds.datasets) == len(descriptions)
    assert len(concat_concat_ds.description) == len(descriptions)
    np.testing.assert_array_equal(cumsums, concat_concat_ds.cumulative_sizes)
    pd.testing.assert_frame_equal(descriptions, concat_concat_ds.description)


def test_split_dataset_failure(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    with pytest.raises(KeyError):
        concat_ds.split("test")

    with pytest.raises(IndexError):
        concat_ds.split([])

    with pytest.raises(ValueError, match="datasets should not be an empty iterable"):
        concat_ds.split([[]])

    with pytest.raises(TypeError):
        concat_ds.split([[[]]])

    with pytest.raises(IndexError):
        concat_ds.split([len(concat_ds.description)])

    with pytest.raises(ValueError):
        concat_ds.split([4], [5])

    with pytest.warns(DeprecationWarning, match="Keyword arguments"):
        concat_ds.split(split_ids=[0])

    with pytest.warns(DeprecationWarning, match="Keyword arguments"):
        concat_ds.split(property="subject")


def test_split_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    splits = concat_ds.split("run")
    assert len(splits) == len(concat_ds.description["run"].unique())

    splits = concat_ds.split([1])
    assert len(splits) == 1
    assert len(splits["0"].datasets) == 1

    splits = concat_ds.split([[2]])
    assert len(splits) == 1
    assert len(splits["0"].datasets) == 1

    original_ids = [1, 2]
    splits = concat_ds.split([[0], original_ids])
    assert len(splits) == 2
    assert list(splits["0"].description.index) == [0]
    assert len(splits["0"].datasets) == 1
    # when creating new BaseConcatDataset, index is reset
    split_ids = [0, 1]
    assert list(splits["1"].description.index) == split_ids
    assert len(splits["1"].datasets) == 2

    for i, ds in enumerate(splits["1"].datasets):
        np.testing.assert_array_equal(
            ds.raw.get_data(), concat_ds.datasets[original_ids[i]].raw.get_data()
        )

    # Test split_ids as dict
    split_ids = dict(train=[1], test=[2])
    splits = concat_ds.split(split_ids)
    assert len(splits) == len(split_ids)
    assert splits.keys() == split_ids.keys()
    assert (splits["train"].description["run"] == "1").all()
    assert (splits["test"].description["run"] == "2").all()


def test_metadata(concat_windows_dataset):
    md = concat_windows_dataset.get_metadata()
    assert isinstance(md, pd.DataFrame)
    assert all([c in md.columns for c in concat_windows_dataset.description.columns])
    assert md.shape[0] == len(concat_windows_dataset)


def test_no_metadata(concat_ds_targets):
    with pytest.raises(TypeError, match="Metadata dataframe can only be"):
        concat_ds_targets[0].get_metadata()


def test_on_the_fly_transforms_base_dataset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    original_X = concat_ds[0][0]
    factor = 10
    transform = lambda x: x * factor  # noqa: E731
    concat_ds.transform = transform
    transformed_X = concat_ds[0][0]

    assert (factor * original_X == transformed_X).all()

    with pytest.raises(ValueError):
        concat_ds.transform = 0


def test_on_the_fly_transforms_windows_dataset(concat_windows_dataset):
    original_X = concat_windows_dataset[0][0]
    factor = 10
    transform = lambda x: x * factor  # noqa: E731
    concat_windows_dataset.transform = transform
    transformed_X = concat_windows_dataset[0][0]

    assert (factor * original_X == transformed_X).all()

    with pytest.raises(ValueError):
        concat_windows_dataset.transform = 0


def test_set_description_base_dataset(concat_ds_targets):
    concat_ds = concat_ds_targets[0]
    assert len(concat_ds.description.columns) == 3
    # add multiple new entries at the same time to concat of base
    concat_ds.set_description(
        {
            "hi": ["who", "are", "you"],
            "how": ["does", "this", "work"],
        }
    )
    assert len(concat_ds.description.columns) == 5
    assert concat_ds.description.loc[1, "hi"] == "are"
    assert concat_ds.description.loc[0, "how"] == "does"

    # do the same again but give a DataFrame this time
    concat_ds.set_description(
        pd.DataFrame.from_dict(
            {
                2: ["", "need", "sleep"],
            }
        )
    )
    assert len(concat_ds.description.columns) == 6
    assert concat_ds.description.loc[0, 2] == ""

    # try to set existing description without overwriting
    with pytest.raises(
        AssertionError,
        match="'how' already in description. Please rename or set overwrite to" " True.",
    ):
        concat_ds.set_description(
            {
                "first": [-1, -1, -1],
                "how": ["this", "will", "fail"],
            },
            overwrite=False,
        )

    # add single entry to single base
    base_ds = concat_ds.datasets[0]
    base_ds.set_description({"test": 4})
    assert "test" in base_ds.description
    assert base_ds.description["test"] == 4

    # overwrite single entry in single base using a Series
    base_ds.set_description(pd.Series({"test": 0}), overwrite=True)
    assert base_ds.description["test"] == 0


def test_set_description_windows_dataset(concat_windows_dataset):
    assert len(concat_windows_dataset.description.columns) == 3
    # add multiple entries to multiple windows
    concat_windows_dataset.set_description(
        {
            "wow": ["this", "is", "cool"],
            3: [1, 2, 3],
        }
    )
    assert len(concat_windows_dataset.description.columns) == 5
    assert concat_windows_dataset.description.loc[2, "wow"] == "cool"
    assert concat_windows_dataset.description.loc[1, 3] == 2

    # do the same, however this time give a DataFrame
    concat_windows_dataset.set_description(
        pd.DataFrame.from_dict(
            {
                "hello": [0, 0, 0],
            }
        )
    )
    assert len(concat_windows_dataset.description.columns) == 6
    assert concat_windows_dataset.description["hello"].to_list() == [0, 0, 0]

    # add single entry in single window
    window_ds = concat_windows_dataset.datasets[-1]
    window_ds.set_description({"4": 123})
    assert "4" in window_ds.description
    assert window_ds.description["4"] == 123

    # overwrite multiple in single window
    window_ds.set_description(
        {
            "4": "overwritten",
            "wow": "not cool",
        },
        overwrite=True,
    )
    assert window_ds.description["4"] == "overwritten"
    assert window_ds.description["wow"] == "not cool"

    # try to set existing description without overwriting using Series
    with pytest.raises(
        AssertionError,
        match="'wow' already in description. Please rename or set overwrite to" " True.",
    ):
        window_ds.set_description(pd.Series({"wow": "error"}), overwrite=False)


def test_concat_dataset_get_sequence_out_of_range(concat_windows_dataset):
    indices = [len(concat_windows_dataset)]
    with pytest.raises(IndexError):
        X, y = concat_windows_dataset[indices]


def test_concat_dataset_target_transform(concat_windows_dataset):
    indices = range(100)
    y = concat_windows_dataset[indices][1]

    concat_windows_dataset.target_transform = sum
    y2 = concat_windows_dataset[indices][1]

    assert y2 == sum(y)


def test_concat_dataset_invalid_target_transform(concat_windows_dataset):
    with pytest.raises(TypeError):
        concat_windows_dataset.target_transform = 0


def test_multi_target_dataset(set_up):
    _, base_dataset, _, _, _, _ = set_up
    base_dataset.target_name = ["pathological", "gender", "age"]
    x, y = base_dataset[0]
    assert len(y) == 3
    assert base_dataset.description.to_list() == y
    concat_ds = BaseConcatDataset([base_dataset])
    windows_ds = create_fixed_length_windows(
        concat_ds,
        window_size_samples=100,
        window_stride_samples=100,
        start_offset_samples=0,
        stop_offset_samples=None,
        drop_last_window=False,
        mapping={True: 1, False: 0, "M": 0, "F": 1},  # map non-digit targets
    )
    x, y, ind = windows_ds[0]
    assert len(y) == 3
    assert y == [1, 0, 48]  # order matters: pathological, gender, age


def test_target_name_list(set_up):
    raw, _, _, _, _, _ = set_up
    target_names = ["pathological", "gender", "age"]
    base_dataset = RawDataset(
        raw=raw,
        description={"pathological": True, "gender": "M", "age": 48},
        target_name=target_names,
    )
    assert base_dataset.target_name == target_names


def test_description_incorrect_type(set_up):
    raw, _, _, _, _, _ = set_up
    with pytest.raises(ValueError):
        RawDataset(
            raw=raw,
            description=("test", 4),
        )


def test_target_name_incorrect_type(set_up):
    raw, _, _, _, _, _ = set_up
    with pytest.raises(
        ValueError, match="target_name has to be None, str, tuple or list"
    ):
        RawDataset(raw, target_name={"target": 1})


def test_target_name_not_in_description(set_up):
    raw, _, _, _, _, _ = set_up
    with pytest.warns(UserWarning):
        base_dataset = RawDataset(raw, target_name=("pathological", "gender", "age"))
    with pytest.raises(TypeError):
        x, y = base_dataset[0]
    base_dataset.set_description({"pathological": True, "gender": "M", "age": 48})
    x, y = base_dataset[0]


def test_windows_dataset_from_target_channels_raise_valuerror(set_up):
    _, _, epochs, _, _, _ = set_up
    with pytest.raises(ValueError):
        WindowsDataset(epochs, None, targets_from="non-existing")


# ==================== Tests for lazy initialization ====================


def test_lazy_cumulative_sizes_not_computed_on_init(concat_ds_targets):
    """Test that cumulative sizes are not computed during init with lazy=True."""
    concat_ds, _ = concat_ds_targets
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)
    assert lazy_ds._cumulative_sizes is None
    assert lazy_ds._lazy is True


def test_lazy_len_triggers_computation(concat_ds_targets):
    """Test that len() correctly triggers cumulative size computation."""
    concat_ds, _ = concat_ds_targets
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)
    assert lazy_ds._cumulative_sizes is None
    length = len(lazy_ds)
    assert lazy_ds._cumulative_sizes is not None
    assert length == len(concat_ds)


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_lazy_getitem_positive_index(concat_ds_targets, idx):
    """Test __getitem__ with positive indices in lazy mode."""
    concat_ds, _ = concat_ds_targets
    if idx >= len(concat_ds):
        pytest.skip(f"Index {idx} out of range for dataset of length {len(concat_ds)}")
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)
    lazy_item = lazy_ds[idx]
    eager_item = concat_ds[idx]
    np.testing.assert_array_equal(lazy_item[0], eager_item[0])
    assert lazy_item[1] == eager_item[1]


@pytest.mark.parametrize("idx", [-1, -2, -3])
def test_lazy_getitem_negative_index(concat_ds_targets, idx):
    """Test __getitem__ with negative indices in lazy mode."""
    concat_ds, _ = concat_ds_targets
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)
    lazy_item = lazy_ds[idx]
    eager_item = concat_ds[idx]
    np.testing.assert_array_equal(lazy_item[0], eager_item[0])
    assert lazy_item[1] == eager_item[1]


def test_lazy_getitem_negative_index_out_of_range(concat_ds_targets):
    """Test __getitem__ with negative index that exceeds dataset length."""
    concat_ds, _ = concat_ds_targets
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)
    with pytest.raises(ValueError, match="absolute value of index should not exceed"):
        lazy_ds[-len(lazy_ds) - 1]


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_identical_results_across_modes(concat_ds_targets, lazy):
    """Test that lazy and eager initialization produce identical results."""
    concat_ds, _ = concat_ds_targets
    ds = BaseConcatDataset(concat_ds.datasets, lazy=lazy)
    reference_ds = BaseConcatDataset(concat_ds.datasets, lazy=False)

    assert len(ds) == len(reference_ds)
    np.testing.assert_array_equal(ds.cumulative_sizes, reference_ds.cumulative_sizes)

    for i in range(min(5, len(reference_ds))):
        item = ds[i]
        ref_item = reference_ds[i]
        np.testing.assert_array_equal(item[0], ref_item[0])
        assert item[1] == ref_item[1]


def test_lazy_flattened_baseconcatdataset(concat_ds_targets):
    """Test that flattening BaseConcatDatasets works correctly with lazy=True."""
    concat_ds, _ = concat_ds_targets
    ds_list = concat_ds.datasets

    concat_ds1 = BaseConcatDataset(ds_list[:2], lazy=True)
    concat_ds2 = BaseConcatDataset(ds_list[2:], lazy=True)
    flattened_ds = BaseConcatDataset([concat_ds1, concat_ds2], lazy=True)

    assert len(flattened_ds.datasets) == len(ds_list)
    assert len(flattened_ds) == len(concat_ds)

    for i in range(min(5, len(concat_ds))):
        flat_item = flattened_ds[i]
        orig_item = concat_ds[i]
        np.testing.assert_array_equal(flat_item[0], orig_item[0])
        assert flat_item[1] == orig_item[1]


def test_lazy_cumulative_sizes_property(concat_ds_targets):
    """Test that cumulative_sizes property works correctly."""
    concat_ds, _ = concat_ds_targets
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)

    cumsizes = lazy_ds.cumulative_sizes
    assert cumsizes is not None
    assert len(cumsizes) == len(concat_ds.datasets)
    np.testing.assert_array_equal(cumsizes, concat_ds.cumulative_sizes)


def test_lazy_get_sequence(concat_ds_targets):
    """Test _get_sequence works correctly with lazy initialization."""
    concat_ds, _ = concat_ds_targets
    lazy_ds = BaseConcatDataset(concat_ds.datasets, lazy=True)
    eager_ds = BaseConcatDataset(concat_ds.datasets, lazy=False)

    indices = list(range(min(5, len(eager_ds))))
    lazy_X, lazy_y = lazy_ds[indices]
    eager_X, eager_y = eager_ds[indices]

    np.testing.assert_array_equal(lazy_X, eager_X)
    np.testing.assert_array_equal(lazy_y, eager_y)


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_empty_dataset_raises_valueerror(lazy):
    """Test that empty dataset raises ValueError in both modes."""
    with pytest.raises(ValueError, match="datasets should not be an empty iterable"):
        BaseConcatDataset([], lazy=lazy)


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_iterable_dataset_raises_typeerror(lazy):
    """Test that providing an IterableDataset raises TypeError in both modes."""
    from torch.utils.data import IterableDataset

    class DummyIterableDataset(IterableDataset):
        def __iter__(self):
            yield 1

    with pytest.raises(TypeError, match="ConcatDataset does not support IterableDataset"):
        BaseConcatDataset([DummyIterableDataset()], lazy=lazy)


# ==================== Tests for __repr__ and _repr_html_ ====================


def test_raw_dataset_repr(set_up):
    """Test __repr__ of RawDataset."""
    _, base_dataset, _, _, _, _ = set_up
    r = repr(base_dataset)
    assert "RawDataset" in r
    assert "ch" in r
    assert "Hz" in r
    assert "samples" in r
    assert "description" in r


def test_windows_dataset_repr(set_up):
    """Test __repr__ of WindowsDataset."""
    _, _, _, windows_dataset, _, _ = set_up
    r = repr(windows_dataset)
    assert "WindowsDataset" in r
    assert "windows" in r
    assert "ch" in r
    assert "Hz" in r
    assert "description" in r


def test_base_concat_dataset_repr(concat_ds_targets):
    """Test __repr__ of BaseConcatDataset of RawDatasets."""
    concat_ds = concat_ds_targets[0]
    r = repr(concat_ds)
    assert "BaseConcatDataset" in r
    assert "RawDataset" in r
    assert "recordings" in r
    assert "Sfreq" in r
    assert "Channels" in r
    assert "Duration" in r
    assert "* from first recording" in r
    assert "Description" in r


def test_base_concat_windows_dataset_repr(concat_windows_dataset):
    """Test __repr__ of BaseConcatDataset of windowed datasets."""
    r = repr(concat_windows_dataset)
    assert "BaseConcatDataset" in r
    assert "Sfreq" in r
    assert "Channels" in r
    assert "Window" in r
    assert "* from first recording" in r


def test_base_concat_dataset_repr_html(concat_ds_targets):
    """Test _repr_html_ of BaseConcatDataset."""
    concat_ds = concat_ds_targets[0]
    html = concat_ds._repr_html_()
    assert "<table" in html
    assert "BaseConcatDataset" in html
    assert "Recordings" in html
    assert "Total samples" in html
    assert "Sfreq" in html
    assert "Channels" in html
    assert "* from first recording" in html


def test_raw_dataset_repr_html(set_up):
    """Test _repr_html_ of RawDataset."""
    _, base_dataset, _, _, _, _ = set_up
    html = base_dataset._repr_html_()
    assert "<table" in html
    assert "RawDataset" in html
    assert "Channels" in html
    assert "Sfreq" in html
    assert "Samples" in html


def test_windows_dataset_repr_html(set_up):
    """Test _repr_html_ of WindowsDataset."""
    _, _, _, windows_dataset, _, _ = set_up
    html = windows_dataset._repr_html_()
    assert "<table" in html
    assert "WindowsDataset" in html
    assert "Channels" in html
    assert "Targets" in html


def test_windows_dataset_repr_metadata(set_up):
    """Test __repr__ of WindowsDataset includes target info."""
    _, _, _, windows_dataset, _, _ = set_up
    r = repr(windows_dataset)
    assert "targets:" in r


def test_eeg_windows_dataset_repr_html(concat_windows_dataset):
    """Test _repr_html_ of EEGWindowsDataset."""
    ds = concat_windows_dataset.datasets[0]
    html = ds._repr_html_()
    assert "<table" in html
    assert "EEGWindowsDataset" in html
    assert "Channels" in html
    assert "Targets" in html


def test_eeg_windows_dataset_repr_metadata(concat_windows_dataset):
    """Test __repr__ of EEGWindowsDataset includes target info."""
    ds = concat_windows_dataset.datasets[0]
    r = repr(ds)
    assert "targets:" in r


def test_base_concat_windows_dataset_repr_metadata(concat_windows_dataset):
    """Test __repr__ of BaseConcatDataset of windowed datasets includes target info."""
    r = repr(concat_windows_dataset)
    assert "Targets" in r


def test_base_concat_windows_dataset_repr_html_metadata(concat_windows_dataset):
    """Test _repr_html_ of BaseConcatDataset of windowed datasets includes target info."""
    html = concat_windows_dataset._repr_html_()
    assert "Targets" in html


def test_eeg_windows_dataset_repr(concat_windows_dataset):
    """Test __repr__ of EEGWindowsDataset."""
    ds = concat_windows_dataset.datasets[0]
    r = repr(ds)
    assert "EEGWindowsDataset" in r
    assert "windows" in r
    assert "ch" in r
    assert "Hz" in r
    assert "samples/win" in r

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import mne
import pytest
from moabb.datasets import FakeDataset
from pandas import DataFrame

from braindecode.datasets import MOABBDataset
from braindecode.datasets.moabb import (
    BNCI2014001,
    _fetch_and_unpack_moabb_data,
    _find_dataset_in_moabb,
)


@pytest.fixture(scope="module")
def moabb_dataset():
    return FakeDataset(
        n_subjects=1, n_sessions=1, n_runs=1, stim=True, annotations=True
    )


def test_moabb_with_raw_moabb(moabb_dataset):
    dataset = MOABBDataset(moabb_dataset, subject_ids=[1])
    assert len(dataset.datasets) == 1
    assert len(dataset.datasets[0].raw.ch_names) == 4


def test_fetch_and_unpack_moabb_data(moabb_dataset):
    subject_id = [1]
    raws, description = _fetch_and_unpack_moabb_data(moabb_dataset, subject_id)

    assert len(raws) == 1
    assert len(description) == 1
    assert isinstance(raws[0], mne.io.BaseRaw)
    assert isinstance(description, DataFrame)


def test_fetch_and_unpack_moabb_data_with_dataset_load(moabb_dataset):
    subject_id = [1]

    cache_config = dict(
        save_raw=False,
        save_epochs=False,
        save_array=False,
        use=True,
        overwrite_raw=False,
        overwrite_epochs=False,
        overwrite_array=False,
        path=None,
    )
    dataset_load_kwargs = {
        "cache_config": cache_config,
    }
    raw, description = _fetch_and_unpack_moabb_data(
        moabb_dataset, subject_id, dataset_load_kwargs
    )

    assert len(raw) == 1
    assert len(description) == 1
    assert isinstance(raw[0], mne.io.BaseRaw)
    assert isinstance(description, DataFrame)


def test_find_dataset_in_moabb(moabb_dataset):
    found_dataset = _find_dataset_in_moabb(moabb_dataset.__class__.__name__)
    assert isinstance(found_dataset, FakeDataset)


def test_not_found_dataset_in_moabb():
    with pytest.raises(ValueError):
        _find_dataset_in_moabb("NotExistingDataset")


def test_BNCI2014001():
    dataset = BNCI2014001([1])
    assert len(dataset.datasets) == 12

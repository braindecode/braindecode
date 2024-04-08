# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)


import pytest
from moabb.datasets import FakeDataset
from braindecode.datasets import MOABBDataset


@pytest.fixture(scope="module")
def moabb_dataset():
    return FakeDataset(n_subjects=1,
                       n_sessions=1,
                       n_runs=1,
                       stim=True,
                       annotations=True)


def test_moabb_with_raw_moabb(moabb_dataset):
    dataset = MOABBDataset(moabb_dataset, subject_ids=[1])
    assert len(dataset.datasets) == 1
    assert len(dataset.datasets[0].raw.ch_names) == 4

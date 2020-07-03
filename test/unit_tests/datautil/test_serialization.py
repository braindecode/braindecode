# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

import os

import pytest
import numpy as np

from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.serialization import (
    recover_windows_dataset, store_windows_dataset)


@pytest.fixture(scope="module")
def setup_windows_ds():
    moabb_ds = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])
    return create_windows_from_events(
        concat_ds=moabb_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0
    )


def test_store_windows_dataset(setup_windows_ds, tmpdir):
    windows_ds = setup_windows_ds
    n_windows_ds = len(windows_ds.datasets)
    store_windows_dataset(
        windows_dataset=windows_ds, path=tmpdir, overwrite=False)
    assert os.path.exists(tmpdir.join(f"description.json"))
    assert os.path.exists(tmpdir.join("./0-epo.fif"))
    assert os.path.exists(tmpdir.join(f"./{n_windows_ds - 1}-epo.fif"))
    assert not os.path.exists(tmpdir.join(f"./{n_windows_ds}-epo.fif"))


def test_recover_windows_dataset(setup_windows_ds, tmpdir):
    windows_ds = setup_windows_ds
    store_windows_dataset(
        windows_dataset=windows_ds, path=tmpdir, overwrite=False)
    recovered_windows_ds = recover_windows_dataset(tmpdir)
    assert len(windows_ds) == len(recovered_windows_ds)
    assert len(windows_ds.datasets) == len(recovered_windows_ds.datasets)
    assert len(windows_ds.description) == len(recovered_windows_ds.description)
    actual_x, actual_y, actual_crop_inds = windows_ds[0]
    x, y, crop_inds = recovered_windows_ds[0]
    np.testing.assert_allclose(x, actual_x, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(y, actual_y, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(crop_inds, actual_crop_inds, rtol=1e-4, atol=1e-5)

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3

from collections import OrderedDict

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets import MOABBDataset
from braindecode.datautil.transforms import transform_concat_ds
from braindecode.datautil.windowers import create_fixed_length_windows


@pytest.fixture(scope="module")
def base_concat_ds():
    return MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1, 2])


@pytest.fixture(scope='module')
def windows_concat_ds(base_concat_ds):
    return create_fixed_length_windows(
        base_concat_ds, start_offset_samples=100, stop_offset_samples=None,
        supercrop_size_samples=1000, supercrop_stride_samples=1000,
        drop_samples=True, mapping=None)


def test_not_ordered_dict():
    with pytest.raises(TypeError):
        transform_concat_ds(None, {'test': 1})


def test_no_raw_or_epochs():
    class EmptyDataset(object):
        def __init__(self):
            self.datasets = [1, 2, 3]

    ds = EmptyDataset()
    with pytest.raises(ValueError):
        transform_concat_ds(ds, OrderedDict())


def test_method_not_available(base_concat_ds):
    transforms = OrderedDict([('this_method_is_not_real', {'indeed': None})])
    with pytest.raises(AttributeError):
        transform_concat_ds(base_concat_ds, transforms)


def test_transform_base_callable(base_concat_ds):
    pass


def test_transform_base_method(base_concat_ds):
    transforms = OrderedDict([
        ("resample", {"sfreq": 50}),
    ])
    transform_concat_ds(base_concat_ds, transforms)
    assert base_concat_ds.datasets[0].raw.info['sfreq'] == 50


def test_transform_windows_callable(windows_concat_ds):
    pass


def test_transform_windows_method(windows_concat_ds):
    transforms = OrderedDict([
        ("filter", {"l_freq": 7, "h_freq": 13}),
    ])
    raw_window = windows_concat_ds[0][0]
    transform_concat_ds(windows_concat_ds, transforms)
    assert not np.array_equal(raw_window, windows_concat_ds[0][0])

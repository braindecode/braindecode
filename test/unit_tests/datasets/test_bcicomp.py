# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD (3-clause)

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from braindecode.datasets import BCICompetitionIVDataset4


@pytest.fixture(scope='module')
def input_data():
    rng = np.random.RandomState(123)
    input_data = {
        'sub1_comp.mat': {
            'train_data': rng.random((120, 3)),
            'train_dg': np.repeat(rng.random((3, 5)), 40, axis=0),
            'test_data': rng.random((120, 3)),
        },
        'sub1_testlabels.mat': {
            'test_dg': np.repeat(rng.random((3, 5)), 40, axis=0)
        },
        'sub2_comp.mat': {
            'train_data': rng.random((120, 3)),
            'train_dg': np.repeat(rng.random((3, 5)), 40, axis=0),
            'test_data': rng.random((120, 3)),
        },
        'sub2_testlabels.mat': {
            'test_dg': np.repeat(rng.random((3, 5)), 40, axis=0)
        },
        'sub3_comp.mat': {
            'train_data': rng.random((160, 3)),
            'train_dg': np.repeat(rng.random((4, 5)), 40, axis=0),
            'test_data': rng.random((120, 3)),
        },
        'sub3_testlabels.mat': {
            'test_dg': np.repeat(rng.random((3, 5)), 40, axis=0)
        },
    }
    return input_data


def _mock_loadmat(x, values):
    file_name = x.split('/')[-1]
    return values[file_name]


def _mock_download(self):
    return 'abc'


def _validate_dataset(ds, subject, input_data, is_train_dataset):
    file_name = f'sub{subject}_comp.mat'
    if is_train_dataset:
        expected_description = {
            'subject': f'{subject}', 'file_name': file_name, 'session': 'train'
        }
        prefix = 'train'
        file_name_array = file_name
    else:
        expected_description = {
            'subject': f'{subject}', 'file_name': file_name, 'session': 'test'
        }
        prefix = 'test'
        file_name_array = file_name.replace('comp', 'testlabels')

    expected_channel_types = 3 * ['ecog'] + 5 * ['misc']
    expected_ch_names = [
        '0', '1', '2', 'target_0', 'target_1', 'target_2', 'target_3', 'target_4'
    ]

    expected_array = np.full_like(input_data[file_name_array][f'{prefix}_dg'], np.nan)
    expected_array[::40] = input_data[file_name_array][f'{prefix}_dg'][::40]
    expected_array = np.concatenate(
        [input_data[file_name][f'{prefix}_data'], expected_array], axis=1
    )
    np.testing.assert_almost_equal(ds.raw.get_data(), expected_array.T)

    assert ds.description.to_dict() == expected_description
    assert ds.raw.get_channel_types() == expected_channel_types
    assert ds.raw.ch_names == expected_ch_names
    np.testing.assert_almost_equal(ds.raw.get_data(), expected_array.T)


@patch.object(BCICompetitionIVDataset4, 'download', _mock_download)
def test_bci_competition_iv_dataset_4(input_data):
    with patch('braindecode.datasets.bcicomp.loadmat',
               side_effect=partial(_mock_loadmat, values=input_data)):
        bci_comp_ds = BCICompetitionIVDataset4()
    assert len(bci_comp_ds.datasets) == 6
    for subject in range(1, 4):
        _validate_dataset(bci_comp_ds.datasets[(subject - 1) * 2], subject, input_data,
                          is_train_dataset=True)
        _validate_dataset(bci_comp_ds.datasets[((subject - 1) * 2) + 1], subject, input_data,
                          is_train_dataset=False)


@patch.object(BCICompetitionIVDataset4, 'download', _mock_download)
def test_wrong_subjects_ids():
    with pytest.raises(
            ValueError,
            match='Wrong subject_ids parameter. Possible values: \\[1, 2, 3\\]. Provided \\[5\\].'
    ):
        BCICompetitionIVDataset4(5)


@patch.object(BCICompetitionIVDataset4, 'download', _mock_download)
def test_wrong_type_subjects_ids():
    with pytest.raises(
            ValueError,
            match='Wrong subject_ids format. Expected types: None, list, tuple, int.'
    ):
        BCICompetitionIVDataset4(5.)

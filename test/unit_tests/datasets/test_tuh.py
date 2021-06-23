# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

from unittest import mock
import numpy as np
import mne

from braindecode.datasets.tuh import (
    _parse_description_from_file_path, _create_chronological_description,
    TUHAbnormal, TUH)


def test_parse_from_tuh_file_path():
    file_path = ("v2.0.0/edf/01_tcp_ar/000/00000021/"
                 "s004_2013_08_15/00000021_s004_t000.edf")
    description = _parse_description_from_file_path(file_path)
    assert len(description) == 9
    assert description['path'] == file_path
    assert description['year'] == 2013
    assert description['month'] == 8
    assert description['day'] == 15
    assert description['subject'] == 21
    assert description['session'] == 4
    assert description['segment'] == 0
    assert description['reference'] == 'ar'
    assert description['version'] == 'v2.0.0'


def test_parse_from_tuh_abnormal_file_path():
    file_path = ("v2.0.0/edf/eval/abnormal/01_tcp_ar/107/00010782/"
                 "s002_2013_10_05/00010782_s002_t001.edf")
    additional_description = (
        TUHAbnormal._parse_additional_description_from_file_path(file_path))
    assert len(additional_description) == 2
    assert additional_description['pathological']
    assert not additional_description['train']

    file_path = ("v2.0.0/edf/train/normal/01_tcp_ar/107/00010782/"
                 "s002_2013_10_05/00010782_s002_t001.edf")
    additional_description = (
        TUHAbnormal._parse_additional_description_from_file_path(file_path))
    assert len(additional_description) == 2
    assert not additional_description['pathological']
    assert additional_description['train']


def test_sort_chronologically():
    file_paths = [
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/"
        "00010831_s002_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s009_2011_09_12/"
        "00000068_s009_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/"
        "00000016_s004_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010839/s001_2013_11_22/"
        "00010839_s001_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s008_2010_09_28/"
        "00000068_s008_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/"
        "00010831_s002_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s005_2013_07_12/"
        "00000016_s005_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010810/s001_2013_10_03/"
        "00010810_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010831/s001_2013_10_03/"
        "00010831_s001_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000019/s002_2013_07_18/"
        "00000019_s002_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010816/s001_2013_10_03/"
        "00010816_s001_t001.edf",
    ]
    description = _create_chronological_description(file_paths)
    expected = [
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s008_2010_09_28/"
        "00000068_s008_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s009_2011_09_12/"
        "00000068_s009_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/"
        "00000016_s004_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s005_2013_07_12/"
        "00000016_s005_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000019/s002_2013_07_18/"
        "00000019_s002_t001.edf",

        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010810/s001_2013_10_03/"
        "00010810_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010816/s001_2013_10_03/"
        "00010816_s001_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010831/s001_2013_10_03/"
        "00010831_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/"
        "00010831_s002_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/"
        "00010831_s002_t001.edf",

        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010839/s001_2013_11_22/"
        "00010839_s001_t000.edf",
    ]
    for p1, p2 in zip(expected, description.T.path):
        assert p1 == p2




file_paths = [
    "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/00010831_s002_t001.edf",
    "v2.0.0/edf/eval/abnormal/01_tcp_ar/10/00010032/s001_2011_01_30/00010031_s001_t002.edf",
]
rng = np.random.RandomState(42)
def fake_raw():
    print("calling me")
    data = rng.randn(2, 1000)
    info = mne.create_info(['1', '2'], 10, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    return raw

@mock.patch('glob.glob', return_value=file_paths)
@mock.patch('mne.io.read_raw_edf', return_v alue=fake_raw())
def test_tuh(mock_glob, mock_mne):
    tuh = TUH('')
    assert True

@mock.patch('glob.glob', return_value=file_paths)
@mock.patch('mne.io.read_raw_edf', return_value=fake_raw())
def test_tuh_abnormal(mock_glob, mock_mne):
    tuh_ab = TUHAbnormal('')
    assert True

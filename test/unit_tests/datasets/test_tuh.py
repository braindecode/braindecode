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
    assert len(additional_description) == 3
    assert additional_description['pathological']
    assert not additional_description['train']
    assert additional_description['version'] == 'v2.0.0'

    file_path = ("v2.0.0/edf/train/normal/01_tcp_ar/107/00010782/"
                 "s002_2013_10_05/00010782_s002_t001.edf")
    additional_description = (
        TUHAbnormal._parse_additional_description_from_file_path(file_path))
    assert len(additional_description) == 3
    assert not additional_description['pathological']
    assert additional_description['train']
    assert additional_description['version'] == 'v2.0.0'


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


TUH_EEG_PATHS = {
    # These are actual file paths and edf headers from the TUH EEG Corpus (v1.1.0 and v1.2.0)
    'tuh_eeg/v1.1.0/edf/01_tcp_ar/000/00000000/s001_2015_12_30/00000000_s001_t000.edf': b'0       00000000 M 01-JAN-1978 00000000 Age:37                                          ',  # noqa E501
    'tuh_eeg/v1.1.0/edf/01_tcp_ar/099/00009932/s004_2014_09_30/00009932_s004_t013.edf': b'0       00009932 F 01-JAN-1961 00009932 Age:53                                          ',  # noqa E501
    'tuh_eeg/v1.1.0/edf/02_tcp_le/000/00000058/s001_2003_02_05/00000058_s001_t000.edf': b'0       00000058 M 01-JAN-2003 00000058 Age:0.0109                                      ',  # noqa E501
    'tuh_eeg/v1.1.0/edf/03_tcp_ar_a/123/00012331/s003_2014_12_14/00012331_s003_t002.edf': b'0       00012331 M 01-JAN-1975 00012331 Age:39                                          ',  # noqa E501
    'tuh_eeg/v1.2.0/edf/03_tcp_ar_a/149/00014928/s004_2016_01_15/00014928_s004_t007.edf': b'0       00014928 F 01-JAN-1933 00014928 Age:83                                          ',  # noqa E501
}
TUH_EEG_ABNORMAL_PATHS = {
    # these are actual file paths and edf headers from TUH Abnormal EEG Corpus (v2.0.0)
    'tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/078/00007871/s001_2011_07_05/00007871_s001_t001.edf': b'0       00007871 F 01-JAN-1988 00007871 Age:23                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/097/00009777/s001_2012_09_17/00009777_s001_t000.edf': b'0       00009777 M 01-JAN-1986 00009777 Age:26                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/083/00008393/s002_2012_02_21/00008393_s002_t000.edf': b'0       00008393 M 01-JAN-1960 00008393 Age:52                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/012/00001200/s003_2010_12_06/00001200_s003_t000.edf': b'0       00001200 M 01-JAN-1963 00001200 Age:47                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/eval/abnormal/01_tcp_ar/059/00005932/s004_2013_03_14/00005932_s004_t000.edf': b'0       00005932 M 01-JAN-1963 00005932 Age:50                                          ',  # noqa E501
}


def _fake_raw(*args, **kwargs):
    data = np.random.randn(2, 1000)
    info = mne.create_info(ch_names=['1', '2'], sfreq=10, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)
    return raw


def _get_header(*args):
    all_paths = {**TUH_EEG_PATHS, **TUH_EEG_ABNORMAL_PATHS}
    return all_paths[args[0]]


@mock.patch('glob.glob', return_value=TUH_EEG_PATHS)
@mock.patch('mne.io.read_raw_edf', new=_fake_raw)
@mock.patch('braindecode.datasets.tuh._read_edf_header', new=_get_header)
def test_tuh(mock_glob):
    tuh = TUH('')
    assert len(tuh.datasets) == 5
    assert tuh.description.shape == (5, 13)
    assert len(tuh) == 5000
    assert tuh.description.age.to_list() == [0, 53, 39, 37, 83]
    assert tuh.description.gender.to_list() == ['M', 'F', 'M', 'M', 'F']
    assert tuh.description.version.to_list() == ['v1.1.0', 'v1.1.0', 'v1.1.0', 'v1.1.0', 'v1.2.0']
    assert tuh.description.year.to_list() == [2003, 2014, 2014, 2015, 2016]
    assert tuh.description.month.to_list() == [2, 9, 12, 12, 1]
    assert tuh.description.day.to_list() == [5, 30, 14, 30, 15]
    assert tuh.description.subject.to_list() == [58, 9932, 12331, 0, 14928]
    assert tuh.description.session.to_list() == [1, 4, 3, 1, 4]
    assert tuh.description.segment.to_list() == [0, 13, 2, 0, 7]
    assert tuh.description.reference.to_list() == ['le', 'ar', 'ar', 'ar', 'ar']
    assert tuh.description.sfreq.to_list() == [10, 10, 10, 10, 10]
    assert tuh.description.n_samples.to_list() == [1000, 1000, 1000, 1000, 1000]


@mock.patch('glob.glob', return_value=TUH_EEG_ABNORMAL_PATHS)
@mock.patch('mne.io.read_raw_edf', new=_fake_raw)
@mock.patch('braindecode.datasets.tuh._read_edf_header', new=_get_header)
def test_tuh_abnormal(mock_glob):
    tuh_ab = TUHAbnormal('')
    assert len(tuh_ab.datasets) == 5
    assert tuh_ab.description.shape == (5, 15)
    assert tuh_ab.description.version.to_list() == [
        'v2.0.0', 'v2.0.0', 'v2.0.0', 'v2.0.0', 'v2.0.0']
    assert tuh_ab.description.pathological.to_list() == [True, False, True, False, True]
    assert tuh_ab.description.train.to_list() == [True, True, True, True, False]

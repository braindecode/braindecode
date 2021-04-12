# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

from braindecode.datasets.tuh import TUHAbnormal


def test_parse_from_file_path():
    # expect filenames as v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/s004_2013_08_15/00000021_s004_t000.edf
    #              version/file type/data_split/label/EEG reference/subset/subject/recording session/file
    # see https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/_AAREADME.txt
    file_path = "v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/s004_2013_08_15/00000021_s004_t000.edf"
    pathological, data_split, subject_id = (
        TUHAbnormal._parse_properties_from_file_path(file_path))
    assert not pathological
    assert data_split == "train"
    assert subject_id == 21
    file_path = "v2.0.0/edf/eval/abnormal/01_tcp_ar/107/00010782/s002_2013_10_05/00010782_s002_t001.edf"
    pathological, data_split, subject_id = (
        TUHAbnormal._parse_properties_from_file_path(file_path))
    assert pathological
    assert data_split == "eval"
    assert subject_id == 10782


def test_sort_chronologically():
    file_paths = [
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/00010831_s002_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s009_2011_09_12/00000068_s009_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/00000016_s004_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010839/s001_2013_11_22/00010839_s001_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s008_2010_09_28/00000068_s008_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/00010831_s002_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s005_2013_07_12/00000016_s005_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010810/s001_2013_10_03/00010810_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010831/s001_2013_10_03/00010831_s001_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000019/s002_2013_07_18/00000019_s002_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010816/s001_2013_10_03/00010816_s001_t001.edf",
    ]
    sorted_file_paths = TUHAbnormal.sort_chronologically(file_paths)
    expected = [
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s008_2010_09_28/00000068_s008_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s009_2011_09_12/00000068_s009_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/00000016_s004_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s005_2013_07_12/00000016_s005_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000019/s002_2013_07_18/00000019_s002_t001.edf",

        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010810/s001_2013_10_03/00010810_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010816/s001_2013_10_03/00010816_s001_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010831/s001_2013_10_03/00010831_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/00010831_s002_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/00010831_s002_t001.edf",

        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010839/s001_2013_11_22/00010839_s001_t000.edf",
    ]
    for p1, p2 in zip(expected, sorted_file_paths):
        assert p1 == p2

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import platform
from datetime import datetime
from typing import Literal

import pytest

from braindecode.datasets.tuh import (
    TUHAbnormal,
    _create_description,
    _parse_description_from_file_path,
    _sort_chronologically,
    _TUHAbnormalMock,
    _TUHEventsMock,
    _TUHMock,
)


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_parse_from_tuh_file_path():
    file_path = (
        "v1.2.0/edf/01_tcp_ar/000/00000021/" "s004_2013_08_15/00000021_s004_t000.edf"
    )
    description = _parse_description_from_file_path(file_path)
    assert len(description) == 8
    assert description["path"] == file_path
    assert description["year"] == 2013
    assert description["month"] == 8
    assert description["day"] == 15
    assert description["subject"] == 21
    assert description["session"] == 4
    assert description["segment"] == 0
    assert description["version"] == "v1.2.0"


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_parse_from_tuh_abnormal_file_path():
    file_path = (
        "v2.0.0/edf/eval/abnormal/01_tcp_ar/107/00010782/"
        "s002_2013_10_05/00010782_s002_t001.edf"
    )
    additional_description = TUHAbnormal._parse_additional_description_from_file_path(
        file_path
    )
    assert len(additional_description) == 3
    assert additional_description["pathological"]
    assert not additional_description["train"]
    assert additional_description["version"] == "v2.0.0"

    file_path = (
        "v2.0.0/edf/train/normal/01_tcp_ar/107/00010782/"
        "s002_2013_10_05/00010782_s002_t001.edf"
    )
    additional_description = TUHAbnormal._parse_additional_description_from_file_path(
        file_path
    )
    assert len(additional_description) == 3
    assert not additional_description["pathological"]
    assert additional_description["train"]
    assert additional_description["version"] == "v2.0.0"


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_sort_chronologically():
    file_paths = [
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/"
        "00010831_s001_t001.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s009_2011_09_12/"
        "00000068_s009_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/"
        "00000016_s004_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010839/s001_2013_11_22/"
        "00010839_s001_t000.edf",
        "v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s008_2010_09_28/"
        "00000068_s008_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010831/s001_2013_10_03/"
        "00010831_s001_t000.edf",
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
    description = _create_description(file_paths, version="v2.0.0", ds_name="abnormal")
    description = _sort_chronologically(description)
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
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010831/s001_2013_10_03/"
        "00010831_s001_t000.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010832/s001_2013_10_03/"
        "00010831_s001_t001.edf",
        "v2.0.0/edf/train/normal/01_tcp_ar/108/00010839/s001_2013_11_22/"
        "00010839_s001_t000.edf",
    ]
    for p1, p2 in zip(expected, description.T.path):
        assert p1 == p2


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
@pytest.mark.parametrize("version", ["v1.1.0", "v1.2.0"])
def test_tuh(version: Literal["v1.1.0"] | Literal["v1.2.0"]):
    tuh = _TUHMock(
        path="",
        n_jobs=1,  # required for test to work. mocking seems to fail otherwise
        version=version,
    )
    files_count = tuh._expected_files_count
    assert files_count is not None
    assert len(tuh.datasets) == files_count
    assert tuh.description.shape == (files_count, 10)
    assert len(tuh) == 3600 * files_count
    assert tuh.description.age.to_list() in [[0, 53, 39, 37], [83]]
    assert tuh.description.gender.to_list() in [["M", "F", "M", "M"], ["F"]]
    assert tuh.description.version.to_list() == [version] * files_count
    assert tuh.description.year.to_list() in [[2003, 2014, 2014, 2015], [2016]]
    assert tuh.description.month.to_list() in [[2, 9, 12, 12], [1]]
    assert tuh.description.day.to_list() in [[5, 30, 14, 30], [15]]
    assert tuh.description.subject.to_list() in [[58, 9932, 12331, 0], [14928]]
    assert tuh.description.session.to_list() in [[1, 4, 3, 1], [4]]
    assert tuh.description.segment.to_list() in [[0, 13, 2, 0], [7]]
    x, y = tuh[0]
    assert x.shape == (21, 1)
    assert y is None

    for ds, (_, desc) in zip(tuh.datasets, tuh.description.iterrows()):
        assert isinstance(ds.raw.info["meas_date"], datetime)
        assert ds.raw.info["meas_date"].year == desc["year"]
        assert ds.raw.info["meas_date"].month == desc["month"]
        assert ds.raw.info["meas_date"].day == desc["day"]

    tuh = _TUHMock(
        path="",
        target_name="gender",
        recording_ids=[0],
        n_jobs=1,
        version=version,
    )
    assert len(tuh.datasets) == 1
    x, y = tuh[0]
    assert y == "F"
    x, y = tuh[-1]
    assert y == "F"


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
@pytest.mark.parametrize("version", ["v2.0.0"])
def test_tuh_abnormal(version):
    tuh_ab = _TUHAbnormalMock(
        path="",
        add_physician_reports=True,
        n_jobs=1,  # required for test to work. mocking seems to fail otherwise
        version="v2.0.0",
    )
    files_count = tuh_ab._expected_files_count
    assert files_count is not None
    assert len(tuh_ab.datasets) == files_count
    assert tuh_ab.description.shape == (files_count, 13)
    assert tuh_ab.description.version.to_list() == ["v2.0.0"] * files_count
    assert tuh_ab.description.pathological.to_list() == [True, False, True, False, True]
    assert tuh_ab.description.train.to_list() == [True, True, True, True, False]
    assert tuh_ab.description.report.to_list() == ["simple_test"] * files_count
    x, y = tuh_ab[0]
    assert x.shape == (21, 1)
    assert y
    x, y = tuh_ab[-1]
    assert y

    for ds, (_, desc) in zip(tuh_ab.datasets, tuh_ab.description.iterrows()):
        assert isinstance(ds.raw.info["meas_date"], datetime)
        assert ds.raw.info["meas_date"].year == desc["year"]
        assert ds.raw.info["meas_date"].month == desc["month"]
        assert ds.raw.info["meas_date"].day == desc["day"]

    tuh_ab = _TUHAbnormalMock(
        path="",
        target_name="age",
        n_jobs=1,
    )
    x, y = tuh_ab[-1]
    assert y == 50
    for ds in tuh_ab.datasets:
        ds.target_name = "gender"
    x, y = tuh_ab[0]
    assert y == "M"


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
@pytest.mark.parametrize("version", ["v2.0.1"])
def test_tuh_events(version):
    tuh_ev = _TUHEventsMock(path="", n_jobs=1, version=version)
    files_count = tuh_ev._expected_files_count
    description = tuh_ev.description
    assert files_count is not None
    assert len(tuh_ev.datasets) == files_count
    assert description.shape == (files_count, 13)
    assert len(tuh_ev) == 3600 * files_count
    assert description.subject.to_list() == ["000", "001", "aaaaaaar"]
    assert description.version.to_list() == [version] * files_count
    assert description.session.to_list() == [1, 1, 1]
    assert description.split.to_list() == ["eval", "eval", "train"]
    assert description.event_prefix.to_list() == ["bckg", "pled", None]
    assert description.run.to_list() == [0, 2, 0]
    assert description.age.to_list() == [36, 68, 19]
    assert description.gender.to_list() == ["F", "F", "F"]

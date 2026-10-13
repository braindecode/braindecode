# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import platform
from datetime import datetime
from pathlib import PureWindowsPath
from typing import Literal

import pandas as pd
import pytest

from braindecode.datasets.tuh import (
    _TUH_EEG_ABNORMAL_PATHS,
    _TUH_EEG_EVENTS_PATHS,
    _TUH_EEG_PATHS,
    TUHAbnormal,
    _create_description,
    _get_header,
    _parse_description_from_file_path,
    _sort_chronologically,
    _TUHAbnormalMock,
    _TUHEventsMock,
    _TUHMock,
)


def test_mock_header_accepts_posix_and_windows_paths():
    """All shared mock headers accept either separator on every host OS."""
    for corpus in (_TUH_EEG_PATHS, _TUH_EEG_ABNORMAL_PATHS, _TUH_EEG_EVENTS_PATHS):
        for headers in corpus.values():
            for path, expected_header in headers.items():
                assert _get_header(path) == expected_header
                assert _get_header(str(PureWindowsPath(path))) == expected_header


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_parse_from_tuh_file_path():
    file_path = (
        "v1.2.0/edf/01_tcp_ar/000/00000021/" "s004_2013_08_15/00000021_s004_t000.edf"
    )
    description = _parse_description_from_file_path(file_path, version="v1.2.0", ds_name="tuh")
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
    assert expected == description.T.path.to_list()


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
    assert set(description.columns) == {
        "path",
        "subject",
        "version",
        "session",
        "split",
        "event_prefix",
        "run",
        "age",
        "gender",
        "year",
        "month",
        "day",
    }
    assert len(tuh_ev) == 3600 * files_count
    assert description.subject.to_list() == ["000", "001", "aaaaaaar"]
    assert description.version.to_list() == [version] * files_count
    assert description.session.to_list() == [1, 1, 1]
    assert description.split.to_list() == ["eval", "eval", "train"]
    event_prefixes = description.event_prefix.to_list()
    assert event_prefixes[:2] == ["bckg", "pled"]
    assert pd.isna(event_prefixes[2])
    assert description.run.to_list() == [0, 2, 0]
    assert description.age.to_list() == [36, 68, 19]
    assert description.gender.to_list() == ["F", "F", "F"]


@pytest.mark.skipif(platform.system() == "Windows", reason="Not supported on Windows")
def test_tuh_shared_montage_matches_a_fresh_one():
    """Every recording gets the positions of a fresh standard_1005 montage."""
    import mne

    from braindecode.datasets.tuh import _standard_1005
    from braindecode.util import resolve_montage_name

    tuh = _TUHMock(
        path="", rename_channels=True, set_montage=True, n_jobs=1, version="v1.1.0"
    )
    fresh = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    assert _standard_1005() == fresh
    for ds in tuh.datasets:
        ref = ds.raw.copy().set_montage(fresh, on_missing="ignore")
        assert ds.raw.info["dig"] == ref.info["dig"]


@pytest.mark.parametrize("date", [{"year": 2013, "month": 8, "day": 15}, {}])
def test_read_date_side_file_roundtrip(tmp_path, date):
    """_read_date returns what _read_date wrote (the `_date.txt` beside each EDF)."""
    from braindecode.datasets.tuh import _read_date

    edf = str(tmp_path / "aaaaaaav_s004_t000.edf")
    pd.Series(date).to_json(edf.replace(".edf", "_date.txt"))
    assert _read_date(edf) == date
    assert all(type(v) is int for v in _read_date(edf).values())


@pytest.mark.parametrize(
    "date,year", [({"year": 1899, "month": 12, "day": 30}, 1899), ({}, 1)]
)
def test_tuh_date_fif_cannot_store_stays_in_description(tmp_path, date, year):
    """De-identified (1899-12-30) and undated recordings save; the date stays in the description."""
    from unittest import mock

    from braindecode.datasets import BaseConcatDataset
    from braindecode.datasets.tuh import TUH, _fake_raw
    from braindecode.datautil import load_concat_dataset

    path = "tuh_abnormal_eeg/v3.0.1/edf/train/normal/01_tcp_ar/aaaaaaav_s004_t000.edf"
    header = b"0       aaaaaaav M 01-JAN-1961 aaaaaaav Age:53" + b" " * 40
    with (
        mock.patch("mne.io.read_raw_edf", new=_fake_raw),
        mock.patch("braindecode.datasets.tuh._read_edf_header", return_value=header),
    ):
        ds = TUH._create_dataset(pd.Series({"path": path, **date}), None, False, False, False, False)
    BaseConcatDataset([ds]).save(str(tmp_path))  # FIF refuses dates before 1901-12-13
    reloaded = load_concat_dataset(tmp_path, preload=False)
    assert reloaded.description["year"].tolist() == [year]
    assert reloaded.datasets[0].raw.info["meas_date"] is None

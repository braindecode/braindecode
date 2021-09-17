"""
Dataset classes for the Temple University Hospital (TUH) EEG Corpus and the
TUH Abnormal EEG Corpus.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import re
import os
import glob
from unittest import mock
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import mne
from joblib import Parallel, delayed

from .base import BaseDataset, BaseConcatDataset


class TUH(BaseConcatDataset):
    """Temple University Hospital (TUH) EEG Corpus
    (www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg).

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    n_jobs: int
        Number of jobs to be used to read files in parallel.
    """
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False, n_jobs=1):
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(os.path.join(path, '**/*.edf'), recursive=True)
        descriptions = _create_chronological_description(file_paths)
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            descriptions = descriptions[recording_ids]
        # this is the second loop (slow)
        # create datasets gathering more info about the files touching them
        # reading the raws and potentially preloading the data
        # disable joblib for tests. mocking seems to fail otherwise
        if n_jobs == 1:
            base_datasets = [self._create_dataset(
                descriptions[i], target_name, preload, add_physician_reports)
                for i in descriptions.columns]
        else:
            base_datasets = Parallel(n_jobs)(delayed(
                self._create_dataset)(
                descriptions[i], target_name, preload, add_physician_reports
            ) for i in descriptions.columns)
        super().__init__(base_datasets)

    @staticmethod
    def _create_dataset(description, target_name, preload,
                        add_physician_reports):
        file_path = description.loc['path']

        # parse age and gender information from EDF header
        age, gender = _parse_age_and_gender_from_edf_header(file_path)
        raw = mne.io.read_raw_edf(file_path, preload=preload)

        # Use recording date from path as EDF header is sometimes wrong
        meas_date = datetime(1, 1, 1, tzinfo=timezone.utc) \
            if raw.info['meas_date'] is None else raw.info['meas_date']
        raw.info['meas_date'] = meas_date.replace(
            *description[['year', 'month', 'day']])

        # read info relevant for preprocessing from raw without loading it
        d = {
            'age': int(age),
            'gender': gender,
        }
        if add_physician_reports:
            physician_report = _read_physician_report(file_path)
            d['report'] = physician_report
        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        base_dataset = BaseDataset(raw, description,
                                   target_name=target_name)
        return base_dataset


def _create_chronological_description(file_paths):
    # this is the first loop (fast)
    descriptions = []
    for file_path in file_paths:
        description = _parse_description_from_file_path(file_path)
        descriptions.append(pd.Series(description))
    descriptions = pd.concat(descriptions, axis=1)
    # order descriptions chronologically
    descriptions.sort_values(
        ["year", "month", "day", "subject", "session", "segment"],
        axis=1, inplace=True)
    # https://stackoverflow.com/questions/42284617/reset-column-index-pandas
    descriptions = descriptions.T.reset_index(drop=True).T
    return descriptions


def _parse_description_from_file_path(file_path):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # expect file paths as tuh_eeg/version/file_type/reference/data_split/
    #                          subject/recording session/file
    # e.g.                 tuh_eeg/v1.1.0/edf/01_tcp_ar/027/00002729/
    #                          s001_2006_04_12/00002729_s001.edf
    version = tokens[-7]
    year, month, day = tokens[-2].split('_')[1:]
    subject_id = tokens[-3]
    session = tokens[-2].split('_')[0]
    segment = tokens[-1].split('_')[-1].split('.')[-2]
    return {
        'path': file_path,
        'version': version,
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'subject': int(subject_id),
        'session': int(session[1:]),
        'segment': int(segment[1:]),
    }


def _read_physician_report(file_path):
    directory = os.path.dirname(file_path)
    txt_file = glob.glob(os.path.join(directory, '**/*.txt'), recursive=True)
    # check that there is at most one txt file in the same directory
    assert len(txt_file) in [0, 1]
    report = ''
    if txt_file:
        txt_file = txt_file[0]
        # somewhere in the corpus, encoding apparently changed
        # first try to read as utf-8, if it does not work use latin-1
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                report = f.read()
        except UnicodeDecodeError:
            with open(txt_file, 'r', encoding='latin-1') as f:
                report = f.read()
    return report


def _read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def _parse_age_and_gender_from_edf_header(file_path):
    header = _read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender


class TUHAbnormal(TUH):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.
    see www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'pathological', 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    """
    def __init__(self, path, recording_ids=None, target_name='pathological',
                 preload=False, add_physician_reports=False, n_jobs=1):
        super().__init__(path=path, recording_ids=recording_ids,
                         preload=preload, target_name=target_name,
                         add_physician_reports=add_physician_reports,
                         n_jobs=n_jobs)
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = (
                self._parse_additional_description_from_file_path(file_path))
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)
        # expect paths as version/file type/data_split/pathology status/
        #                     reference/subset/subject/recording session/file
        # e.g.            v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/
        #                     s004_2013_08_15/00000021_s004_t000.edf
        assert ('abnormal' in tokens or 'normal' in tokens), (
            'No pathology labels found.')
        assert ('train' in tokens or 'eval' in tokens), (
            'No train or eval set information found.')
        return {
            'version': tokens[-9],
            'train': 'train' in tokens,
            'pathological': 'abnormal' in tokens,
        }


def _fake_raw(*args, **kwargs):
    sfreq = 10
    ch_names = [
        'EEG A1-REF', 'EEG A2-REF',
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
    duration_min = 6
    data = np.random.randn(len(ch_names), duration_min * sfreq * 60)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)
    return raw


def _get_header(*args, **kwargs):
    all_paths = {**_TUH_EEG_PATHS, **_TUH_EEG_ABNORMAL_PATHS}
    return all_paths[args[0]]


_TUH_EEG_PATHS = {
    # These are actual file paths and edf headers from the TUH EEG Corpus (v1.1.0 and v1.2.0)
    'tuh_eeg/v1.1.0/edf/01_tcp_ar/000/00000000/s001_2015_12_30/00000000_s001_t000.edf': b'0       00000000 M 01-JAN-1978 00000000 Age:37                                          ',  # noqa E501
    'tuh_eeg/v1.1.0/edf/01_tcp_ar/099/00009932/s004_2014_09_30/00009932_s004_t013.edf': b'0       00009932 F 01-JAN-1961 00009932 Age:53                                          ',  # noqa E501
    'tuh_eeg/v1.1.0/edf/02_tcp_le/000/00000058/s001_2003_02_05/00000058_s001_t000.edf': b'0       00000058 M 01-JAN-2003 00000058 Age:0.0109                                      ',  # noqa E501
    'tuh_eeg/v1.1.0/edf/03_tcp_ar_a/123/00012331/s003_2014_12_14/00012331_s003_t002.edf': b'0       00012331 M 01-JAN-1975 00012331 Age:39                                          ',  # noqa E501
    'tuh_eeg/v1.2.0/edf/03_tcp_ar_a/149/00014928/s004_2016_01_15/00014928_s004_t007.edf': b'0       00014928 F 01-JAN-1933 00014928 Age:83                                          ',  # noqa E501
}
_TUH_EEG_ABNORMAL_PATHS = {
    # these are actual file paths and edf headers from TUH Abnormal EEG Corpus (v2.0.0)
    'tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/078/00007871/s001_2011_07_05/00007871_s001_t001.edf': b'0       00007871 F 01-JAN-1988 00007871 Age:23                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/097/00009777/s001_2012_09_17/00009777_s001_t000.edf': b'0       00009777 M 01-JAN-1986 00009777 Age:26                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/083/00008393/s002_2012_02_21/00008393_s002_t000.edf': b'0       00008393 M 01-JAN-1960 00008393 Age:52                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/012/00001200/s003_2010_12_06/00001200_s003_t000.edf': b'0       00001200 M 01-JAN-1963 00001200 Age:47                                          ',  # noqa E501
    'tuh_abnormal_eeg/v2.0.0/edf/eval/abnormal/01_tcp_ar/059/00005932/s004_2013_03_14/00005932_s004_t000.edf': b'0       00005932 M 01-JAN-1963 00005932 Age:50                                          ',  # noqa E501
}


class _TUHMock(TUH):
    """Mocked class for testing and examples."""
    @mock.patch('glob.glob', return_value=_TUH_EEG_PATHS.keys())
    @mock.patch('mne.io.read_raw_edf', new=_fake_raw)
    @mock.patch('braindecode.datasets.tuh._read_edf_header',
                new=_get_header)
    def __init__(self, mock_glob, path, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False, n_jobs=1):
        super().__init__(path=path, recording_ids=recording_ids,
                         target_name=target_name, preload=preload,
                         add_physician_reports=add_physician_reports,
                         n_jobs=n_jobs)


class _TUHAbnormalMock(TUHAbnormal):
    """Mocked class for testing and examples."""
    @mock.patch('glob.glob', return_value=_TUH_EEG_ABNORMAL_PATHS.keys())
    @mock.patch('mne.io.read_raw_edf', new=_fake_raw)
    @mock.patch('braindecode.datasets.tuh._read_edf_header',
                new=_get_header)
    @mock.patch('braindecode.datasets.tuh._read_physician_report',
                return_value='simple_test')
    def __init__(self, mock_glob, mock_report, path, recording_ids=None,
                 target_name='pathological', preload=False,
                 add_physician_reports=False, n_jobs=1):
        super().__init__(path=path, recording_ids=recording_ids,
                         target_name=target_name, preload=preload,
                         add_physician_reports=add_physician_reports,
                         n_jobs=n_jobs)

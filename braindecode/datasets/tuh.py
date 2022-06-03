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
from warnings import warn
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
    indexer: None | callable
        When called, receives the path of the dataset and is supposed to return
        a list of absolute recording file paths. By default, glob is used to
        search for .edf files.
        Note that the order of the file paths matters.
    parser: None | callable
        When called, receives a list of absolute recording file paths and is
        supposed to return a list of dictionaries that describe the recordings.
        The dictionaries are required to contain {'path': file_path}.
        By default, extracts version, year, month, day, subject, session,
        segment.
        Note that the order of the descriptions matters.
    reader: None | callable
        When called, receives an absolute recording file path and is supposed
        to return a mne.io.Raw object. By default, uses mne.io.read_raw_edf with
        preload=False and verbose='ERROR'.
    selector: None | callable | list
        If callable, receives the description of one recording and is supposed
        to return a boolean whether or not to include this recording in the
        dataset.
        If a list, it is assumed to contain integer ids that are then used to
        slice descriptions. Exists to assure backwards compatibility with
        'recording_ids'.
    target_name: str
        Can be 'gender', or 'age'.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    load: bool
        If False, only creates the index of .edf file paths through indexer and
        the descriptions thereof through parser (fast).
        If True, additionally touches all .edf files and creates nme.io.Raw
        objects through reader (slow).
    n_jobs: int
        Number of jobs to be used to read files in parallel.
    recording_ids: list(int) | int
        Deprecated.
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later than the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    preload: bool
        Deprecated.
        If True, preload the data of the Raw objects.
    """
    def __init__(
            self, path, indexer=None, parser=None, reader=None, selector=None,
            target_name=None, add_physician_reports=False, load=True, n_jobs=1,
            # deprecated
            recording_ids=None, preload=False,
    ):
        # deprecated
        if recording_ids is not None or preload:
            warn(
                "Arguments 'recording_ids' and 'preload'"
                " are deprecated. For selecting recordings use 'selector'."
                " To change file reading parameters use 'reader'.")
        if recording_ids is not None and selector is not None:
            raise ValueError("Please only use 'selector'.")
        if recording_ids is not None:
            selector = recording_ids
        self._preload = preload
        # deprecated end
        self._path = path
        self._indexer = self._indexer if indexer is None else indexer
        self._parser = self._parser if parser is None else parser
        self._reader = self._reader if reader is None else reader
        self._selector = selector
        self._target_name = target_name
        self._add_physician_reports = add_physician_reports
        self._n_jobs = n_jobs
        self._file_paths = self._indexer(path)
        self._descriptions = self._parser(self._file_paths)
        assert 'path' in self._descriptions[0]
        if self._selector is not None and not callable(self._selector):
            self._descriptions = [self._descriptions[i] for i in self._selector]
        if load:
            self.load(n_jobs=self._n_jobs)

    @staticmethod
    def _indexer(path):
        return glob.glob(path + '**/*.edf', recursive=True)

    def _parser(self, descriptions):
        descriptions = [self._parse(d) for d in descriptions]
        key = ('year', 'month', 'day', 'subject', 'session', 'segment')
        descriptions = sorted(
            descriptions,
            key=lambda d: [d[k] for k in key],
        )
        return descriptions

    def _parse(self, path):
        # /tuh_eeg/v1.1.0/edf/02_tcp_le/000/00000013/s001_2002_09_03/
        #  00000013_s001_t002.edf
        pattern = os.sep.join([
            r'(v\d.\d.\d)',  # version
            r'\w+',
            r'\w+',
            r'\d+',
            r'(\d+)',  # subject
            r's(\d+)_(\d+)_(\d+)_(\d+)',  # session, year, month, day
            r'\d+_s(\d+)_t(\d+)\.edf$',  # session, segment
        ])
        pattern = re.compile(pattern)
        matches = re.findall(pattern, path)[0]
        (version, subject, session, year, month, day, _, segment) = matches
        d = {
            'path': path,
            'version': version,
            'year': int(year),
            'month': int(month),
            'day': int(day),
            'subject': int(subject),
            'session': int(session),
            'segment': int(segment),
        }
        return d

    def load(self, n_jobs):
        n_jobs = self._n_jobs if n_jobs == 1 else n_jobs
        args = (
            self._reader, self._selector,
            self._target_name, self._add_physician_reports,
            # deprecated
            self._preload,
        )
        if n_jobs == 1:
            datasets = [
                self._create_dataset(i, d, *args)
                for i, d in enumerate(self._descriptions)
            ]
        else:
            datasets = Parallel(n_jobs=n_jobs)(
                delayed(self._create_dataset)(i, d, *args)
                for i, d in enumerate(self._descriptions)
            )
        if self._selector is not None:
            datasets = [d for d in datasets if d is not None]
        super().__init__(datasets)

    @staticmethod
    def _create_dataset(
            i, d, reader, selector, target_name, add_physician_reports,
            # deprecated
            preload,
    ):
        age, gender = TUH._parse_age_and_gender_from_edf_header(d['path'])
        if add_physician_reports:
            report = TUH._read_physician_report(d['path'])
            d['report'] = report
        s = pd.Series({
            **d,
            'age': age,
            'gender': gender,
        }, name=i)
        if selector is not None and callable(selector):
            if not selector(s):
                return
        raw = reader(
            s,
            # deprecated
            preload=preload,
            verbose='ERROR',
        )
        return BaseDataset(raw, s, target_name=target_name)

    @staticmethod
    def _reader(d, preload, verbose):
        raw = mne.io.read_raw_edf(d.path, preload=preload, verbose=verbose)
        # Use recording date from path as EDF header is sometimes wrong
        meas_date = datetime(1, 1, 1, tzinfo=timezone.utc) \
            if raw.info['meas_date'] is None else raw.info['meas_date']
        raw.set_meas_date(meas_date.replace(
            *d[['year', 'month', 'day']]))
        return raw

    @staticmethod
    def _read_physician_report(file_path):
        directory = os.path.dirname(file_path)
        txt_file = glob.glob(
            os.path.join(directory, '**/*.txt'), recursive=True)
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

    @staticmethod
    def _read_edf_header(file_path):
        f = open(file_path, "rb")
        header = f.read(88)
        f.close()
        return header

    @staticmethod
    def _parse_age_and_gender_from_edf_header(file_path):
        header = TUH._read_edf_header(file_path)
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
    indexer: None | callable
        When called, receives the path of the dataset and is supposed to return
        a list of absolute recording file paths. By default, glob is used to
        search for .edf files.
        Note that the order of the file paths matters.
    parser: None | callable
        When called, receives a list of absolute recording file paths and is
        supposed to return a list of dictionaries that describe the recordings.
        The dictionaries are required to contain {'path': file_path}.
        By default, extracts version, year, month, day, subject, session,
        segment.
        Note that the order of the descriptions matters.
    reader: None | callable
        When called, receives an absolute recording file path and is supposed
        to return a mne.io.Raw object. By default, uses mne.io.read_raw_edf with
        preload=False and verbose='ERROR'.
    selector: None | callable | list
        If callable, receives the description of one recording and is supposed
        to return a boolean whether or not to include this recording in the
        dataset.
        If a list, it is assumed to contain integer ids that are then used to
        slice descriptions.
    target_name: str
        Can be 'gender', or 'age'.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    load: bool
        If False, only creates the index of .edf file paths through indexer and
        the descriptions thereof through parser (fast).
        If True, additionally touches all .edf files and creates nme.io.Raw
        objects through reader (slow).
    n_jobs: int
        Number of jobs to be used to read files in parallel.
    recording_ids: list(int) | int
        Deprecated.
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later than the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    preload: bool
        Deprecated.
        If True, preload the data of the Raw objects.
    """
    def __init__(
            self, path, indexer=None, parser=None, reader=None,
            selector=None, target_name='pathological',
            add_physician_reports=False, load=True, n_jobs=1,
            # deprecated
            recording_ids=None, preload=False,
    ):
        super().__init__(
            path=path, indexer=indexer, parser=parser, reader=reader,
            selector=selector, target_name=target_name,
            add_physician_reports=add_physician_reports,
            load=load, n_jobs=n_jobs,
            # deprecated
            recording_ids=recording_ids, preload=preload,
        )

    def _parse(self, path):
        # tuh_eeg_abnormal/v2.0.0/edf/eval/normal/01_tcp_ar/058/00005864/
        #  s001_2009_09_03/00005864_s001_t000.edf
        pattern = os.sep.join([
            r'(v\d.\d.\d)',  # version
            r'\w+',
            r'(\w+)',  # train
            r'(\w+)',  # pathological
            r'\w+',
            r'\d+',
            r'(\d+)',  # subject
            r's(\d+)_(\d+)_(\d+)_(\d+)',  # session, year, month, day
            r'\d+_s(\d+)_t(\d+)\.edf$',  # session, segment
        ])
        pattern = re.compile(pattern)
        matches = re.findall(pattern, path)[0]
        (version, train, pathological, subject,
         session, year, month, day, _, segment) = matches
        d = {
            'path': path,
            'version': version,
            'year': int(year),
            'month': int(month),
            'day': int(day),
            'subject': int(subject),
            'session': int(session),
            'segment': int(segment),
            # specific for abnormal
            'train': train == 'train',
            'pathological': pathological == 'abnormal',
        }
        return d


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
    @mock.patch('braindecode.datasets.tuh.TUH._read_edf_header',
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
    @mock.patch('braindecode.datasets.tuh.TUH._read_edf_header',
                new=_get_header)
    @mock.patch('braindecode.datasets.tuh.TUH._read_physician_report',
                return_value='simple_test')
    def __init__(self, mock_glob, mock_report, path, recording_ids=None,
                 target_name='pathological', preload=False,
                 add_physician_reports=False, n_jobs=1):
        super().__init__(path=path, recording_ids=recording_ids,
                         target_name=target_name, preload=preload,
                         add_physician_reports=add_physician_reports,
                         n_jobs=n_jobs)

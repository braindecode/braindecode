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

import pandas as pd
import mne

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
    """
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False):
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(os.path.join(path, '**/*.edf'), recursive=True)
        descriptions = _create_chronological_description(file_paths)
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            descriptions = descriptions[recording_ids]
        # create datasets gathering more info about the files touching them
        # reading the raws and potentially preloading the data
        base_datasets = self._create_datasets(
            descriptions, target_name, preload, add_physician_reports)
        super().__init__(base_datasets)

    @staticmethod
    def _create_datasets(descriptions, target_name, preload,
                         add_physician_reports):
        # this is the second loop (slow)
        base_datasets = []
        for file_path_i, file_path in descriptions.loc['path'].iteritems():
            # parse age and gender information from EDF header
            age, gender = _parse_age_and_gender_from_edf_header(file_path)
            raw = mne.io.read_raw_edf(file_path, preload=preload)
            # read info relevant for preprocessing from raw without loading it
            sfreq = raw.info['sfreq']
            n_samples = raw.n_times
            d = {
                'sfreq': float(sfreq),
                'n_samples': int(n_samples),
                'age': int(age),
                'gender': gender,
            }
            if add_physician_reports:
                physician_report = _read_physician_report(file_path)
                d['report'] = physician_report
            additional_description = pd.Series(d)
            description = pd.concat(
                [descriptions.pop(file_path_i), additional_description])
            base_dataset = BaseDataset(raw, description,
                                       target_name=target_name)
            base_datasets.append(base_dataset)
        return base_datasets


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
    reference = tokens[-5].split('_')[2]
    return {
        'path': file_path,
        'version': version,
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'subject': int(subject_id),
        'session': int(session[1:]),
        'segment': int(segment[1:]),
        'reference': reference,
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
                 preload=False, add_physician_reports=False):
        super().__init__(path=path, recording_ids=recording_ids,
                         preload=preload,
                         add_physician_reports=add_physician_reports)
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = (
                self._parse_additional_description_from_file_path(file_path))
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)
        # set target name of base datasets
        for ds_i, ds in enumerate(self.datasets):
            ds.target_name = target_name

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

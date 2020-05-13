import os
import re
import glob

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset


class TUHAbnormal(BaseConcatDataset):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.

    Parameters
    ----------
    path: str
        parent directory of the dataset
    subject_ids: list(int) | int
        (list of) int of subject(s) to be read
    target_name: str
        can be 'pathological', 'gender', or 'age'
    preload: bool
        if True, preload the data of the Raw objects.
    """
    def __init__(self, path, subject_ids=None, target_name="pathological",
                 preload=False):
        all_file_paths = read_all_file_names(
            path, extension='.edf', key=self._time_key)
        if subject_ids is None:
            subject_ids = np.arange(len(all_file_paths))

        all_base_ds = []
        for subject_id in subject_ids:
            file_path = all_file_paths[subject_id]
            raw = mne.io.read_raw_edf(file_path, preload=preload)
            path_splits = file_path.split("/")
            if "abnormal" in path_splits:
                pathological = True
            else:
                assert "normal" in path_splits
                pathological = False
            if "train" in path_splits:
                session = "train"
            else:
                assert "eval" in path_splits
                session = "eval"
            age, gender = _parse_age_and_gender_from_edf_header(file_path)
            description = pd.Series(
                {'age': age, 'pathological': pathological, 'gender': gender,
                'session': session, 'subject': subject_id}, name=subject_id)
            base_ds = BaseDataset(raw, description, target_name=target_name)
            all_base_ds.append(base_ds)

        super().__init__(all_base_ds)

    @staticmethod
    def _time_key(file_path):
        # the splits are specific to tuh abnormal eeg data set
        splits = file_path.split('/')
        p = r'(\d{4}_\d{2}_\d{2})'
        [date] = re.findall(p, splits[-2])
        date_id = [int(token) for token in date.split('_')]
        recording_id = _natural_key(splits[-1])
        session_id = re.findall(r'(s\d*)_', (splits[-2]))
        return date_id + session_id + recording_id


# TODO: this is very slow. how to improve?
def read_all_file_names(directory, extension, key):
    """Read all files with specified extension from given path and sorts them
    based on a given sorting key.

    Parameters
    ----------
    directory: str
        file path on HDD
    extension: str
        file path extension, i.e. '.edf' or '.txt'
    key: calable
        sorting key for the file paths

    Returns
    -------
    file_paths: list(str)
        a list to all files found in (sub)directories of path
    """
    assert extension.startswith(".")
    file_paths = glob.glob(directory + '**/*' + extension, recursive=True)
    file_paths = sorted(file_paths, key=key)
    assert len(file_paths) > 0, (
        f"something went wrong. Found no {extension} files in {directory}")
    return file_paths


def _natural_key(string):
    pattern = r'(\d+)'
    key = [int(split) if split.isdigit() else None
           for split in re.split(pattern, string)]
    return key


def _parse_age_and_gender_from_edf_header(file_path, return_raw_header=False):
    assert os.path.exists(file_path), f"file not found {file_path}"
    f = open(file_path, 'rb')
    content = f.read(88)
    f.close()
    if return_raw_header:
        return content
    patient_id = content[8:88].decode('ascii')
    [age] = re.findall(r"Age:(\d+)", patient_id)
    [gender] = re.findall(r"\s(\w)\s", patient_id)
    return int(age), gender

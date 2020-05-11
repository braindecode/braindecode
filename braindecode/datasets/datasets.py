"""Dataset objects for some public datasets.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import os
import re
import glob

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset, WindowsDataset
from ..datautil.windowers import (
    create_fixed_length_windows, create_windows_from_events,
    _compute_supercrop_inds)


def _find_dataset_in_moabb(dataset_name):
    # soft dependency on moabb
    from moabb.datasets.utils import dataset_list
    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            # return an instance of the found dataset class
            return dataset()
    raise ValueError("'dataset_name' not found in moabb datasets")


def _fetch_and_unpack_moabb_data(dataset, subject_ids):
    data = dataset.get_data(subject_ids)
    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id, subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                # set annotation if empty
                if len(raw.annotations) == 0:
                    annots = _annotations_from_moabb_stim_channel(raw, dataset)
                    raw.set_annotations(annots)
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame({
        'subject': subject_ids,
        'session': session_ids,
        'run': run_ids
    })
    return raws, description


def _annotations_from_moabb_stim_channel(raw, dataset):
    # find events from stim channel
    events = mne.find_events(raw)

    # get annotations from events
    event_desc = {k: v for v, k in dataset.event_id.items()}
    annots = mne.annotations_from_events(events, raw.info['sfreq'], event_desc)

    # set trial on and offset given by moabb
    onset, offset = dataset.interval
    annots.onset += onset
    annots.duration += offset - onset
    return annots


def fetch_data_with_moabb(dataset_name, subject_ids):
    # ToDo: update path to where moabb downloads / looks for the data
    """Fetch data using moabb.

    Parameters
    ----------
    dataset_name: str
        the name of a dataset included in moabb
    subject_ids: list(int) | int
        (list of) int of subject(s) to be fetched

    Returns
    -------
    raws: mne.Raw
    info: pandas.DataFrame
    """
    dataset = _find_dataset_in_moabb(dataset_name)
    subject_id = [subject_ids] if isinstance(subject_ids, int) else subject_ids
    return _fetch_and_unpack_moabb_data(dataset, subject_id)


class MOABBDataset(BaseConcatDataset):
    """A class for moabb datasets.

    Parameters
    ----------
    dataset_name: name of dataset included in moabb to be fetched
    subject_ids: list(int) | int
        (list of) int of subject(s) to be fetched
    """
    def __init__(self, dataset_name, subject_ids):
        raws, description = fetch_data_with_moabb(dataset_name, subject_ids)
        all_base_ds = [BaseDataset(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        super().__init__(all_base_ds)


class BNCI2014001(MOABBDataset):
    """See moabb.datasets.bnci.BNCI2014001"""
    def __init__(self, *args, **kwargs):
        super().__init__("BNCI2014001", *args, **kwargs)


class HGD(MOABBDataset):
    """See moabb.datasets.schirrmeister2017.Schirrmeister2017"""
    def __init__(self, *args, **kwargs):
        super().__init__("Schirrmeister2017", *args, **kwargs)


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
    [age] = re.findall("Age:(\d+)", patient_id)
    [gender] = re.findall("\s(\w)\s", patient_id)
    return int(age), gender

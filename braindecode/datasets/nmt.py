"""
Dataset classes for the NMT EEG Corpus 
"""

# Authors: Mohammad Bayazi <mj.darvishi92@gmail.com>
#
# License: BSD (3-clause)

import os
import warnings
import pandas as pd
from .tuh import TUH, _read_date


def _create_description(file_paths):
    descriptions = [_parse_description_from_file_path(f) for f in file_paths]
    descriptions = pd.DataFrame(descriptions)
    return descriptions.T

def _parse_description_from_file_path(file_path):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # Extract info
    if ('train' in tokens) or ('eval' in tokens):  # _eeg_abnormal
        abnormal = True

    else:  # tuh_eeg
        abnormal = False

    subject_id = tokens[-1].split('.')[0]
    description = _read_date(file_path)
    description.update({
        'path': file_path,
        'subject': subject_id,
        # 'session': int(session[1:]),
        # 'segment': int(segment[1:]),
    })
    if not abnormal:
        year, month, day = tokens[-3].split('_')[1:]
        description['year'] = int(year)
        description['month'] = int(month)
        description['day'] = int(day)
    return description

class NMT(TUH):
    """ NMT EEG Corpus.
    see https://dll.seecs.nust.edu.pk/downloads/

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
        descriptions = _create_description(file_paths)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in description. '__getitem__'")
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
        # expect paths as /pathology status//data_split/file.edf
        # e.g.            /normal/train/00001.edf
        assert ('abnormal' in tokens or 'normal' in tokens), (
            'No pathology labels found.')
        assert ('train' in tokens or 'eval' in tokens), (
            'No train or eval set information found.')
        return {
            'train': 'train' in tokens,
            'pathological': 'abnormal' in tokens,
        }
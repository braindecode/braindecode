"""Dataset objects for some public datasets.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset, WindowsDataset


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

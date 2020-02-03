"""
BCI competition IV 2a dataset
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

from .base import BaseDataset, BaseConcatDataset

try:
    from mne import annotations_from_events
except ImportError:
    # XXX: Remove try/except once the following function is in an MNE release
    #      (probably 19.3).
    from mne import Annotations
    from mne.utils import _validate_type
    import collections

    def _check_event_description(event_desc, events):
        """Check event_id and convert to default format."""
        if event_desc is None:  # convert to int to make typing-checks happy
            event_desc = list(np.unique(events[:, 2]))

        if isinstance(event_desc, dict):
            for val in event_desc.values():
                _validate_type(val, (str, None), "Event names")
        elif isinstance(event_desc, collections.Iterable):
            event_desc = np.asarray(event_desc)
            if event_desc.ndim != 1:
                raise ValueError(
                    "event_desc must be 1D, got shape {}".format(
                        event_desc.shape
                    )
                )
            event_desc = dict(zip(event_desc, map(str, event_desc)))
        elif callable(event_desc):
            pass
        else:
            raise ValueError(
                "Invalid type for event_desc (should be None, list, "
                "1darray, dict or callable). Got {}".format(type(event_desc))
            )

        return event_desc


    def _select_events_based_on_id(events, event_desc):
        """Get a collection of events and returns index of selected."""
        event_desc_ = dict()
        func = event_desc.get if isinstance(event_desc, dict) else event_desc
        event_ids = events[np.unique(events[:, 2], return_index=True)[1], 2]
        for e in event_ids:
            trigger = func(e)
            if trigger is not None:
                event_desc_[e] = trigger

        event_sel = [ii for ii, e in enumerate(events) if e[2] in event_desc_]

        # if len(event_sel) == 0:
        #     raise ValueError('Could not find any of the events you specified.')

        return event_sel, event_desc_

    def annotations_from_events(
        events,
        sfreq,
        event_desc=None,
        first_samp=0,
        orig_time=None,
        verbose=None,
    ):
        """Convert an event array to an Annotations object.
        Parameters
        ----------
        events : ndarray, shape (n_events, 3)
            The events.
        sfreq : float
            Sampling frequency.
        event_desc : dict | array-like | callable | None
            Events description. Can be:
            - **dict**: map integer event codes (keys) to descriptions (values).
            Only the descriptions present will be mapped, others will be ignored.
            - **array-like**: list, or 1d array of integers event codes to include.
            Only the event codes present will be mapped, others will be ignored.
            Event codes will be passed as string descriptions.
            - **callable**: must take a integer event code as input and return a
            string description or None to ignore it.
            - **None**: Use integer event codes as descriptions.
        first_samp : int
            The first data sample (default=0). See :attr:`mne.io.Raw.first_samp`
            docstring.
        orig_time : float | str | datetime | tuple of int | None
            Determines the starting time of annotation acquisition. If None
            (default), starting time is determined from beginning of raw data
            acquisition. For details, see :meth:`mne.Annotations` docstring.
        %(verbose)s
        Returns
        -------
        annot : instance of Annotations
            The annotations.
        Notes
        -----
        Annotations returned by this function will all have zero (null) duration.
        """
        event_desc = _check_event_description(event_desc, events)
        event_sel, event_desc_ = _select_events_based_on_id(events, event_desc)
        events_sel = events[event_sel]
        onsets = (events_sel[:, 0] - first_samp) / sfreq
        descriptions = [event_desc_[e[2]] for e in events_sel]
        durations = np.zeros(len(events_sel))  # dummy durations

        # Create annotations
        annots = Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=orig_time,
        )

        return annots


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
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame(zip(subject_ids, session_ids, run_ids),
                        columns=["subject", "session", "run"])
    return raws, description


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
    def __init__(
            self, dataset_name, subject_ids):
        raws, description = fetch_data_with_moabb(dataset_name, subject_ids)
        all_base_ds = []
        for i_raw, raw in enumerate(raws):
            base_ds = BaseDataset(raw, description.iloc[[i_raw]])
            all_base_ds.append(base_ds)
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
    """
    def __init__(self, path, subject_ids=None, target_name="pathological"):
        all_file_paths = read_all_file_names(
            path, extension='.edf', key=self._time_key)
        if subject_ids is None:
            subject_ids = np.arange(len(all_file_paths))

        all_base_ds = []
        for subject_id in subject_ids:
            file_path = all_file_paths[subject_id]
            raw = mne.io.read_raw_edf(file_path)
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
            description = pd.DataFrame(
                [[age, pathological, gender, session, subject_id]],
                columns=["age", "pathological", "gender",
                "session", "subject"], index=[subject_id])
            base_ds = BaseDataset(raw, description, target_name=target_name)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    def _time_key(self, file_path):
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
        f"something went wrong. Found no {extension} files in {path}")
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

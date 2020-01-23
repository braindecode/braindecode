"""
BCI competition IV 2a dataset
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import mne

from torch.utils.data import ConcatDataset, Subset
from .dataset import WindowsDataset, BaseDataset
from ..datautil.windowers import EventWindower, FixedLengthWindower

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


def _find_dataset(dataset_name):
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
        info = pd.DataFrame(zip(subject_ids, session_ids, run_ids),
                            columns=["subject", "session", "run"])
        return raws, info


def fetch_data_with_moabb(dataset_name, subject_ids, path=None):
    """
    Class that fetches data using moabb.
    """
    dataset = _find_dataset(dataset_name)
    subject_id = [subject_ids] if isinstance(subject_ids, int) else subject_ids
    if path is not None:
        # ToDo: mne update (path)
        pass
    return _fetch_and_unpack_moabb_data(dataset, subject_id)


class MOABBDataset(ConcatDataset):
    # TODO: include preprocessing at different stages
    def __init__(
            self, dataset_name, subject_ids, trial_start_offset_samples,
            trial_stop_offset_samples, supercrop_size_samples,
            supercrop_stride_samples, drop_samples=False, ignore_events=False):
        if ignore_events:
            windower = FixedLengthWindower
        else:
            windower = EventWindower
        windower = windower(
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            supercrop_size_samples=supercrop_size_samples,
            supercrop_stride_samples=supercrop_stride_samples,
            drop_samples=drop_samples)

        raw_data, info = fetch_data_with_moabb(dataset_name, subject_ids)
        all_windows_ds = []
        for data_i, data in enumerate(raw_data):
            base_ds = BaseDataset(data, info.iloc[data_i])
            windows_ds = WindowsDataset(base_ds, windower)
            all_windows_ds.append(windows_ds)
        super().__init__(all_windows_ds)
        self.info = info

    def split(self, some_property):
        split_ids = _split_ids(self.info, some_property)
        return [Subset(self, split) for split in split_ids]

def _split_ids(df, some_property):
    """
    Split dataset into subsets based on a column in info DataFrame
    """
    assert some_property in df
    split_ids = []
    for group_name, group in df.groupby(some_property):
        split_ids.append(list(group.index))
    return split_ids


class BNCI2014001(MOABBDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("BNCI2014001", *args, **kwargs)


class HGD(MOABBDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("Schirrmeister2017", *args, **kwargs)


# TODO: read all edfs (sorted by time)
all_file_paths = [
    "/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/s004_2013_08_15/00000021_s004_t000.edf",
    "/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000016/s004_2012_02_08/00000016_s004_t000.edf"]
class TUHAbnormal(ConcatDataset):
    def __init__(self, trial_start_offset_samples,
                 trial_stop_offset_samples, supercrop_size_samples,
                 supercrop_stride_samples, subject_ids=None,
                 drop_samples=False, target="pathological", mapping=None):
        windower = FixedLengthWindower(
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            supercrop_size_samples=supercrop_size_samples,
            supercrop_stride_samples=supercrop_stride_samples,
            drop_samples=drop_samples, mapping=mapping)

        if subject_ids is None:
            subject_ids = np.arange(len(all_file_paths))

        all_windows_ds, all_infos = [], []
        for subject_id in subject_ids:
            raw = mne.io.read_raw_edf(all_file_paths[subject_id])
            # TODO: parse age and gender from edf file header and add to info
            path_splits = all_file_paths_sorted_by_time[subject_id].split("/")
            if "abnormal" in path_splits:
                pathological = True
            else:
                assert "normal" in path_splits
                pathological = False
            age, gender, session = 48, "M", "train"
            info = pd.DataFrame(
                [[age, pathological, gender, session, subject_id]],
                columns=["age", "pathological", "gender",
                "session", "subject"], index=[subject_id])
            info = info.rename(columns={target: "target"})
            base_ds = BaseDataset(raw, info)
            windows_ds = WindowsDataset(base_ds, windower)
            all_windows_ds.append(windows_ds)
            all_infos.append(info)

        super().__init__(all_windows_ds)
        self.info = pd.concat(all_infos)

    def split(self, some_property):
        split_ids = _split_ids(self.info, some_property)
        return [Subset(self, split) for split in split_ids]

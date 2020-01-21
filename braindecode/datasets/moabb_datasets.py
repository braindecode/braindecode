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
import mne

from torch.utils.data import ConcatDataset
from .dataset import WindowsDataset

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


def find_data_set(dataset_name):
    # soft dependency on moabb
    from moabb.datasets.utils import dataset_list
    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            # return an instance of the found dataset class
            return dataset()
    raise ValueError("'dataset_name' not found in moabb datasets")


class MOABBFetcher(list):
    """
    Class that fetches data using moabb.
    TODO: Should replace parts of MOABBDataset
    """
    def __init__(self, dataset_name, subject, path=None):
        self.dataset = find_data_set(dataset_name)
        self.subject = [subject] if isinstance(subject, int) else subject
        if path is not None:
            # ToDo: mne update (path)
            pass
        self._fetch_data_with_moabb()

    def _fetch_data_with_moabb(self):
        data = self.dataset.get_data(self.subject)
        for subj_id, subj_data in data.items():
            for sess_id, sess_data in subj_data.items():
                for run_id, raw in sess_data.items():
                    # 0 - Get events and remove stim channel
                    raw = self._populate_raw_from_moabb(
                        raw, subj_id, sess_id, run_id)

                    if len(raw.annotations.onset) == 0:
                        continue

                    self.append(raw)


    def _populate_raw_from_moabb(self, raw, subj_id, sess_id, run_id):
        """Populate raw with subject, events, session and run information

        Parameters
        ----------
        raw : mne.io.Raw
            raw data to populate
        sess_id : int
            session id
        run_id : int
            run id

        Returns
        -------
        mne.io.Raw
            populated raw
        """
        fs = raw.info['sfreq']

        raw.info['subject_info'] = {
            'id': subj_id,
            'his_id': None,
            'last_name': None,
            'first_name': None,
            'middle_name': None,
            'birthday': None,
            'sex': None,
            'hand': None
        }
        raw.info['session'] = sess_id
        raw.info['run'] = run_id
        if len(raw.annotations) == 0:
            events = mne.find_events(raw)
            event_onset, event_offset = self.dataset.interval  # in seconds
            events[:, 0] += int(event_onset * fs)

            raw.info['events'] = events
            mapping = {v: k for k, v in self.dataset.event_id.items()}

            annots = annotations_from_events(
                raw.info['events'], raw.info['sfreq'], event_desc=mapping,
                first_samp=raw.first_samp, orig_time=None)

            annots.duration += event_offset - event_onset

            raw.set_annotations(annots)
        return raw



class MOABBDataset(ConcatDataset):
    """ Class that fetches data using moabb, applies given transformers on
    raw data, creates mne epochs from raw and applies given transformers on
    windows

    Parameters
    ----------
    dataset_name : str
        name of the dataset according to moabb notation
    subject : int | list of int
        subject id[s]
    raw_transformer : sklearn.base.TansformerMixim
        raw transformers applied before windowing
    windower : sklearn.base.TansformerMixim
        windower transformer
    transformer : sklearn.base.TansformerMixim
        transformer applied after windowing
    transform_online : bool
        if True, apply window transformers on the fly. Otherwise apply on loaded
        data.
    """

    def __init__(
        self,
        dataset_name,
        subject,
        raw_transformer=None,
        windower=None,
        transformer=None,
        transform_online=False,
        path=None,
    ):
        self.dataset_name = dataset_name
        self.moabb_dataset = self.find_data_set(self.dataset_name)
        self.subject = [subject] if isinstance(subject, int) else subject
        self.raw_transformer = (
            raw_transformer
            if isinstance(raw_transformer, list) or raw_transformer is None
            else [raw_transformer])
        self.windower = windower
        self.transformer = (
            transformer
            if isinstance(transformer, list) or transformer is None
            else [transformer])
        self.transform_online = transform_online

        if path is not None:
            # ToDo: mne update (path)
            pass

        base_datasets = self._base_datasets_from_moabb()

        # Concatenate datasets
        super().__init__(base_datasets)

    def get_mapping(self):
        """Get mapping of final events

        Returns
        -------
        dict
            mapping for events {key: value}

        """
        return self.mapping(self.dataset_name)

    def _base_datasets_from_moabb(self):
        data = self.moabb_dataset.get_data(self.subject)

        base_datasets = list()
        trial_durations = list()
        windows_fs = []
        for subj_id, subj_data in data.items():
            for sess_id, sess_data in subj_data.items():
                for run_id, raw in sess_data.items():
                    # 0 - Get events and remove stim channel
                    raw = self._populate_raw_from_moabb(
                        raw, subj_id, sess_id, run_id)
                    if len(raw.annotations.onset) == 0:
                        continue
                    tmin = self.windower.trial_start_offset_samples / raw.info['sfreq']
                    trial_durations.append(
                        raw.annotations.duration - tmin
                    )
                    picks = mne.pick_types(raw.info, meg=False, eeg=True)
                    raw = raw.pick_channels(np.array(raw.ch_names)[picks])

                    # 1- Apply preprocessing
                    if self.raw_transformer is not None:
                        for transformer in self.raw_transformer:
                            raw = transformer(raw)

                    # 2- Epoch
                    if self.windower.mapping is None:
                        self.windower.mapping = self.mapping(self.dataset_name)
                    windows = self.windower(raw,)
                    if self.transform_online:
                        transformer = self.transformer
                    else:
                        # XXX: Apply transformer
                        window_transformer = None
                        raise NotImplementedError

                    windows_fs.append(windows.info["sfreq"])
                    windows.event_id = self.get_mapping()
                    # 3- Create BaseDataset
                    base_datasets.append(WindowsDataset(
                        windows, transforms=transformer))

        assert len(set(windows_fs)) == 1, (
            "inconsistent sampling frequencies detected"
        )
        self.trial_durations_seconds = np.concatenate(trial_durations, axis=0)
        # in the case of resampling on the fly, this might give incorrect
        # sampling frequency
        self.fs = windows_fs[-1]
        return base_datasets

    def get_trial_durations_samples(self):
        """Returns the trial duration of experiment in samples

        Returns
        -------
        ndarray
            trial durations in samples
        """
        trial_durations_samples = self.trial_durations_seconds * self.fs
        return trial_durations_samples.astype(int)

    def _populate_raw_from_moabb(self, raw, subj_id, sess_id, run_id):
        """Populate raw with subject, events, session and run information

        Parameters
        ----------
        raw : mne.io.Raw
            raw data to populate
        sess_id : int
            session id
        run_id : int
            run id

        Returns
        -------
        mne.io.Raw
            populated raw
        """
        fs = raw.info["sfreq"]

        raw.info["subject_info"] = {
            "id": subj_id,
            "his_id": None,
            "last_name": None,
            "first_name": None,
            "middle_name": None,
            "birthday": None,
            "sex": None,
            "hand": None,
        }
        raw.info["session"] = sess_id
        raw.info["run"] = run_id
        mapping = self.moabb_dataset.event_id
        mapping = {k: v for v, k in mapping.items()}
        if len(raw.annotations) == 0:
            events = mne.find_events(raw)
            event_onset, event_offset = self.moabb_dataset.interval  # seconds
            events[:, 0] += int(event_onset * fs)
            raw.info["events"] = events

            annots = annotations_from_events(
                raw.info["events"],
                raw.info["sfreq"],
                event_desc=mapping,
                first_samp=raw.first_samp,
                orig_time=None,
            )

            annots.duration += event_offset - event_onset

            raw.set_annotations(annots)
        return raw

    @staticmethod
    def mapping(dataset_name, return_value_mapping=False):
        """

        Parameters
        ----------
        dataset_name str
            Dataset name as used by MOABB. See MOABB for further details
        return_value_mapping
            bool
            if True, the mapping of original event id's to new values (starting
            from 0) are returned

        Returns
        -------
        dict
            sorted mapping having event id values from 0 on
        """
        dataset = MOABBDataset.find_data_set(dataset_name)
        mapping = dataset.event_id
        # sort mapping
        mapping = {k: v for k, v in sorted(mapping.items(),
                                           key=lambda item: item[1])}
        if return_value_mapping:
            # reuturn mapping from old mapping values to new (starting from 0)
            return {old: new for new, old in enumerate(mapping.values())}
        else:
            # return new {key: value} mapping starting from 0 on
            return {k: v for v, k in enumerate(mapping.keys())}

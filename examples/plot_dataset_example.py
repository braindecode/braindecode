"""
=================
Creating Datasets
=================

"""
# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict

import numpy as np
import mne
from torch.utils.data import Dataset, ConcatDataset
from moabb.datasets import BNCI2014001
from sklearn.pipeline import Pipeline

# preprocessing
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.preprocessors import FilteringTransformer, ZscoreTransformer
from braindecode.windowers import FixedLengthWindower, EventWindower


try:
    from mne import annotations_from_events
except ImportError:
    # XXX: Remove try/except once the following function is in an MNE release
    #      (probably 19.3).
    from mne import Annotations
    from mne.utils import _validate_type

    def _check_event_description(event_desc, events):
        """Check event_id and convert to default format."""
        if event_desc is None:  # convert to int to make typing-checks happy
            event_desc = list(np.unique(events[:, 2]))

        if isinstance(event_desc, dict):
            for val in event_desc.values():
                _validate_type(val, (str, None), 'Event names')
        elif isinstance(event_desc, collections.Iterable):
            event_desc = np.asarray(event_desc)
            if event_desc.ndim != 1:
                raise ValueError('event_desc must be 1D, got shape {}'.format(
                                event_desc.shape))
            event_desc = dict(zip(event_desc, map(str, event_desc)))
        elif callable(event_desc):
            pass
        else:
            raise ValueError('Invalid type for event_desc (should be None, list, '
                            '1darray, dict or callable). Got {}'.format(
                                type(event_desc)))

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


    def annotations_from_events(events, sfreq, event_desc=None, first_samp=0,
                                orig_time=None, verbose=None):
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
        annots = Annotations(onset=onsets,
                            duration=durations,
                            description=descriptions,
                            orig_time=orig_time)

        return annots


###############################################################################
# Fetching and downloading data with MOABB
class BNCI2014001Dataset(ConcatDataset):
    """[summary]
        
    Parameters
    ----------
    subject : [type]
        [description]
    path : [type], optional
        [description], by default None
    force_update : bool, optional
        [description], by default False
    preprocessor : [type], optional
        [description], by default None
    """      
    def __init__(self, subject, path=None, force_update=False, 
                 preprocessor=None, windower=None, transformer=None,
                 transform_online=False):
        """
        0- Check whether files exist given path 
        1- Download files if they don't exist
        """

        self.preprocessor = preprocessor
        self.windower = windower
        self.transformer = transformer
        self.transform_online = transform_online

        # Preprocessing parameters
        input_time_length = 1000
        low_cut_hz = 4
        high_cut_hz = 38
        factor_new = 1e-3
        init_block_size = 1000

        # Epoching
        ival = [-500, 4000]
        
        # Sampling [SHOULD GO OUTSIDE]
        batch_size = 60
        max_epochs = 800
        max_increase_epochs = 80
        valid_set_fraction = 0.2

        self.subject = [subject] if isinstance(subject, int) else subject
        data = BNCI2014001().get_data(self.subject)

        mapping = {
            1: 'Left hand',
            2: 'Right hand',
            3: 'Foot',
            4: 'Tongue'
        }

        base_datasets = list()
        for subj_id, subj_data in data.items():
            for sess_id, sess_data in subj_data.items():
                for run_id, raw in sess_data.items():

                    # 0 - Get events and remove stim channel
                    raw = self._populate_raw(
                        raw, subj_id, sess_id, run_id, mapping)
                    if len(raw.annotations.onset) == 0:
                        continue
                    picks = mne.pick_types(raw.info, meg=False, eeg=True)
                    raw = raw.pick_channels(np.array(raw.ch_names)[picks])

                    # 1- Apply preprocessing
                    raw = self.preprocessor.fit_transform(raw)

                    # 2- Epoch
                    windows = self.windower.fit_transform(raw)
                    if self.transform_online:
                        transformer = self.transformer
                    else:
                        # XXX: Apply transformer
                        transformer = None

                    # 3- Create BaseDataset
                    base_datasets.append(WindowsDataset(
                        windows, transformer=transformer))

        # Concatenate datasets
        super().__init__(base_datasets)

        

    def _populate_raw(self, raw, subj_id, sess_id, run_id, mapping):
        """TO CLEAN UP/REMOVE?
        
        Parameters
        ----------
        raw : [type]
            [description]
        sess_id : [type]
            [description]
        run_id : [type]
            [description]
        mapping : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
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
        events = mne.find_events(raw, stim_channel='stim')

        events[:, 0] += int(2 * fs)  # start of motor movement 2s after trial onset

        raw.info['events'] = events
        annots = annotations_from_events(
            raw.info['events'], raw.info['sfreq'], event_desc=mapping, 
            first_samp=raw.first_samp, orig_time=None)

        annots.duration += 4.  # duration of motor task 4s

        raw.set_annotations(annots)
        return raw


class BaseDataset(Dataset):
    """[summary]
    
    Parameters
    ----------
    Dataset : [type]
        [description]
    """
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class WindowsDataset(Dataset):
    """[summary]
    
    Parameters
    ----------
    Dataset : [type]
        [description]
    """
    def __init__(self, windows, target='target', transformer=None):
        self.windows = windows
        self.target = target
        if self.target != 'target':
            assert self.target in self.windows.info['subject_info'].keys()
        # XXX Handle multitarget case
            
    def __getitem__(self, index):
        """[summary]
        
        Parameters
        ----------
        index : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        X = np.squeeze(self.windows[index].get_data())
        if self.target == 'target':
            y = self.windows.metadata.iloc[index]['target']
        else:
            y = self.windows.info['subject_info'][self.target]
            
        return X, y

    def __len__(self):
        return self.windows.metadata.shape[0]
        # XXX: The following would fail if data has not been preloaded yet:
        # return len(self.windows)



# ###############################################################################
# # Let's create a dataset!

filter_ = FilteringTransformer(l_freq=4, h_freq=12)
zscorer = ZscoreTransformer()
prepr_pipeline = Pipeline(
    [('bandpass_filter', filter_), ('zscorer', zscorer)])
#windower = Windower(window_size_samples=None,
#                    overlap_size_samples=0)
mapping = {
    1: 'Left hand',
    2: 'Right hand',
    3: 'Foot',
    4: 'Tongue'
}
event_windower = EventWindower(
    window_size_samples=200, tmin=0, chunk_duration_samples=200, mapping=mapping)
# fixed_length_windower = FixedLengthWindower(
#     window_size_samples=200, tmin=0, chunk_duration_samples=200, mapping=mapping)

bnci = BNCI2014001Dataset([2], preprocessor=prepr_pipeline, windower=event_windower)



# ###############################################################################
# # Since downsampling reduces the timing precision of events, we recommend
# # first extracting epochs and downsampling the Epochs object:
# events = mne.find_events(raw)
# epochs = mne.Epochs(raw, events, event_id=2, tmin=-0.1, tmax=0.8, preload=True)

# # Downsample to 100 Hz
# print('Original sampling rate:', epochs.info['sfreq'], 'Hz')
# epochs_resampled = epochs.copy().resample(100, npad='auto')
# print('New sampling rate:', epochs_resampled.info['sfreq'], 'Hz')

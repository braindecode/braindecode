# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


import os

import numpy as np
import pandas as pd
import mne
from mne.datasets.sleep_physionet.age import fetch_data

from .base import BaseDataset, BaseConcatDataset


class SleepPhysionet(BaseConcatDataset):
    """Sleep Physionet dataset.
<<<<<<< HEAD
    Sleep dataset from https://physionet.org/content/sleep-edfx/1.0.0/.
    Contains overnight recordings from 78 healthy subjects.
    See [MNE example](https://mne.tools/stable/auto_tutorials/sample-datasets/plot_sleep.html).
=======

    Sleep dataset from https://physionet.org/content/sleep-edfx/1.0.0/.
    Contains overnight recordings from 78 healthy subjects.

    See [MNE example](https://mne.tools/stable/auto_tutorials/sample-datasets/plot_sleep.html).

>>>>>>> braindecode_hubert/sleep-example
    Parameters
    ----------
    subject_ids: list(int) | int
        (list of) int of subject(s) to be loaded.
    recording_ids: list(int)
        Recordings to load per subject (each subject except 13 has two
        recordings). Can be [1], [2] or [1, 2].
    preload: bool
        if True, preload the data of the Raw objects.
    load_eeg_only: bool
        If True, only load the EEG channels and discard the others (EOG, EMG,
        temperature, respiration) to avoid resampling the other signals.
    crop_wake_mins: float
        Number of minutes of wake time to keep before the first sleep event
        and after the last sleep event. Used to reduce the imbalance in this
        dataset. Default of 30 mins.
    """
<<<<<<< HEAD

=======
>>>>>>> braindecode_hubert/sleep-example
    def __init__(self, subject_ids=None, recording_ids=None, preload=False,
                 load_eeg_only=True, crop_wake_mins=30):
        if subject_ids is None:
            subject_ids = range(83)
        if recording_ids is None:
            recording_ids = [1, 2]

<<<<<<< HEAD
        paths = fetch_data(subject_ids, recording=recording_ids, on_missing="warn")
=======
        paths = fetch_data(subject_ids, recording=recording_ids)
>>>>>>> braindecode_hubert/sleep-example

        all_base_ds = list()
        for p in paths:
            raw, desc = self._load_raw(
                p[0], p[1], preload=preload, load_eeg_only=load_eeg_only,
                crop_wake_mins=crop_wake_mins)
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _load_raw(raw_fname, ann_fname, preload, load_eeg_only=True,
                  crop_wake_mins=False):
        ch_mapping = {
            'EOG horizontal': 'eog',
            'Resp oro-nasal': 'misc',
            'EMG submental': 'misc',
            'Temp rectal': 'misc',
            'Event marker': 'misc'
        }
        exclude = ch_mapping.keys() if load_eeg_only else ()

        raw = mne.io.read_raw_edf(raw_fname, preload=preload, exclude=exclude)
        annots = mne.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [
                x[-1] in ['1', '2', '3', '4', 'R'] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]['onset'] - \
<<<<<<< HEAD
                crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
                crop_wake_mins * 60
=======
                   crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
                   crop_wake_mins * 60
>>>>>>> braindecode_hubert/sleep-example
            raw.crop(tmin=tmin, tmax=tmax)

        # Rename EEG channels
        ch_names = {
            i: i.replace('EEG ', '') for i in raw.ch_names if 'EEG' in i}
        mne.rename_channels(raw.info, ch_names)

        if not load_eeg_only:
            raw.set_channel_types(ch_mapping)

        basename = os.path.basename(raw_fname)
        subj_nb = int(basename[3:5])
        sess_nb = int(basename[5])
        desc = pd.Series({'subject': subj_nb, 'recording': sess_nb}, name='')

        return raw, desc

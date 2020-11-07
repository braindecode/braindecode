import os
from joblib import Memory

import numpy as np
import pandas as pd
import mne

from mne.datasets.sleep_physionet.age import fetch_data
from braindecode.datautil import create_windows_from_events
from braindecode.datautil.preprocess import (
    MNEPreproc, NumpyPreproc, preprocess)
from .base import BaseDataset, BaseConcatDataset


cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


class SleepPhysionet(BaseConcatDataset):
    """Sleep Physionet dataset.

    Sleep dataset from https://physionet.org/content/sleep-edfx/1.0.0/.
    Contains overnight recordings from 78 healthy subjects.

    See [MNE example](https://mne.tools/stable/auto_tutorials/sample-datasets/plot_sleep.html).

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

    def __init__(self, subject_ids=None, recording_ids=None, preload=False,
                 load_eeg_only=True, crop_wake_mins=30):
        if subject_ids is None:
            subject_ids = range(83)
        if recording_ids is None:
            recording_ids = [1, 2]

        paths = fetch_data(
            subject_ids,
            recording=recording_ids,
            on_missing="ignore")

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
                crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
                crop_wake_mins * 60
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


@memory.cache
def get_dummy_sample(preprocessing=["microvolt_scaling", "filtering"]):
    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=[0],
        valid_subjects=[1],
        test_subjects=[2],
        recording=[1],
        crop_wake_mins=0,
        preprocessing=preprocessing)
    # for i in range(len(train_sample)):
    #     train_sample[i] = (train_sample[i][0][:50], train_sample[i][1],
    #                        train_sample[i][2])
    # for i in range(len(test_sample)):
    #     test_sample[i] = (test_sample[i][0][:50],
    #                       test_sample[i][1], test_sample[i][2])
    # print("lol")
    test_choice = np.random.choice(
        range(len(test_sample)),
        size=2,
        replace=False)
    valid_choice = np.random.choice(
        range(len(test_sample)),
        size=2,
        replace=False)
    train_tinying_dict = {0: [350, 1029, 1291, 1650, 1571]}
    test_tinying_dict = {0: valid_choice}
    valid_tinying_dict = {0: test_choice}
    train_sample = tinying_dataset(train_sample, train_tinying_dict)
    test_sample = tinying_dataset(test_sample, test_tinying_dict)
    valid_sample = tinying_dataset(valid_sample, valid_tinying_dict)

    return train_sample, valid_sample, test_sample


def get_epochs_data(num_train=None, num_test=None, num_valid=None,
                    train_subjects=tuple(range(0, 50)),
                    valid_subjects=tuple(range(50, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    preprocessing=["microvolt_scaling", "filtering"],
                    crop_wake_mins=30,
                    random_seed=None):

    if num_train is not None:
        np.random.seed(random_seed)
        rand_sub = np.random.choice(
            tuple(range(83)), size=num_train + num_test + num_valid,
            replace=True, p=None)
        train_subjects = rand_sub[:num_train]
        test_subjects = rand_sub[num_train:num_train + num_test]
        valid_subjects = rand_sub[num_train + num_test:]

    train_sample = build_epoch(
        train_subjects, recording, crop_wake_mins, preprocessing)
    valid_sample = build_epoch(
        valid_subjects, recording, crop_wake_mins, preprocessing,)
    test_sample = build_epoch(test_subjects, recording,
                              crop_wake_mins, preprocessing)

    return train_sample, valid_sample, test_sample


def get_sample(train_dataset, transform_list, sample_size, random_state=None):
    rng = np.random.RandomState(random_state)
    tf_list_len = len(transform_list)
    len_aug_dataset = len(train_dataset) * tf_list_len
    subset_sample = rng.choice(
        range(int(len_aug_dataset / tf_list_len)),
        size=int(sample_size *
                 len_aug_dataset /
                 tf_list_len),
        replace=False)
    subset_aug_sample = np.array([(np.arange(i * tf_list_len, i * tf_list_len +
                                             tf_list_len))
                                  for i in subset_sample]).flatten()
    subset_aug_labels = np.array([(np.full(tf_list_len, train_dataset[i][1]))
                                  for i in subset_sample]).flatten()

    # train_subset = Subset(
    #     dataset=train_dataset,
    #     indices=subset_aug_sample)
    return subset_aug_sample, subset_aug_labels

    # TODO Trouver autre solution subset.
    # TODO mne-tools/mne-features


def build_epoch(subjects, recording, crop_wake_mins, preprocessing,
                train=True):
    dataset = SleepPhysionet(subject_ids=subjects,
                             recording_ids=recording,
                             crop_wake_mins=crop_wake_mins)

    if preprocessing:
        preprocessors = []
        if "microvolt_scaling" in preprocessing:
            preprocessors.append(NumpyPreproc(fn=lambda x: x * 1e6))
        if "filtering" in preprocessing:
            high_cut_hz = 30
            preprocessors.append(
                MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz)
            )

        # Transform the data
        preprocess(dataset, preprocessors)
    mapping = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }

    window_size_s = 30
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True,
        mapping=mapping)

    return windows_dataset


def tinying_dataset(concat_dataset, subset_dict):

    def take_dataset_subset(windows_dataset, indice_list):
        windows_dataset.windows = windows_dataset.windows[tuple(
            indice_list)]
        windows_dataset.y = windows_dataset.y[indice_list]
        windows_dataset.crop_inds = windows_dataset.crop_inds[indice_list]
        return(windows_dataset)

    for i in subset_dict.keys():
        concat_dataset.datasets[i] = take_dataset_subset(
            concat_dataset.datasets[i], subset_dict[i])
    concat_dataset.cumulative_sizes = concat_dataset.cumsum(
        concat_dataset.datasets)
    return(concat_dataset)

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

from joblib import Memory
import numpy as np
from braindecode.datautil.preprocess import (
    MNEPreproc, NumpyPreproc, preprocess)
from braindecode.datautil import create_windows_from_events
from braindecode.datasets import SleepPhysionet
cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)

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
    train_sample.tinying_dataset(train_tinying_dict)
    test_sample.tinying_dataset(test_tinying_dict)
    valid_sample.tinying_dataset(valid_tinying_dict)

    return train_sample, valid_sample, test_sample


def get_epochs_data(num_train=None, num_test=None, num_valid=None,
                    train_subjects=tuple(range(0, 50)),
                    valid_subjects=tuple(range(50, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    preprocessing=["microvolt_scaling", "filtering"], crop_wake_mins=30,
                    random_seed=None):

    if num_train is not None:
        np.random.seed(random_seed)
        rand_sub = np.random.choice(
            tuple(range(83)), size=num_train + num_test + num_valid, replace=True, p=None)
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
    subset_aug_sample = np.array([(np.arange(i * tf_list_len, i * tf_list_len
                                             + tf_list_len))
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

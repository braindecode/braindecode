import mne
import numpy as np
import scipy.io

from braindecode.datasets import BaseDataset, BaseConcatDataset


def _load_data_to_mne(file_path):
    # TODO: add option for loading test data
    f = scipy.io.loadmat(file_path)
    train_data = f['train_data']
    test_data = f['test_data']
    original_targets = f['train_dg']

    signal_sfreq = 1000
    original_target_sfreq = 25
    targets_stride = int(signal_sfreq / original_target_sfreq)

    upsampled_targets = np.full_like(original_targets, np.nan)
    upsampled_targets[::targets_stride] = original_targets[::targets_stride]

    test_targets = np.full((test_data.shape[0], original_targets.shape[1]), np.nan)
    # TODO: fix no target for test data
    test_targets[::targets_stride] = -1
    ch_names = [f'{i}' for i in range(train_data.shape[1])]
    ch_names += [f'target_{i}' for i in range(original_targets.shape[1])]
    ch_types = ['ecog' for _ in range(train_data.shape[1])] + ['misc' for _ in range(original_targets.shape[1])]

    info = mne.create_info(sfreq=signal_sfreq, ch_names=ch_names, ch_types=ch_types)
    info.target_sfreq = original_target_sfreq
    train_data = np.concatenate([train_data, upsampled_targets], axis=1)
    test_data = np.concatenate([test_data, test_targets], axis=1)

    raw_train = mne.io.RawArray(train_data.T, info=info)
    raw_test = mne.io.RawArray(test_data.T, info=info)
    #TODO: show how to resample targets
    return raw_train, raw_test


def load_bci_iv_ecog(dataset_path, subject_ids=None):
    if subject_ids is None:
        subject_ids = [1, 2, 3]
    files_list = [f'{dataset_path}/sub{i}_comp.mat' for i in subject_ids]
    datasets = []
    for file_path in files_list:
        raw_train, raw_test = _load_data_to_mne(file_path)
        desc_train = dict(
            subject=file_path.split('/')[-1].split('sub')[1][0],
            file_name=file_path.split('/')[-1],
            session='train'
        )
        desc_test = dict(
            subject=file_path.split('/')[-1].split('sub')[1][0],
            file_name=file_path.split('/')[-1],
            session='test'
        )
        datasets.append(BaseDataset(raw_train, description=desc_train))
        datasets.append(BaseDataset(raw_test, description=desc_test))
    return BaseConcatDataset(datasets)

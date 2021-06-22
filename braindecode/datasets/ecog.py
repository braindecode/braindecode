import mne
import numpy as np
import scipy.io

from braindecode.datasets import BaseDataset, BaseConcatDataset


def _load_data_to_mne(file_path):
    # TODO: add option for loading test data
    f = scipy.io.loadmat(file_path)
    train_data = f['train_data']
    original_targets = f['train_dg']

    signal_sfreq = 1000
    original_target_sfreq = 25
    targets_stride = int(signal_sfreq / original_target_sfreq)

    upsampled_targets = np.ones_like(original_targets) * np.nan
    upsampled_targets[::targets_stride] = original_targets[::targets_stride]

    ch_names = [f'{i}' for i in range(train_data.shape[1])]
    ch_names += [f'target_{i}' for i in range(original_targets.shape[1])]
    ch_types = ['ecog' for _ in range(train_data.shape[1])] + ['misc' for _ in range(original_targets.shape[1])]

    info = mne.create_info(sfreq=signal_sfreq, ch_names=ch_names, ch_types=ch_types)
    info.target_sfreq = original_target_sfreq
    train_data = np.concatenate([train_data, upsampled_targets], axis=1)
    raw_train = mne.io.RawArray(train_data.T, info=info)
    return raw_train


def load_bci_iv_ecog(dataset_path, subject_ids=None):
    if subject_ids is None:
        subject_ids = [1, 2, 3]
    files_list = [f'{dataset_path}/sub{i}_comp.mat' for i in subject_ids]
    datasets = []
    for file_path in files_list:
        raw = _load_data_to_mne(file_path)
        desc = dict(
            subject=file_path.split('/')[-1].split('sub')[1][0],
            file_name=file_path.split('/')[-1],
        )
        ds = BaseDataset(raw, description=desc)
        datasets.append(ds)
    return BaseConcatDataset(datasets)

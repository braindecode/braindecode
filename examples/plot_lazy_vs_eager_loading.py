"""Comparing eager and lazy loading
===================================

In this example, we compare the execution time and memory requirements of 1)
eager loading, i.e., preloading the entire data into memory and 2) lazy loaging,
i.e., only loading examples from disk when they are required.

While eager loading might be required for some preprocessing steps to be carried
out on continuous data (e.g., temporal filtering), it also allows fast access to
the data during training. However, this might come at the expense of large
memory usage, and can ultimately become impossible to do if the dataset does not
fit into memory (e.g., the TUH EEG dataset's >1,5 TB of recordings will not fit
in the memory of most machines).

Lazy loading avoids this potential memory issue by loading examples from disk
when they are required. This means large datasets can be used for training,
however this introduces some file-reading overhead every time an example must
be extracted. Some preprocessing steps that require continuous data also cannot
be applied as they normally would.

The following compares eager and lazy loading in a realistic scenario and shows
that...

For lazy loading to be possible, files must be saved in an MNE-compatible format
such as 'fif', 'edf', etc.
-> MOABB datasets are usually preloaded already?


Steps:
-> Initialize simple model
-> For loading in ('eager', 'lazy'):
    a) Load BNCI dataset with preload=True or False
    b) Apply raw transform (either eager, or keep it for later)
    b) Apply windower (either eager, or keep it for later)
    c) Add window transform (either eager, or keep it for later)
    d) Train for 10 epochs
-> Measure
    -> Total running time
    -> Time per batch
    -> Max and min memory consumption (or graph across time?)
    -> CPU/GPU usage across time


TODO:
- Automate the getting of TUH
- Cast data to float, targets to long in Dataset itself
    -> Should the conversion to torch.Tensor be made explicitly in the
       dataset class?

"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from braindecode.datasets import TUHAbnormal
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.transforms import transform_concat_ds
from braindecode.models import ShallowFBCSPNet, Deep4Net


mne.set_log_level('WARNING')  # avoid messages everytime a window is extracted

CUDA = True
TUH_PATH = '/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/'
WINDOW_LEN_S = 4
N_EPOCHS = 5
N_REPETITIONS = 10


def load_example_data(preload):
    subject_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    ds = TUHAbnormal(
        TUH_PATH, subject_ids=subject_ids, target_name="pathological",
        preload=preload)

    fs = ds.datasets[0].raw.info['sfreq']
    window_len_samples = int(fs * WINDOW_LEN_S)
    windows_ds = create_fixed_length_windows(
        ds, start_offset_samples=0, stop_offset_samples=None,
        supercrop_size_samples=window_len_samples,
        supercrop_stride_samples=window_len_samples, drop_samples=True,
        preload=preload)

    # Drop bad epochs
    for ds in windows_ds.datasets:
        ds.windows.drop_bad()
        assert ds.windows.preload == preload

    return windows_ds


def create_example_model(n_channels, n_classes, window_len_samples,
                         kind='shallow', cuda=False):
    if kind == 'shallow':
        model = ShallowFBCSPNet(
            n_channels, n_classes, input_time_length=window_len_samples,
            n_filters_time=40, filter_time_length=25, n_filters_spat=40,
            pool_time_length=75, pool_time_stride=15, final_conv_length='auto',
            split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
            drop_prob=0.5)
    elif kind == 'deep':
        model = Deep4Net(
            n_channels, n_classes, input_time_length=window_len_samples,
            final_conv_length='auto', n_filters_time=25, n_filters_spat=25,
            filter_time_length=10, pool_time_length=3, pool_time_stride=3,
            n_filters_2=50, filter_length_2=10, n_filters_3=100,
            filter_length_3=10, n_filters_4=200, filter_length_4=10,
            first_pool_mode="max", later_pool_mode="max", drop_prob=0.5,
            double_time_convs=False, split_first_layer=True, batch_norm=True,
            batch_norm_alpha=0.1, stride_before_pool=False)
    else:
        raise ValueError

    optimizer = optim.Adam(model.parameters())
    loss = nn.NLLLoss()

    if cuda:
        model.cuda()

    return model, loss, optimizer


def run_training(model, dataloader, loss, optimizer, n_epochs=1, cuda=False):
    for i in range(n_epochs):
        loss_vals = list()
        for X, y, _  in dataloader:
            model.train()
            model.zero_grad()

            X, y = X.float(), y.long()
            if cuda:
                X, y = X.cuda(), y.cuda()

            loss_val = loss(model(X), y)
            loss_vals.append(loss_val.item())

            loss_val.backward()
            optimizer.step()

        print(f'Epoch {i + 1} - mean training loss: {np.mean(loss_vals)}')

    return model


# ======================
# Lazy vs. eager loading
# ======================

times = list()

for i in range(N_REPETITIONS):
    print(f'\nRepetition {i + 1}/{N_REPETITIONS}:')
    for name, preload in zip(['lazy', 'eager'], [False, True]):
        print(f'\n{name} loading...\n')

        data_loading_start = time.time()
        dataset = load_example_data(preload=preload)
        data_loading_end = time.time()

        # Define data loader
        training_setup_start = time.time()
        dataloader = DataLoader(
            dataset, batch_size=256, shuffle=False, pin_memory=True,
            num_workers=4)

        # Instantiate model and optimizer
        n_channels = len(dataset.datasets[0].windows.ch_names)
        window_len_samples = len(dataset.datasets[0].windows.times)
        n_classes = 2
        model, loss, optimizer = create_example_model(
            n_channels, n_classes, window_len_samples, kind='deep', cuda=CUDA)
        training_setup_end = time.time()

        model_training_start = time.time()
        trained_model = run_training(
            model, dataloader, loss, optimizer, n_epochs=N_EPOCHS, cuda=CUDA)
        model_training_end = time.time()

        times.append({
            'loading_type': name,
            'data_preparation': data_loading_end - data_loading_start,
            'training_setup': training_setup_end - training_setup_start,
            'model_training': model_training_end - model_training_start
        })

times_df = pd.DataFrame(times)
print(times_df.groupby('loading_type').mean())
print(times_df.groupby('loading_type').median())

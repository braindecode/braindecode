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

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from braindecode.datasets import TUHAbnormal
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.transforms import transform_concat_ds
from braindecode.models import ShallowFBCSPNet

# =============
# Eager loading
# =============

###############################################################################
# Eager loading
# -------------
# First, we create a dataset by loading three subjects from the TUH Abnormal
# EEG dataset:
path = '/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/'
subject_ids = [0, 1, 2]
ds = TUHAbnormal(
    path, subject_ids=subject_ids, target_name="pathological", preload=True)

# Let's check whether the data is preloaded
print(ds.datasets[0].raw.preload)

##############################################################################
# We apply temporal filtering on the continuous data, which could not be done
# without loading the entire continuous recordings first.

# XXX: pick_types and pick_channels don't work in place
# XXX: can we use "apply_method" as a way to make this work?
transform_dict = OrderedDict({
    # 'pick_types': {"eeg": True, "meg": False, "stim": False},
    # 'pick_channels': [],
    # 'resample': {"sfreq": 100},
    'filter': {
        'l_freq': 3, 
        'h_freq': 30, 
        'picks': ['eeg']  # This controls which channels are filtered, but it keeps all of them.
    }
})

# The data should have zero-mean after filtering
print('Mean amplitude: ', ds.datasets[0].raw.get_data().mean())
print('Number of channels: ', ds.datasets[0].raw.get_data().shape)
transform_concat_ds(ds, transform_dict)
print('Mean amplitude: ', ds.datasets[0].raw.get_data().mean())
print('Number of channels: ', ds.datasets[0].raw.get_data().shape)

# ###############################################################################
# # We can easily split ds based on a criteria applied to the description
# # DataFrame:
# subsets = ds.split("session")
# print({subset_name: len(subset) for subset_name, subset in subsets.items()})

###############################################################################
# We create evenly spaced 4-s windows:

fs = ds.datasets[0].raw.info['sfreq']

window_len_samples = int(fs * 4)
windows_ds = create_fixed_length_windows(
    ds, start_offset_samples=0, stop_offset_samples=None,
    supercrop_size_samples=window_len_samples, 
    supercrop_stride_samples=window_len_samples, drop_samples=True, preload=True)

print(len(windows_ds))
for x, y, supercrop_ind in windows_ds:
    print(x.shape, y, supercrop_ind)
    break

# Let's check whether the data is preloaded
print(windows_ds.datasets[0].windows.preload)

###############################################################################
# We apply an additional filtering step, but this time on the windowed data.
# This is an example of a step that a user might choose to perform on-the-fly.

windows_transform_dict = OrderedDict({
    'filter': {
        'l_freq': 10, 
        'h_freq': 20, 
        'picks': ['eeg']  # This controls which channels are filtered, but it keeps all of them.
    }
})

print('Mean amplitude: ', windows_ds.datasets[0].windows.get_data().mean())
print('Number of channels: ', windows_ds.datasets[0].windows.get_data().shape)
transform_concat_ds(windows_ds, windows_transform_dict)
print('Mean amplitude: ', windows_ds.datasets[0].windows.get_data().mean())
print('Number of channels: ', windows_ds.datasets[0].windows.get_data().shape)

# ###############################################################################
# # Again, we can easily split windows_ds based on some criteria in the
# # description DataFrame:
# subsets = windows_ds.split("session")
# print({subset_name: len(subset) for subset_name, subset in subsets.items()})


###############################################################################
# We now have an eager-loaded dataset. We can use it to train a neural network.

use_cuda = False
n_epochs = 1

# Define data loader
dataloader = DataLoader(
    windows_ds, batch_size=128, shuffle=False, sampler=None, batch_sampler=None, 
    num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, 
    timeout=0, worker_init_fn=None)

# Instantiate model and optimizer
n_channels = len(windows_ds.datasets[0].windows.ch_names)
n_classes = 2
model = ShallowFBCSPNet(
    n_channels, n_classes, input_time_length=window_len_samples, 
    n_filters_time=40, filter_time_length=25, n_filters_spat=40, 
    pool_time_length=75, pool_time_stride=15, final_conv_length=30, 
    split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1, 
    drop_prob=0.5)

optimizer = optim.Adam(model.parameters())
if use_cuda:
    model.cuda()
    X, y = X.cuda(), y.cuda()
loss = nn.CrossEntropyLoss()

# Train model on fake data
for _ in range(n_epochs):
    for X, y, _  in dataloader:
        model.train()
        model.zero_grad()

        y_hat = torch.sum(model(X.float()), axis=-1)
        loss_val = loss(y_hat, y.long())
        print(loss_val)

        loss_val.backward()
        optimizer.step()

# start = time.time()
# duration = (time.time() - start) * 1e3 / n_minibatches  # in ms
# print(f'Took {duration} ms per minibatch.')

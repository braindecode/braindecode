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

"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict

import matplotlib.pyplot as plt
from IPython.display import display

from braindecode.datasets import TUHAbnormal
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.transforms import transform_concat_ds


###############################################################################
# Eager loading
# -------------
# First, we create a dataset based on BCIC IV 2a fetched with MOABB,
path = '/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/'
subject_ids = [0, 1, 2]
ds = TUHAbnormal(
    path, subject_ids=subject_ids, target_name="pathological", preload=True)

# Let's check whether the data is preloaded
print(ds.datasets[0].raw.preload)

alain

##############################################################################
# We apply temporal filtering on the continuous data.

# XXX: pick_types and pick_channels don't work in place
# XXX: can we use "apply_method" as a way to make this work?
transform_dict = OrderedDict({
    'pick_types': {"eeg": True, "meg": False, "stim": False},
    'pick_channels': [],
    'resample': {"sfreq": 100},
    'filter': {}
})
transform_concat_ds(ds, transform_dict)

###############################################################################
# We can easily split ds based on a criteria applied to the description
# DataFrame:
subsets = ds.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})

###############################################################################
# Next, we use a windower to extract events from the dataset based on events:
windows_ds = create_windows_from_events(
    ds, trial_start_offset_samples=0, trial_stop_offset_samples=1000,
    supercrop_size_samples=1000, supercrop_stride_samples=1000,
    drop_samples=False)

###############################################################################
# We can iterate through the windows_ds which yields a supercrop/window x,
# a target y, and supercrop_ind (which itself contains `i_supercrop_in_trial`,
# `i_start_in_trial`, and `i_stop_in_trial`, which are required for combining
# supercrop/window predictions in the scorer).
for x, y, supercrop_ind in windows_ds:
    print(x.shape, y, supercrop_ind)
    break

###############################################################################
# We visually inspect the supercrops/windows:
max_i = 2
fig, ax_arr = plt.subplots(1, max_i + 1, figsize=((max_i + 1) * 7, 5),
                           sharex=True, sharey=True)
for i, (x, y, supercrop_ind) in enumerate(windows_ds):
    ax_arr[i].plot(x.T)
    ax_arr[i].set_ylim(-0.0002, 0.0002)
    ax_arr[i].set_title(f"label={y}")
    if i == max_i:
        break

###############################################################################
# Alternatively, we can create evenly spaced ("sliding") windows using a
# different windower.
sliding_windows_ds = create_fixed_length_windows(
    ds, start_offset_samples=0, stop_offset_samples=None,
    supercrop_size_samples=1200, supercrop_stride_samples=1000,
    drop_samples=False)

print(len(sliding_windows_ds))
for x, y, supercrop_ind in sliding_windows_ds:
    print(x.shape, y, supercrop_ind)
    break

###############################################################################
# Transforms can also be applied on supercrops/windows in the same way as shown
# above on continuous data:

def crop_windows(windows, start_offset_samples, stop_offset_samples):
    fs = windows.info["sfreq"]
    windows.crop(tmin=start_offset_samples / fs, tmax=stop_offset_samples / fs,
                 include_tmax=False)

epochs_transform_dict = OrderedDict({
    "pick_types": {"eeg": True, "meg": False, "stim": False},
    crop_windows: {"start_offset_samples": 100, "stop_offset_samples": 900}
})

print(windows_ds.datasets[0].windows.info["ch_names"],
      len(windows_ds.datasets[0].windows.times))
transform_concat_ds(windows_ds, epochs_transform_dict)
print(windows_ds.datasets[0].windows.info["ch_names"],
      len(windows_ds.datasets[0].windows.times))

max_i = 2
fig, ax_arr = plt.subplots(1, max_i+1, figsize=((max_i+1)*7, 5),
                           sharex=True, sharey=True)
for i, (x, y, supercrop_ind) in enumerate(windows_ds):
    ax_arr[i].plot(x.T)
    ax_arr[i].set_ylim(-0.0002, 0.0002)
    ax_arr[i].set_title(f"label={y}")
    if i == max_i:
        break

###############################################################################
# Again, we can easily split windows_ds based on some criteria in the
# description DataFrame:
subsets = windows_ds.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})


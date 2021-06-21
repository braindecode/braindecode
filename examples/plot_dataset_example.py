"""MOABB Dataset Example
========================

In this example, we show how to fetch and prepare a MOABB dataset for usage
with Braindecode.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
from IPython.display import display

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing.windowers import \
    create_windows_from_events, create_fixed_length_windows
from braindecode.preprocessing.preprocess import preprocess, Preprocessor

###############################################################################
# First, we create a dataset based on BCIC IV 2a fetched with MOABB,
ds = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

###############################################################################
# ds has a pandas DataFrame with additional description of its internal datasets
display(ds.description)

##############################################################################
# We can iterate through ds which yields one time point of a continuous signal x,
# and a target y (which can be None if targets are not defined for the entire
# continuous signal).
for x, y in ds:
    print(x.shape, y)
    break

##############################################################################
# We can apply preprocessing transforms that are defined in mne and work
# in-place, such as resampling, bandpass filtering, or electrode selection.
preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=True),
    Preprocessor('resample', sfreq=100)
]
print(ds.datasets[0].raw.info["sfreq"])
preprocess(ds, preprocessors)
print(ds.datasets[0].raw.info["sfreq"])

###############################################################################
# We can easily split ds based on a criteria applied to the description
# DataFrame:
subsets = ds.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})

###############################################################################
# Next, we use a windower to extract events from the dataset based on events:
windows_ds = create_windows_from_events(
    ds, start_offset_samples=0, stop_offset_samples=100,
    window_size_samples=400, window_stride_samples=100,
    drop_last_window=False)

###############################################################################
# We can iterate through the windows_ds which yields a window x,
# a target y, and window_ind (which itself contains `i_window_in_trial`,
# `i_start_in_trial`, and `i_stop_in_trial`, which are required for combining
# window predictions in the scorer).
for x, y, window_ind in windows_ds:
    print(x.shape, y, window_ind)
    break

###############################################################################
# We visually inspect the windows:
max_i = 2
fig, ax_arr = plt.subplots(1, max_i + 1, figsize=((max_i + 1) * 7, 5),
                           sharex=True, sharey=True)
for i, (x, y, window_ind) in enumerate(windows_ds):
    ax_arr[i].plot(x.T)
    ax_arr[i].set_ylim(-0.0002, 0.0002)
    ax_arr[i].set_title(f"label={y}")
    if i == max_i:
        break

###############################################################################
# Alternatively, we can create evenly spaced ("sliding") windows using a
# different windower.
sliding_windows_ds = create_fixed_length_windows(
    ds, start_offset_samples=0, stop_offset_samples=0,
    window_size_samples=1200, window_stride_samples=1000,
    drop_last_window=False)

print(len(sliding_windows_ds))
for x, y, window_ind in sliding_windows_ds:
    print(x.shape, y, window_ind)
    break

###############################################################################
# Transforms can also be applied on windows in the same way as shown
# above on continuous data:


def crop_windows(windows, start_offset_samples, stop_offset_samples):
    fs = windows.info["sfreq"]
    windows.crop(tmin=start_offset_samples / fs, tmax=stop_offset_samples / fs,
                 include_tmax=False)


epochs_preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    Preprocessor(crop_windows, apply_on_array=False, start_offset_samples=100,
                 stop_offset_samples=900)
]

print(windows_ds.datasets[0].windows.info["ch_names"],
      len(windows_ds.datasets[0].windows.times))
preprocess(windows_ds, epochs_preprocessors)
print(windows_ds.datasets[0].windows.info["ch_names"],
      len(windows_ds.datasets[0].windows.times))

max_i = 2
fig, ax_arr = plt.subplots(1, max_i + 1, figsize=((max_i + 1) * 7, 5),
                           sharex=True, sharey=True)
for i, (x, y, window_ind) in enumerate(windows_ds):
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

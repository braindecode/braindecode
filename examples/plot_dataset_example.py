"""MOABB Dataset Example
========================

Showcasing how to fetch and prepare a MOABB dataset for usage with Braindecode.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.display import display

from braindecode.datasets import MOABBDataset
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.transforms import transform_concat_ds

###############################################################################
# Create a dataset based on BCIC IV 2a fetched with MOABB
ds = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])
# ds has a pandas DataFrame with additional description of its datasets
display(ds.description)

##############################################################################
# We can iterate through ds which yields one time point of a continuous signal x,
# and a target y (which can be None if targets are not defined for the entire
# continuous signal but for events within the continuous signal)
for x, y in ds:
    print(x.shape, y)
    break

# we can apply preprocessing transforms which work in-place, such as resampling,
# bandpass filtering, or electrode selection
raw_transform_dict = OrderedDict({
    "pick_types": {"eeg": True, "meg": False, "stim": True},
    "resample": {"sfreq": 100},
})
print(ds.datasets[0].raw.info["sfreq"])
transform_concat_ds(ds, raw_transform_dict)
print(ds.datasets[0].raw.info["sfreq"])

###############################################################################
# We can easily split ds based on a criteria in the description DataFrame
subsets = ds.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})

###############################################################################
# We can create a windower to extract events from the dataset
windows_ds = create_windows_from_events(
    ds, trial_start_offset_samples=0, trial_stop_offset_samples=1000,
    supercrop_size_samples=1000, supercrop_stride_samples=1000,
    drop_samples=False)

###############################################################################
# We can iterate through the windows_ds which yields a supercrop/window x,
# a target y, and supercrop_ind as i_supercrop_in_trial, i_start_in_trial, and
# i_stop_in_trial which is required for combining supercrop/window predictions
# in the scorer
for x, y, supercrop_ind in windows_ds:
    print(x.shape, y, supercrop_ind)
    break

###############################################################################
# We can inspect the supercrops/windows
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
# or apply further preprocessing on the supercrops/windows
def crop_windows(windows, start_offset_samples, stop_offset_samples):
    fs = windows.info["sfreq"]
    windows.crop(tmin=start_offset_samples/fs, tmax=stop_offset_samples/fs,
                 include_tmax=False)

def scale_windows(windows, factor):
    windows.load_data()
    windows._data *= factor

windows_transforms_dict = OrderedDict({
    "pick_types": {"eeg": True, "meg": False, "stim": False},
    scale_windows: {"factor": 1e6},
    crop_windows: {"start_offset_samples": 100, "stop_offset_samples": 900}
})
transform_concat_ds(windows_ds, windows_transforms_dict)

max_i = 2
fig, ax_arr = plt.subplots(1, max_i+1, figsize=((max_i+1)*7, 5),
                           sharex=True, sharey=True)
for i, (x, y, supercrop_ind) in enumerate(windows_ds):
    ax_arr[i].plot(x.T)
    ax_arr[i].set_title(f"label={y}")
    if i == max_i:
        break

print(windows_ds.datasets[0].windows.info["sfreq"])

###############################################################################
# Again, we can easily split windows_ds based on some criteria in the
# description DataFrame
subsets = windows_ds.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})

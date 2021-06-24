"""
MNE Dataset Example
===================
"""
##############################################################################
# This example shows how to convert data from mne.Raws or mne.Epochs to a
# braindecode compatible data format.

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne

from braindecode.datasets import (
    create_from_mne_raw, create_from_mne_epochs)

###############################################################################
# First, fetch some data using mne:

# 5, 6, 7, 10, 13, 14 are codes for executed and imagined hands/feet
subject_id = 22
event_codes = [5, 6, 9, 10, 13, 14]
# event_codes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(
    subject_id, event_codes, update_path=False)

# Load each of the files
parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto')
         for path in physionet_paths]

###############################################################################
# Convert mne.RawArrays to a compatible data format:
descriptions = [{"event_code": code, "subject": subject_id}
                for code in event_codes]
windows_dataset = create_from_mne_raw(
    parts,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=500,
    window_stride_samples=500,
    drop_last_window=False,
    descriptions=descriptions,
)

###############################################################################
# If trials were already cut beforehand and are available as mne.Epochs:
list_of_epochs = [mne.Epochs(raw, [[0, 0, 0]], tmin=0, baseline=None)
                  for raw in parts]
windows_dataset = create_from_mne_epochs(
    list_of_epochs,
    window_size_samples=50,
    window_stride_samples=50,
    drop_last_window=False
)

windows_dataset.description

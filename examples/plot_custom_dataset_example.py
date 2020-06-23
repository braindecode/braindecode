"""Custom Dataset Example
=========================
"""
##############################################################################
# This example shows how to convert data from X and y to a braindecode
# compatible data format.

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne

from braindecode.datautil.xy import create_from_X_y

###############################################################################
# To set up the example, we first fetch some data using mne:

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
# We take the required data, targets and additional information sampling
# frequency and channel names from the loaded data. Note that this data and
# information can originate from any source.
X = [raw.get_data() for raw in parts]
y = event_codes
sfreq = parts[0].info["sfreq"]
ch_names = parts[0].info["ch_names"]

###############################################################################
# Convert to data format compatible with skorch and braindecode:
windows_dataset = create_from_X_y(
    X, y, drop_last_window=False, sfreq=sfreq, ch_names=ch_names,
    window_stride_samples=500,
    window_size_samples=500,
)

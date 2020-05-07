"""Custom Dataset Example
=========================
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne

from braindecode.datasets.datasets import create_from_X_y

###############################################################################
# To set up the example, we first fetch some data using mne.

# 5, 6, 7, 10, 13, 14 are codes for executed and imagined hands/feet
subject_id = 22
event_codes = [5, 6, 9, 10, 13, 14]
# event_codes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

# Load each of the files
parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto', verbose='WARNING')
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
# Convert to data format compatible with skorch and braindecode
windows_datasets = create_from_X_y(X=X, y=y, sfreq=sfreq, ch_names=ch_names)

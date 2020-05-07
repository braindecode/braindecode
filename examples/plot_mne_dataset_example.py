"""MNE Dataset Example
======================
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne

from braindecode.datasets.datasets import create_from_mne_raw

###############################################################################
# First, fetch some data using mne.

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
# Convert to data format compatible with skorch and braindecode
windows_datasets = create_from_mne_raw(
    parts,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    supercrop_size_samples=500,
    supercrop_stride_samples=500,
    drop_samples=False,
    description=[{"event_code": code, "subject": subject_id}
                 for code in event_codes],
)

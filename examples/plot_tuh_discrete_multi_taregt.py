"""Multiple discrete targets with the TUH EEG Corpus
====================================================

In this example, we showcase usage of multiple discrete targets per recording
with the TUH EEG Corpus.
"""

# Author: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne
from torch.utils.data.dataloader import DataLoader

from braindecode.datasets import TUH
from braindecode.preprocessing.windowers import create_fixed_length_windows

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted


###############################################################################
# If you want to try this code with the actual data, please delete the next
# section. We are required to mock some dataset functionality, since the data
# is not available at creation time of this example.
from braindecode.datasets.tuh import _TUHMock as TUH  # noqa F811


###############################################################################
# We start by creating a TUH dataset. First, the class generates a description
# of the recordings in `TUH_PATH` (which is later accessible as
# `tuh.description`) without actually touching the files. This will parse
# information from file paths such as patient id, recording data, etc and should
# be really fast. Afterwards, the files are sorted chronologically by year,
# month, day, patient id, recording session and segment.
# In the following, a subset of the description corresponding to `recording_ids`
# is used.
# Afterwards, the files will be iterated a second time, slower than before.
# The files are now actually touched. Additional information about subjects
# like age and gender are parsed directly from the EDF file header. If existent,
# the physician report is added to the description. Furthermore, the recordings
# are read with `mne.io.read_raw_edf` with `preload=False`. Finally, we will get
# a `BaseConcatDataset` of `BaseDatasets` each holding a single
# `nme.io.Raw` which is fully compatible with other braindecode functionalities.
# # In the constructor, we do not specify the target as a string, but actually
# give multiple strings as a list.

TUH_PATH = 'please insert actual path to data here'
tuh = TUH(
    path=TUH_PATH,
    recording_ids=None,
    target_name=('age', 'gender'),  # use both age and gender as decoding target
    preload=False,
    add_physician_reports=False,
)


###############################################################################
# Iterating through the dataset gives x as ndarray(n_channels x 1) as well as
# the target as [age of the subject, gender of the subject].
tuh[0]


###############################################################################
# We will skip preprocessing steps for now, since it is not the aim of this
# example. Instead, we will directly create compute windows. We specify a
# mapping from genders 'M' and 'F' to integers, since this is required for
# decoding.

tuh_windows = create_fixed_length_windows(
    tuh,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=1000,
    window_stride_samples=1000,
    drop_last_window=False,
    mapping={'M': 0, 'F': 1},  # map non-digit targets
)
# store the number of windows required for loading later on
tuh_windows.set_description({
    "n_windows": [len(d) for d in tuh_windows.datasets]})


###############################################################################
# Iterating through the dataset gives x as ndarray(n_channels x 1000), y as
# [age, gender], and ind.
tuh_windows[0]


###############################################################################
# We give the dataset to a pytorch DataLoader, such that it can be used for
# model training.
dl = DataLoader(
    dataset=tuh_windows,
    batch_size=4,
)


###############################################################################
# Iterating through the DataLoader gives batch_X as tensor(4 x n_channels x
# 1000), batch_y as [tensor([4 x age of subject]), tensor([4 x gender of
# subject])], and batch_ind.
for batch_X, batch_y, batch_ind in dl:
    break
batch_X, batch_y, batch_ind

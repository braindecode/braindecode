"""Process TUH EEG Corpus and use multiple discrete targets per recording
=========================================================================

In this example, we showcase usage of multiple targets per recording with
the TUH EEG Corpus.
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
# We start by creating a TUH dataset with 20 recordings of the TUH EEG Corpus.
# In the constructor, we do not specify the target as a string, but actually
# give multiple strings as a list.
TUH_PATH = '/data/datasets/TUH/EEG/tuh_eeg/v1.2.0/edf/'
tuh = TUH(
    path=TUH_PATH,
    recording_ids=range(2),  # use a very tiny subset
    target_name=['age', 'gender'],  # use both age and gender as decoding target
    preload=False,
    add_physician_reports=False,
)


###############################################################################
# Iterating through the dataset gives x as n_channels x 1 as well as the
# target as a [age of the subject, gender of the subject].
x, y = tuh[0]


###############################################################################
# We will skip preprocessing steps for now, since it is not the aim of this
# example. Instead, we will directly create compute windows. We specify a
# mapping from genders 'M' and 'F' to integers, since fit is required for
# decoding.

tuh_windows = create_fixed_length_windows(
    tuh,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=1000,
    window_stride_samples=1000,
    drop_last_window=False,
    mapping={'M': 0, 'F': 1},
)
# store the number of windows required for loading later on
tuh_windows.description["n_windows"] = [len(d) for d in
                                        tuh_windows.datasets]


###############################################################################
# Iterating through the dataset gives x as n_channels x 1000, y as
# [age, gender], and ind.
x, y, ind = tuh_windows[0]


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

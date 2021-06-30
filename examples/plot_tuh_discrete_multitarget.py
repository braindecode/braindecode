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
# We start by creating a TUH dataset. Instead of just a str, we give it
# multiple strings as target names. Each of the strings has to exist as a
# column in the description DataFrame.

TUH_PATH = 'please insert actual path to data here'
tuh = TUH(
    path=TUH_PATH,
    recording_ids=None,
    target_name=('age', 'gender'),  # use both age and gender as decoding target
    preload=False,
    add_physician_reports=False,
)
tuh.description


###############################################################################
# Iterating through the dataset gives x as ndarray(n_channels x 1) as well as
# the target as [age of the subject, gender of the subject]. Let's look at the last example
# as it has more interesting age/gender labels (compare to the last row of the dataframe above).
x, y = tuh[-1]
print('x:', x)
print('y:', y)


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
# [age, gender], and ind. Let's look at the last example again.
x, y, ind = tuh_windows[-1]
print('x:', x)
print('y:', y)
print('ind:', ind)


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
# subject])], and batch_ind. We will iterate to the end to look at the last example
# again.
for batch_X, batch_y, batch_ind in dl:
    pass
print('batch_X:', batch_X)
print('batch_y:', batch_y)
print('batch_ind:', batch_ind)

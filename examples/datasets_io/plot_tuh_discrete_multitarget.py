"""
Multiple discrete targets with the TUH EEG Corpus
=================================================

Welcome to this tutorial where we demonstrate how to work with multiple discrete
 targets for each recording in the TUH EEG Corpus. We'll guide you through the
 process step by step.

"""

# Author: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import mne
from torch.utils.data import DataLoader

from braindecode.datasets import TUH
from braindecode.preprocessing import create_fixed_length_windows

# Setting Logging Level
# ----------------------
#
# We'll set the logging level to 'ERROR' to avoid excessive messages when
# extracting windows:

mne.set_log_level('ERROR')  # avoid messages every time a window is extracted


###############################################################################
# If you want to try this code with the actual data, please delete the next
# section. We are required to mock some dataset functionality, since the data
# is not available at creation time of this example.
from braindecode.datasets.tuh import _TUHMock as TUH  # noqa F811


###############################################################################
# Creating Temple University Hospital (TUH) EEG Corpus Dataset
# ------------------------------------------------------------
#
# We start by creating a TUH dataset. Instead of just a `str, we give it
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
print(tuh.description)

###############################################################################
# Exploring Data
# --------------
#
# Iterating through the dataset gives `x` as an ndarray with shape
# `(n_channels x 1)` and `y` as a list containing `[age of the subject, gender
# of the subject]`.
# Let's look at the last example as it has more interesting age/gender labels
# (compare to the last row of the dataframe above).
x, y = tuh[-1]

print(f'{x=}\n{y=}')


###############################################################################
# Creating Windows
# ----------------
#
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
# Exploring Windows
# -----------------
#
# Iterating through the dataset gives `x` as an ndarray with shape
# `(n_channels x 1000)`, `y` as `[age, gender]`, and `ind`.
# Let's look at the last example again.
x, y, ind = tuh_windows[-1]
print(f'{x=}\n{y=}\n{ind=}')


###############################################################################
# DataLoader for Model Training
# -----------------------------
#
# We give the dataset to a pytorch DataLoader, such that it can be used for
# model training.
dl = DataLoader(
    dataset=tuh_windows,
    batch_size=4,
)


###############################################################################
# Exploring DataLoader
# --------------------
#
# When iterating through the DataLoader, we get `batch_X` as a tensor with shape
# `(4 x n_channels x 1000)`, `batch_y` as `[tensor([4 x age of subject]),
# tensor([4 x gender of subject])]`, and `batch_ind`. To view the last example,
# simply iterate through the DataLoader:

for batch_X, batch_y, batch_ind in dl:
    pass

print(f'{batch_X=}\n{batch_y=}\n{batch_ind=}')

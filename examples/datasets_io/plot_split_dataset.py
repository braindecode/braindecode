"""
Split Dataset Example
=====================

In this example, we aim to show multiple ways of how you can split your datasets for
training, testing, and evaluating your models.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

###############################################################################
# Loading the dataset
# -------------------------------------
#
# Firstly, we create a dataset using the braindecode class <MOABBDataset> to load
# it fetched from MOABB. In this example, we're using Dataset 2a from BCI
# Competition IV.

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

###############################################################################
# Splitting
# -------------------------------------
#
# By description information
# ~~~~~~~~~~~~~
#
# The class <MOABBDataset> has a pandas DataFrame containing additional
# description of its internal datasets, which can be used to help splitting the data
# based on recording information, such as subject, session, and run of each trial.

dataset.description

###############################################################################
# Here, we're splitting the data based on different runs. The method split returns
# a dictionary with string keys corresponding to unique entries in the description
# DataFrame column.

splits = dataset.split("run")
print(splits)
splits["4"].description

###############################################################################
# By row index
# ~~~~~~~~~~~~~
#
# Another way we can split the dataset is based on a list of integers corresponding to
# rows in the description. In this case, the returned dictionary will have
# '0' as the only key.

splits = dataset.split([0, 1, 5])
print(splits)
splits["0"].description

###############################################################################
# However, if we want multiple splits based on indices, we can also define a list
# containing lists of integers. In this case, the dictionary will have string keys
# representing the index of the dataset split in the order of the given list of
# integers.

splits = dataset.split([[0, 1, 5], [2, 3, 4], [6, 7, 8, 9, 10, 11]])
print(splits)
splits["2"].description

###############################################################################
# You can also name each split in the output dictionary by specifying the keys
# of each list of indexes in the input dictionary:

splits = dataset.split(
    {"train": [0, 1, 5], "valid": [2, 3, 4], "test": [6, 7, 8, 9, 10, 11]}
)
print(splits)
splits["test"].description

###############################################################################
# Observation
# -------------------------------------
#
# Similarly, we can split datasets after creating windows using the same methods.

windows = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0)

###############################################################################

# Splitting by different runs
print("Using description info")
splits = windows.split("run")
print(splits)
print()

# Splitting by row index
print("Splitting by row index")
splits = windows.split([4, 8])
print(splits)
print()

print("Multiple row index split")
splits = windows.split([[4, 8], [5, 9, 11]])
print(splits)
print()

# Specifying output's keys
print("Specifying keys")
splits = windows.split(dict(train=[4, 8], test=[5, 9, 11]))
print(splits)

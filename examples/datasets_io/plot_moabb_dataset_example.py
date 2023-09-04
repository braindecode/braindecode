"""MOABB Dataset Example
========================

In this example, we show how to fetch and prepare a MOABB dataset for usage
with Braindecode.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          Daniel Wilson <dan.c.wil@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor

###############################################################################
# First, we create a dataset based on BCIC IV 2a fetched with MOABB,
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

###############################################################################
# The dataset has a pandas DataFrame with additional description of its internal datasets
dataset.description

##############################################################################
# We can iterate through dataset which yields one time point of a continuous signal x,
# and a target y (which can be None if targets are not defined for the entire
# continuous signal).
for x, y in dataset:
    print(x.shape, y)
    break

##############################################################################
# We can apply preprocessing transforms that are defined in mne and work
# in-place, such as resampling, bandpass filtering, or electrode selection.
preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=True),
    Preprocessor('resample', sfreq=100)
]
print(dataset.datasets[0].raw.info["sfreq"])
preprocess(dataset, preprocessors)
print(dataset.datasets[0].raw.info["sfreq"])

###############################################################################
# We can easily split the dataset based on a criteria applied to the description
# DataFrame:
subsets = dataset.split("session")
print({subset_name: len(subset) for subset_name, subset in subsets.items()})

##############################################################################
# See our `Trialwise Decoding <../model_building/plot_bcic_iv_2a_moabb_trial.html>`__ and
# `Cropped Decoding <../model_building/plot_bcic_iv_2a_moabb_cropped.html>`__ examples for
# training with this dataset.

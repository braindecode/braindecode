"""
Pytorch-compatible braindecode dataset from MOABB
=================================================

In this example we show how to create a pytorch compatible braindecode dataset
along with preprocessing and windowing steps. More specifically we introduce two
types of transformers:

1. raw transformers, that work on raw objects
2. window transformers, that work on numpy arrays representing windows

.. warning::
    This code will download subject 4 of the BCI competition IV 2a dataset via
    moabb

"""
# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from sklearn.pipeline import Pipeline

from braindecode.datautil.transforms import (
    FilterRaw,
    ZscoreRaw,
    ZscoreWindow,
    FilterWindow,
)
from braindecode.datautil.windowers import EventWindower
from braindecode.datasets.moabb_datasets import MOABBDataset
##############################################################################
# Define transformers that operate on raw objects
# -----------------------------------------------
#
# There are currently two types of transformers. Here, we define
# 1. the raw transformers that are applied first on continuous data
# 2. the windower that transforms raw data into epoched data
# 3. window transformers for processing steps applied on windowed data
#
# For demonstration purposes only, we apply filtering and normalization to both
# the raw and windowed data, however this shouldn't be done in practice.
#
# .. warning::
#
#    Don't do this at home!
#


# 1. raw transformers
# define band-pass filter to be used on raw data
filter_raw = FilterRaw(l_freq=1, h_freq=80)

# define zscore transformer for channel wise normalization
zscorer_raw = ZscoreRaw()

# transformers can be chained using the scikit-learn Pipeline object
raw_transformer = [filter_raw, zscorer_raw]

# 2. window data
# define mapping for event windower

# define event windower
event_windower = EventWindower(
    200, 200, True, tmin=0
)

# 2nd case
# fixed_length_windower = FixedLengthWindower(window_size_samples=200, tmin=0,
# chunk_duration_samples=200, mapping=mapping)
# 3nd case
# windower = FixedLengthWindower(window_size_samples=None,
# overlap_size_samples=0)

# 3. window transformers
# define FIR filter for windowed data. The sampling frequency 'sfreq' has to be
# specified by the user as there is no information contained in the
# 'numpy.array' the filter later is applied to.
filter_window = FilterWindow(sfreq=250, l_freq=8, h_freq=12)

# zscore for normalization
zscorer_window = ZscoreWindow()

# again, transformers can be chained using the scikit-learn Pipeline object
array_transformer = filter_window  # [filter_window, zscorer_window]

##############################################################################
# Instantiate pytorch based braindecode dataset
# ---------------------------------------------
#
#
dataset = 'BNCI2014001'
bnci2014001 = MOABBDataset(dataset, subject=4, raw_transformer=raw_transformer,
                           windower=event_windower,
                           transformer=array_transformer,
                           transform_online=True)

print(f'As expected, the number of epochs is {len(bnci2014001)} (2 sessions\n'
      f'of 6 runs with 12 repetitions of 4 motor imagery tasks with 5 windows\n'
      f'each)')

x, y = bnci2014001[0]

print(f'As expected, the shape of the first epoch is {x.shape} and its class'
      f'label is {y}')

# ToDo
# - labels in FixedLength
# - metadata of session and run information (from moabb)
# - lazyloading
# - transform_online=False

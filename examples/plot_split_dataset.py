"""Split Dataset Example
========================

In this example, we show multiple ways of how to split datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from IPython.display import display

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing.windowers import create_windows_from_events

###############################################################################
# First, we create a dataset based on BCIC IV 2a fetched with MOABB,
ds = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

###############################################################################
# ds has a pandas DataFrame with additional description of its internal datasets
display(ds.description)

###############################################################################
# We can split the dataset based on the info in the description, for example
# based on different runs. The returned dictionary will have string keys
# corresponding to unique entries in the description DataFrame column
splits = ds.split("run")
display(splits)
display(splits["run_4"].description)

###############################################################################
# We can also split the dataset based on a list of integers corresponding to
# rows in the description. In this case, the returned dictionary will have
# '0' as the only key
splits = ds.split([0, 1, 5])
display(splits)
display(splits["0"].description)

###############################################################################
# If we want multiple splits based on indices, we can also specify a list of
# list of integers. In this case, the dictionary will have string keys
# representing the id of the dataset split in the order of the given list of
# integers
splits = ds.split([[0, 1, 5], [2, 3, 4], [6, 7, 8, 9, 10, 11]])
display(splits)
display(splits["2"].description)

###############################################################################
# Similarly, we can split datasets after creating windows
windows = create_windows_from_events(
    ds, start_offset_samples=0, stop_offset_samples=0)
splits = windows.split("run")
display(splits)
splits = windows.split([4, 8])
display(splits)
splits = windows.split([[4, 8], [5, 9, 11]])
display(splits)

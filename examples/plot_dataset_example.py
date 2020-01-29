"""
Simple Moabb Dataset Example
=========================

Showcasing how to fetch and crop a moabb dataset.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets import MOABBDataset

# create a dataset based on BCIC IV 2a fetched with moabb
ds = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[4],
                  trial_start_offset_samples=0, trial_stop_offset_samples=1000,
                  supercrop_size_samples=1000, supercrop_stride_samples=1000)

# we can iterate through ds which yields an example x, target y,
# and info as i_supercrop_in_trial, i_start_in_trial, and i_stop_in_trial
# which is required for combining supercrop predictions in the scorer
for x, y, info in ds:
    print(x.shape, y, info)
    break

# each base_ds in ds has its own info DataFrame
print(ds.datasets[-1].info)
# ds has a concattenation of all DataFrames of its datasets
print(ds.info)

# we can easily split ds based on a criterium in the info DataFrame
subsets = ds.split("session")
print(subsets)

# again we can iterate through the subsets as through the ds
for x, y, info in subsets["session_E"]:
    print(x.shape, y, info)
    break

# create a dataset based on TUH Abnormal EEG Corpus (v2.0.0)
# for this dataset, no events exist but a label (pathological / non-pathological
# is valid for the entire recording
# ds = TUHAbnormal(path="/path/to/the/directory/",
#                  subject_ids=[0, 1], trial_start_offset_samples=0,
#                  trial_stop_offset_samples=1000, supercrop_size_samples=1000,
#                  supercrop_stride_samples=1000, mapping={False: 0, True: 1})

# as before, we can iterate through the dataset, getting the same kind of info
# for x, y, info in ds:
#     print(x.get_data().shape, y, info)
#     break

# we can change the target for this dataset to 'age'
# ds = TUHAbnormal(path="/path/to/the/directory/",
#                  subject_ids=[0, 1], trial_start_offset_samples=0,
#                  trial_stop_offset_samples=1000, supercrop_size_samples=1000,
#                  supercrop_stride_samples=1000, target="age",
#                  mapping={False: 0, True: 1})

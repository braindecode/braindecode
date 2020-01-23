# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets.moabb_datasets import BNCI2014001

# create a dataset based on BCIC IV 2a fetched with moabb
ds = BNCI2014001(subject_ids=[4], trial_start_offset_samples=0,
                 trial_stop_offset_samples=1000, supercrop_size_samples=1000,
                 supercrop_stride_samples=1000)

# we can iterate through ds which yields an example x, target y,
# and info as i_supercrop_in_trial, i_start_in_trial, and i_stop_in_trial
# which is required for combining supercrop predictions in the scorer
for x, y, info in ds:
    print(x.get_data().shape, y, info)
    break

# each base_ds in ds has its own info DataFrame
print(ds.datasets[-1].info)
# ds has a concattenation of all DataFrames of its datasets
print(ds.info)

# we can easily split ds based on a criterium in the info DataFrame
eval_set, train_set = ds.split("session")

# quick check whether the split did what we intended
print(ds.info.iloc[train_set.indices])
print(ds.info.iloc[eval_set.indices])

# again we can iterate through the subsets as through the ds
for x, y, info in eval_set:
    print(x.get_data().shape, y, info)
    break

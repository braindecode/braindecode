"""
Load and save dataset example
=============================

In this example, we show how to load and save braindecode datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import preprocess, MNEPreproc
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.preprocessing.windowers import create_windows_from_events


###############################################################################
# First, we load some dataset using MOABB.
ds = MOABBDataset(
    dataset_name='BNCI2014001',
    subject_ids=[1],
)

###############################################################################
# We can apply preprocessing steps to the dataset. It is also possible to skip
# this step and not apply any preprocessing.
preprocess(
    concat_ds=ds,
    preprocessors=[MNEPreproc(fn='resample', sfreq=10)]
)

###############################################################################
# We save the dataset to a an existing directory. It will create a '.fif' file
# for every dataset in the concat dataset. Additionally it will create two
# JSON files, the first holding the description of the dataset, the second
# holding the name of the target. If you want to store to the same directory
# several times, for example due to trying different preprocessing, you can
# choose to overwrite the existing files.
ds.save(
    path='./',
    overwrite=False,
)

##############################################################################
# We load the saved dataset from a directory. Signals can be preloaded in
# compliance with mne. Optionally, only specific '.fif' files can be loaded
# by specifying their ids. The target name can be changed, if the dataset
# supports it (TUHAbnormal for example supports 'pathological', 'age', and
# 'gender'. If you stored a preprocessed version with target 'pathological'
# it is possible to change the target upon loading).
ds_loaded = load_concat_dataset(
    path='./',
    preload=True,
    ids_to_load=[1, 3],
    target_name=None,
)

##############################################################################
# The serialization utility also supports WindowsDatasets, so we create
# compute windows next.
windows_ds = create_windows_from_events(
    concat_ds=ds_loaded,
    start_offset_samples=0,
    stop_offset_samples=0,
)

##############################################################################
# Again, we save the dataset to an existing directory. It will create a
# '-epo.fif' file for every dataset in the concat dataset. Additionally it
# will create a JSON file holding the description of the dataset. If you
# want to store to the same directory several times, for example due to
# trying different windowing parameters, you can choose to overwrite the
# existing files.
windows_ds.save(
    path='./',
    overwrite=True,
)

##############################################################################
# Load the saved dataset from a directory. Signals can be preloaded in
# compliance with mne. Optionally, only specific '-epo.fif' files can be
# loaded by specifying their ids.
windows_ds_loaded = load_concat_dataset(
    path='./',
    preload=False,
    ids_to_load=[0],
    target_name=None,
)

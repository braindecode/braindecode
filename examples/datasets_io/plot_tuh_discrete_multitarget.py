""".. _multiple-discrete-targets-TUH:

Multiple discrete targets with the TUH EEG Corpus
=================================================

Welcome to this tutorial where we demonstrate how to work with multiple discrete
targets for each recording in the TUH EEG Corpus. We'll guide you through the
process step by step.

"""

# Author: Lukas Gemein <l.gemein@gmail.com>
#         Daria Agafonova
#
# License: BSD (3-clause)

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from braindecode.datasets import TUH, BaseConcatDataset, RawDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.util import set_random_seeds

# Setting Logging Level
# ----------------------
#
# We'll set the logging level to 'ERROR' to avoid excessive messages when
# extracting windows:

mne.set_log_level("ERROR")  # avoid messages every time a window is extracted

cuda = torch.cuda.is_available()
seed = 20240205
set_random_seeds(seed=seed, cuda=cuda)


###############################################################################
# If you want to try this code with the actual data, please delete the next
# section. We are required to mock some dataset functionality, since the data
# is not available at creation time of this example.
# We use the mock's four-recording v1.1.0 corpus so the subset example below
# includes recordings with different labels.
from braindecode.datasets.tuh import _TUHMock as TUH  # noqa F811


###############################################################################
# Creating Temple University Hospital (TUH) EEG Corpus Dataset
# ------------------------------------------------------------
#
# We start by creating a TUH dataset. Instead of just a `str`, we give it
# multiple strings as target names. Each of the strings has to exist as a
# column in the description DataFrame.

TUH_PATH = "please insert actual path to data here"
tuh = TUH(
    path=TUH_PATH,
    version="v1.1.0",
    recording_ids=None,
    target_name=("age", "gender"),  # use both age and gender as decoding target
    preload=False,
    add_physician_reports=False,
)
print(tuh.description)

###############################################################################
# Exploring Data
# --------------
#
# Iterating through the dataset gives ``x`` as an ndarray with shape
# ``(n_channels x 1)`` and ``y`` as a list containing ``[age of the subject, gender
# of the subject]``.
# Let's look at the last example as it has more interesting age/gender labels
# (compare to the last row of the dataframe above).
x, y = tuh[-1]

print(f"{x=}\n{y=}")


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
    on_last_window="overlap",
    mapping={"M": 0, "F": 1},  # map non-digit targets
)
# store the number of windows required for loading later on
tuh_windows.set_description({"n_windows": [len(d) for d in tuh_windows.datasets]})


###############################################################################
# Exploring Windows
# -----------------
#
# Iterating through the dataset gives ``x`` as an ndarray with shape
# ``(n_channels x 1000)``, ``y`` as ``[age, gender]``, and ``ind``.
# Let's look at the last example again.
x, y, ind = tuh_windows[-1]
print(f"{x=}\n{y=}\n{ind=}")


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
# When iterating through the DataLoader, we get ``batch_X`` as a tensor with shape
# ``(4 x n_channels x 1000)``, ``batch_y`` as ``[tensor([4 x age of subject]),
# tensor([4 x gender of subject])]``, and ``batch_ind``. To view the last example,
# simply iterate through the DataLoader:

for batch_X, batch_y, batch_ind in dl:
    pass

print(f"{batch_X=}\n{batch_y=}\n{batch_ind=}")


###############################################################################
# Numeric targets without a mapping
# ---------------------------------
#
# A mapping is unnecessary when the description already contains numeric
# targets. Here we encode the corpus's gender labels in a new description
# column, and reuse the same recordings with ``age`` and ``gender_code`` as
# targets. This leaves the original dataset and its description unchanged.
# The target order follows ``target_name``, not the description column order.

numeric_description = tuh.description.assign(
    gender_code=tuh.description["gender"].map({"M": 0, "F": 1})
)
numeric_tuh: BaseConcatDataset[RawDataset] = BaseConcatDataset(
    [
        RawDataset(ds.raw, description, target_name=("age", "gender_code"))
        for ds, (_, description) in zip(tuh.datasets, numeric_description.iterrows())
    ]
)
numeric_windows = create_fixed_length_windows(
    numeric_tuh,
    window_size_samples=1000,
    window_stride_samples=1000,
    on_last_window="overlap",
)
print(f"Target without a mapping: {numeric_windows[0][1]}")

###############################################################################
# Choosing the batch target layout
# --------------------------------
#
# Numeric list targets still collate into one tensor per target, as above.
# For a model expecting a single ``(batch_size, n_targets)`` tensor, convert
# each target list to a NumPy array before collation. This is useful, for
# example, with multiple continuous regression targets; choose the target
# dtype and loss to match the task. Encoding gender as a number does not make
# it a continuous regression target.

numeric_windows.target_transform = np.asarray
_, matrix_y, _ = next(iter(DataLoader(numeric_windows, batch_size=4)))
print(f"Matrix target shape: {tuple(matrix_y.shape)}")
print(matrix_y)
assert matrix_y.shape == (4, 2)

###############################################################################
# Changing the prediction target
# -------------------------------
#
# Use :meth:`~braindecode.datasets.BaseConcatDataset.set_target` to switch to a
# different target without loading the recordings or creating the windows again.
# For example, select the numeric class label we added to the description.
# Clear the array transform from the multi-target example so indexing returns a
# scalar target again.

numeric_windows.target_transform = None
numeric_windows.set_target("gender_code")
print(f"Classification target: {numeric_windows[-1][1]}")

###############################################################################
# The same windows can instead return age as a regression target. Selecting a
# column does not choose a model, loss, or dtype: here ``float`` converts the
# returned target to a floating-point value. This transform does not change the
# stored metadata.

numeric_windows.set_target("age")
numeric_windows.target_transform = float
print(f"Regression target: {numeric_windows[-1][1]}")

###############################################################################
# Recording descriptions and window metadata
# -------------------------------------------
#
# ``description`` has one row per recording; ``get_metadata()`` has one row per
# window, with recording descriptions repeated alongside the window information.
# ``set_target`` first looks for the column in each window dataset's metadata,
# then in its description. A description value such as age is broadcast to all
# windows of that recording. Editing the DataFrame returned by ``get_metadata()``
# alone does not update the targets returned by dataset indexing.

description = numeric_windows.description
print(description[["age", "gender_code"]])
metadata = numeric_windows.get_metadata()
print(metadata[["age", "gender_code", "target", "i_window_in_trial"]])

###############################################################################
# Selecting a subset of recordings
# ---------------------------------
#
# Keep only recordings with ``gender_code == 1``, independently of the current
# prediction target. To keep several values, use ``isin`` instead of ``eq``.
# The row positions passed to ``split`` refer to recordings, not individual
# windows. This selection keeps all windows from each selected recording; it
# does not filter event classes that vary within a recording.
# See :ref:`split-dataset-example` for other recording-level splits.

selected_ids = description.index[description["gender_code"].eq(1)].tolist()
selected_windows = numeric_windows.split({"selected": selected_ids})["selected"]
print(selected_windows.description[["age", "gender_code"]])
print(f"Selected {len(selected_windows)} of {len(numeric_windows)} windows")

###############################################################################
# ``split`` reuses the underlying datasets rather than copying the signals.
# Later calls to ``set_target`` also affect splits sharing those datasets. Here
# we finish changing targets before constructing the subset.

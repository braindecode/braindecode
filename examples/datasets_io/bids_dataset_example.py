""".. _bids-dataset-example:

BIDS Dataset Example
========================

In this example, we show how to fetch and prepare a BIDS dataset for usage
with Braindecode.
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

from pathlib import Path

import openneuro

from braindecode.datasets import BIDSDataset

###############################################################################
# First, we download a collection of (fake/empty) BIDS datasets.

# import tempfile
# data_dir = tempfile.mkdtemp()
data_dir = Path("~/mne_data/openneuro/").expanduser()
dataset_name = "ds004745"  # 200Mb dataset
dataset_root = data_dir / dataset_name

if not dataset_root.exists():
    openneuro.download(dataset=dataset_name, target_dir=dataset_root)

###############################################################################
# Now, loading the dataset is simply a one-line command:
bids_ds = BIDSDataset(dataset_root)

###############################################################################
# And we can see that the events of this dataset are set in the ``.annotations`` attribute of the raw data:
print(bids_ds.datasets[0].raw.annotations)

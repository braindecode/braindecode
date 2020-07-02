"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os

import mne
import pandas as pd

from ..datasets.base import BaseConcatDataset, WindowsDataset


def store_windows_dataset(path, dataset, overwrite):
    """Store a braindecode WindowsDatasets to files

    Parameters
    ----------
    path: str
        directory to which .fif and .json files are stored
    dataset: BaseConcatDataset of WindowsDatasets
    overwrite: bool
        whether to overwrite existing files
    """
    for ds_i, ds in enumerate(dataset.datasets):
        ds.windows.save(os.path.join(path, "{}-epo.fif".format(ds_i)),
                        overwrite=overwrite)
    dataset.description.to_json(os.path.join(path, "description.json"))


def recover_windows_dataset(path, ids_to_load=None):
    """Recover a stored windows dataset from files

    Parameters
    ----------
    path: str
        path to the directory of the .fif and .json files
    ids_to_load: None | list(int)
        ids of specific windows datasets to load

    Returns
    -------
    windows_datasets: BaseConcatDataset of WindowsDataset
    """
    epochs, description = _load_epochs_and_description(path, ids_to_load)
    datasets = []
    for windows_i, windows in enumerate(epochs):
        datasets.append(WindowsDataset(windows, description.iloc[windows_i]))
    return BaseConcatDataset(datasets)


def _load_epochs_and_description(path, ids_to_load=None):
    """Load mne.Epochs and their description from files

    Parameters
    ----------
    path: str
        directory from which .fif and .json files are loaded
    ids_to_load: None | list(int)
        ids of specific windows datasets to load

    Returns
    -------
    epochs, description: mne.Epochs, pandas.DataFrame
    """
    all_mne_epochs = []
    fif_file_name = "{}-epo.fif"
    description_df = pd.read_json(os.path.join(path, "description.json"))
    if ids_to_load is None:
        i = 0
        fif_file = os.path.join(path, fif_file_name.format(i))
        while os.path.exists(fif_file):
            all_mne_epochs.append(mne.read_epochs(fif_file))
            i += 1
            fif_file = os.path.join(path, fif_file_name.format(i))
    else:
        for i in ids_to_load:
            fif_file = os.path.join(path, fif_file_name.format(i))
            all_mne_epochs.append(mne.read_epochs(fif_file))
        description_df = description_df.iloc[ids_to_load]
    return all_mne_epochs, description_df

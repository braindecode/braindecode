"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os

import mne
import pandas as pd

from ..datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset


def save_concat_dataset(path, concat_dataset, concat_of_raws, overwrite):
    """Save a BaseConcatDataset of BaseDatasets or WindowsDatasets to files

    Parameters
    ----------
    path: str
        directory to which .fif and .json files are stored
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
        to save to files
    concat_of_raws: bool
        if true assumes that concat_dataset contains raws, if false assumes
        that concat_dataset contains epochs
    overwrite: bool
        whether to overwrite existing files
    """
    file_name = "{}-raw.fif" if concat_of_raws else "{}-epo.fif"
    for ds_i, ds in enumerate(concat_dataset.datasets):
        if concat_of_raws:
            ds.raw.save(os.path.join(path, file_name.format(ds_i)),
                        overwrite=overwrite)
        else:
            ds.windows.save(os.path.join(path, file_name.format(ds_i)),
                            overwrite=overwrite)
    concat_dataset.description.to_json(os.path.join(path, "description.json"))


def load_concat_dataset(path, preload, concat_of_raws, ids_to_load=None):
    """Load a stored BaseConcatDataset of BaseDatasets or WindowsDatasets from
    files

    Parameters
    ----------
    path: str
        path to the directory of the .fif and .json files
    preload: bool
        whether to preload the data
    concat_of_raws: bool
        if true assumes that stored data are raws, if false assumes
        that stored data are epochs
    ids_to_load: None | list(int)
        ids of specific signals to load

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    datasets = []
    if concat_of_raws:
        all_raws, description = _load_signals_and_description(
            path=path, preload=preload, raws=True, ids_to_load=ids_to_load)
        for raw_i, raw in enumerate(all_raws):
            datasets.append(BaseDataset(raw, description.iloc[raw_i]))
    else:
        all_epochs, description = _load_signals_and_description(
            path=path, preload=preload, raws=False, ids_to_load=ids_to_load)
        for epochs_i, epochs in enumerate(all_epochs):
            datasets.append(WindowsDataset(epochs, description.iloc[epochs_i]))
    return BaseConcatDataset(datasets)


def _load_signals_and_description(path, preload, raws, ids_to_load=None):
    all_signals = []
    file_name = "{}-raw.fif" if raws else "{}-epo.fif"
    description_df = pd.read_json(os.path.join(path, "description.json"))
    if ids_to_load is None:
        i = 0
        fif_file = os.path.join(path, file_name.format(i))
        while os.path.exists(fif_file):
            all_signals.append(_load_signals(fif_file, preload, raws))
            i += 1
            fif_file = os.path.join(path, file_name.format(i))
    else:
        for i in ids_to_load:
            fif_file = os.path.join(path, file_name.format(i))
            all_signals.append(_load_signals(fif_file, preload, raws))
        description_df = description_df.iloc[ids_to_load]
    return all_signals, description_df


def _load_signals(fif_file, preload, raws):
    if raws:
        signals = mne.io.read_raw_fif(fif_file, preload=preload)
    else:
        signals = mne.read_epochs(fif_file, preload=preload)
    return signals

"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import json
import os
from glob import glob

import mne
import pandas as pd

from ..datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset


def save_concat_dataset(path, concat_dataset, overwrite=False):
    """Save a BaseConcatDataset of BaseDatasets or WindowsDatasets to files

    Parameters
    ----------
    path: str
        directory to which .fif and .json files are stored
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
        to save to files
    overwrite: bool
        whether to overwrite existing files (will delete old fif files in
        specified directory)
    """
    assert len(concat_dataset.datasets) > 0, "Expect at least one dataset"
    assert (hasattr(concat_dataset.datasets[0], 'raw') + hasattr(
        concat_dataset.datasets[0], 'windows') == 1), (
        "dataset should have either raw or windows attribute")
    concat_of_raws = hasattr(concat_dataset.datasets[0], 'raw')
    file_name = "{}-raw.fif" if concat_of_raws else "{}-epo.fif"
    description_file_name = os.path.join(path, 'description.json')
    target_file_name = os.path.join(path, 'target_name.json')
    if overwrite:
        file_names = glob(os.path.join(path, f"*{file_name.lstrip('{}')}"))
        _ = [os.remove(f) for f in file_names]
        if os.path.isfile(target_file_name):
            os.remove(target_file_name)
        if os.path.isfile(description_file_name):
            os.remove(description_file_name)

    if concat_of_raws:
        # for checks that all have same target name and for
        # saving later
        target_name = concat_dataset.datasets[0].target_name
    for i_ds, ds in enumerate(concat_dataset.datasets):
        full_file_path = os.path.join(path, file_name.format(i_ds))
        if concat_of_raws:
            ds.raw.save(full_file_path, overwrite=overwrite)
            assert ds.target_name == target_name, (
                "All datasets should have same target name")
        else:
            ds.windows.save(full_file_path, overwrite=overwrite)

    if concat_of_raws:
        json.dump({'target_name': target_name}, open(target_file_name, 'w'))
    concat_dataset.description.to_json(description_file_name)


def load_concat_dataset(path, preload, ids_to_load=None, target_name=None):
    """Load a stored BaseConcatDataset of BaseDatasets or WindowsDatasets from
    files

    Parameters
    ----------
    path: str
        path to the directory of the .fif and .json files
    preload: bool
        whether to preload the data
    ids_to_load: None | list(int)
        ids of specific signals to load
    target_name: None or str
        Load specific column as target. If not given, take saved target name.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    assert ((os.path.isfile(os.path.join(path, '0-raw.fif')) +
             os.path.isfile(os.path.join(path, '0-epo.fif'))) == 1), (
        "Expect either raw or epo to exist inside the directory")
    concat_of_raws = os.path.isfile(os.path.join(path, '0-raw.fif'))

    if concat_of_raws and target_name is None:
        target_file_name = os.path.join(path, 'target_name.json')
        target_name = json.load(open(target_file_name, "r"))['target_name']

    all_signals, description = _load_signals_and_description(
        path=path, preload=preload, raws=concat_of_raws, ids_to_load=ids_to_load
    )
    datasets = []
    for i_signal, signal in enumerate(all_signals):
        if concat_of_raws:
            datasets.append(BaseDataset(signal, description.iloc[i_signal],
                                        target_name=target_name))
        else:
            datasets.append(WindowsDataset(signal, description.iloc[i_signal]))
    return BaseConcatDataset(datasets)


def _load_signals_and_description(path, preload, raws, ids_to_load=None):
    all_signals = []
    file_name = "{}-raw.fif" if raws else "{}-epo.fif"
    description_df = pd.read_json(os.path.join(path, "description.json"))
    if ids_to_load is None:
        file_names = glob(os.path.join(path, f"*{file_name.lstrip('{}')}"))
        # Extract ids, e.g.,
        # '/home/schirrmr/data/preproced-tuh/all-sensors/11-raw.fif' ->
        # '11-raw.fif' -> 11
        ids_to_load = sorted(
            [int(os.path.split(f)[-1].split('-')[0]) for f in file_names])
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


def load_concat_datasets(path, preload, ids_to_load=None, target_name=None):
    """Load a series of stored BaseConcatDataset of BaseDatasets or
    WindowsDatasets from files that were stored to individual subdirectories
    that are ascendingly numerated. This is for datasets, that do not fit
    entirely into RAM, s.t. each recording has to be individually preprocessed
    and stored.

    Parameters
    ----------
    path: str
        path to the directory of the .fif and .json files
    preload: bool
        whether to preload the data
    ids_to_load: None | list(int)
        ids of specific signals to load
    target_name: None or str
        Load specific column as target. If not given, take saved target name.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    path = os.path.join(path, '*', '')
    paths = glob(path)
    paths = sorted(paths, key=lambda p: int(p.split('/')[-2]))
    if ids_to_load is not None:
        paths = [paths[i] for i in ids_to_load]
    all_concat_ds = []
    for path in paths:
        concat_ds = load_concat_dataset(
            os.path.join(path), preload=preload, target_name=target_name
        )
        all_concat_ds.append(concat_ds)
    return BaseConcatDataset(all_concat_ds)

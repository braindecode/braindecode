"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import json
import os
from glob import glob
import warnings

import mne
import pandas as pd

from braindecode.datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset


def save_concat_dataset(path, concat_dataset, overwrite=False):
    warnings.warn('"save_concat_dataset()" is deprecated and will be removed in the future. '
                  'Use dataset.save() instead.')
    concat_dataset.save(path=path, overwrite=overwrite)


def load_concat_dataset(path, preload, ids_to_load=None, target_name=None):
    """Load a stored BaseConcatDataset of BaseDatasets or WindowsDatasets from
    files.

    Parameters
    ----------
    path: str
        Path to the directory of the .fif / -epo.fif and .json files.
    preload: bool
        Whether to preload the data.
    ids_to_load: None | list(int)
        Ids of specific files to load.
    target_name: None or str
        Load specific description column as target. If not given, take saved target name.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    assert ((os.path.isfile(os.path.join(path, '0-raw.fif')) +
             os.path.isfile(os.path.join(path, '0-epo.fif'))) == 1), (
        "Expect either raw or epo to exist inside the directory")
    concat_of_raws = os.path.isfile(os.path.join(path, '0-raw.fif'))
    assert not (not concat_of_raws and target_name is not None), (
        'Setting a new target is only supported for raws.')

    if concat_of_raws and target_name is None:
        target_file_name = os.path.join(path, 'target_name.json')
        target_name = json.load(open(target_file_name, "r"))['target_name']

    all_signals, description = _load_signals_and_description(
        path=path, preload=preload, raws=concat_of_raws, ids_to_load=ids_to_load)
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

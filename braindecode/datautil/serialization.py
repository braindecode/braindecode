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
from joblib import Parallel, delayed

from ..datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset


def save_concat_dataset(path, concat_dataset, overwrite=False):
    warnings.warn('"save_concat_dataset()" is deprecated and will be removed in the future. '
                  'Use dataset.save() instead.')
    concat_dataset.save(path=path, overwrite=overwrite)


def _load_concat_dataset(path, preload, ids_to_load=None, target_name=None,):
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
    # assume we have a single concat dataset to load
    concat_of_raws = os.path.isfile(os.path.join(path, '0-raw.fif'))
    assert not (not concat_of_raws and target_name is not None), (
        'Setting a new target is only supported for raws.')
    concat_of_epochs = os.path.isfile(os.path.join(path, '0-epo.fif'))
    paths = [path]
    # assume we have multiple concat datasets to load
    if not (concat_of_raws or concat_of_epochs):
        concat_of_raws = os.path.isfile(os.path.join(path, '0', '0-raw.fif'))
        concat_of_epochs = os.path.isfile(os.path.join(path, '0', '0-epo.fif'))
        path = os.path.join(path, '*', '')
        paths = glob(path)
        paths = sorted(paths, key=lambda p: int(p.split(os.sep)[-2]))
        if ids_to_load is not None:
            paths = [paths[i] for i in ids_to_load]
        ids_to_load = None
    # if we have neither a single nor multiple datasets, something went wrong
    assert concat_of_raws or concat_of_epochs, (
        f'Expect either raw or epo to exist in {path} or in '
        f'{os.path.join(path, "0")}')

    datasets = []
    for path in paths:
        if concat_of_raws and target_name is None:
            target_file_name = os.path.join(path, 'target_name.json')
            target_name = json.load(open(target_file_name, "r"))['target_name']

        all_signals, description = _load_signals_and_description(
            path=path, preload=preload, raws=concat_of_raws,
            ids_to_load=ids_to_load
        )
        for i_signal, signal in enumerate(all_signals):
            if concat_of_raws:
                datasets.append(
                    BaseDataset(signal, description.iloc[i_signal],
                                target_name=target_name))
            else:
                datasets.append(
                    WindowsDataset(signal, description.iloc[i_signal])
                )
    concat_ds = BaseConcatDataset(datasets)
    for kwarg_name in ['raw_preproc_kwargs', 'window_kwargs', 'window_preproc_kwargs']:
        kwarg_path = os.path.join(path, '.'.join([kwarg_name, 'json']))
        if os.path.exists(kwarg_path):
            kwargs = json.load(open(kwarg_path, 'r'))
            kwargs = [tuple(kwarg) for kwarg in kwargs]
            setattr(concat_ds, kwarg_name, kwargs)
    return concat_ds


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


def load_concat_dataset(path, preload, ids_to_load=None, target_name=None,
                        n_jobs=-1):
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
        Load specific description column as target. If not given, take saved
        target name.
    n_jobs: int
        Number of jobs to be used to read files in parallel.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    # if we encounter a dataset that was saved in 'the old way', call the
    # corresponding 'old' loading function
    if _is_outdated_saved(path):
        warnings.warn("The way your dataset was saved is deprecated by now. "
                      "Please save it again using '.save()'.")
        return _load_concat_dataset(
            path=path, preload=preload, ids_to_load=ids_to_load,
            target_name=target_name)

    # else we have a dataset saved in the new way with
    # - subdirectories in path for every dataset
    # - description.json and -epo.fif or -raw.fif in every subdirectory
    # - target_name.json in path if we were given raws
    # grab all the fif and json files, numbers have to match
    fif_files = glob(os.path.join(path, '**/*.fif'))
    description_files = glob(os.path.join(path, '**/description.json'))
    assert len(fif_files) == len(description_files)
    # sort files by id of the datasets
    fif_files = sorted(fif_files, key=lambda p: int(p.split(os.sep)[-2]))
    description_files = sorted(
        description_files, key=lambda p: int(p.split(os.sep)[-2]))
    # optionally make a selection of files
    if ids_to_load is not None:
        fif_files = [fif_files[i] for i in ids_to_load]
        description_files = [description_files[i] for i in ids_to_load]
    raws = fif_files[0].endswith('-raw.fif')
    # Parallelization of mne.read_epochs with preload=False fails with
    # 'TypeError: cannot pickle '_io.BufferedReader' object'.
    # So ignore n_jobs in that case and load with a single job.
    if not raws and n_jobs != 1:
        warnings.warn(
            'Parallelized reading with `preload=False` is not supported for '
            'windowed data. Will use `n_jobs=1`.')
        n_jobs = 1
    datasets = Parallel(n_jobs)(
        delayed(_load_parallel)(
            fif_files[i], description_files[i], path, preload, raws)
        for i in range(len(fif_files)))
    return BaseConcatDataset(datasets)


def _load_parallel(fif_file, description_file, path, preload, raws):
    signals = _load_signals(fif_file, preload, raws)
    description = pd.read_json(description_file, typ='series')
    if raws:
        target_name = None
        target_file = os.path.join(path, 'target_name.json')
        if os.path.exists(target_file):
            target_name = json.load(open(target_file, "r"))['target_name']
            # target_name = pd.read_json(target_file, typ='series')
        dataset = BaseDataset(signals, description, target_name)
    else:
        dataset = WindowsDataset(signals, description)
    return dataset


def _is_outdated_saved(path):
    """Data was saved in the old way if there are 'description.json', '-raw.fif'
    or '-epo.fif' in path OR if there are more 'fif' files than
    'description.json' files."""
    multiple2 = False
    i = 0
    while True:
        this_dir = os.path.join(path, str(i))
        if os.path.exists(this_dir):
            # in the new way of saving, there exist exactly two files per
            # subdirectory: -epo.fif / -raw.fif AND description.json
            if len(os.listdir(this_dir)) > 2:
                multiple2 = True
                break
        else:
            break
        i += 1
    description_files = glob(os.path.join(path, '**/description.json'))
    fif_files = glob(os.path.join(path, '**/*.fif'))
    multiple = len(description_files) != len(fif_files)
    return (os.path.exists(os.path.join(path, 'description.json')) or
            os.path.exists(os.path.join(path, '0-raw.fif')) or
            os.path.exists(os.path.join(path, '0-epo.fif')) or
            multiple or multiple2)

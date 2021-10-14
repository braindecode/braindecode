"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os
import json
import warnings
from glob import glob

import mne
import pandas as pd
from joblib import Parallel, delayed

from ..datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset


def save_concat_dataset(path, concat_dataset, overwrite=False):
    warnings.warn('"save_concat_dataset()" is deprecated and will be removed in'
                  ' the future. Use dataset.save() instead.')
    concat_dataset.save(path=path, overwrite=overwrite)


def _outdated_load_concat_dataset(path, preload, ids_to_load=None,
                                  target_name=None):
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

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    # assume we have a single concat dataset to load
    is_raw = os.path.isfile(os.path.join(path, '0-raw.fif'))
    assert not (not is_raw and target_name is not None), (
        'Setting a new target is only supported for raws.')
    is_epochs = os.path.isfile(os.path.join(path, '0-epo.fif'))
    paths = [path]
    # assume we have multiple concat datasets to load
    if not (is_raw or is_epochs):
        is_raw = os.path.isfile(os.path.join(path, '0', '0-raw.fif'))
        is_epochs = os.path.isfile(os.path.join(path, '0', '0-epo.fif'))
        path = os.path.join(path, '*', '')
        paths = glob(path)
        paths = sorted(paths, key=lambda p: int(p.split(os.sep)[-2]))
        if ids_to_load is not None:
            paths = [paths[i] for i in ids_to_load]
        ids_to_load = None
    # if we have neither a single nor multiple datasets, something went wrong
    assert is_raw or is_epochs, (
        f'Expect either raw or epo to exist in {path} or in '
        f'{os.path.join(path, "0")}')

    datasets = []
    for path in paths:
        if is_raw and target_name is None:
            target_file_name = os.path.join(path, 'target_name.json')
            target_name = json.load(open(target_file_name, "r"))['target_name']

        all_signals, description = _load_signals_and_description(
            path=path, preload=preload, is_raw=is_raw,
            ids_to_load=ids_to_load
        )
        for i_signal, signal in enumerate(all_signals):
            if is_raw:
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


def _load_signals_and_description(path, preload, is_raw, ids_to_load=None):
    all_signals = []
    file_name = "{}-raw.fif" if is_raw else "{}-epo.fif"
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
        all_signals.append(_load_signals(fif_file, preload, is_raw))
    description_df = description_df.iloc[ids_to_load]
    return all_signals, description_df


def _load_signals(fif_file, preload, is_raw):
    if is_raw:
        signals = mne.io.read_raw_fif(fif_file, preload=preload)
    elif fif_file.endswith('-epo.fif'):
        signals = mne.read_epochs(fif_file, preload=preload)
    else:
        raise ValueError('fif_file must end with raw.fif or epo.fif.')
    return signals


def load_concat_dataset(path, preload, ids_to_load=None, target_name=None,
                        n_jobs=1):
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
                      "Please save it again using dataset.save().", UserWarning)
        return _outdated_load_concat_dataset(
            path=path, preload=preload, ids_to_load=ids_to_load,
            target_name=target_name)

    # else we have a dataset saved in the new way with subdirectories in path
    # for every dataset with description.json and -epo.fif or -raw.fif,
    # target_name.json, raw_preproc_kwargs.json, window_kwargs.json,
    # window_preproc_kwargs.json
    if ids_to_load is None:
        ids_to_load = [os.path.split(p)[-1] for p in os.listdir(path)]
        ids_to_load = sorted(ids_to_load, key=lambda i: int(i))
    ids_to_load = [str(i) for i in ids_to_load]
    first_raw_fif_path = os.path.join(
        path, ids_to_load[0], f'{ids_to_load[0]}-raw.fif')
    is_raw = os.path.exists(first_raw_fif_path)
    # Parallelization of mne.read_epochs with preload=False fails with
    # 'TypeError: cannot pickle '_io.BufferedReader' object'.
    # So ignore n_jobs in that case and load with a single job.
    if not is_raw and n_jobs != 1:
        warnings.warn(
            'Parallelized reading with `preload=False` is not supported for '
            'windowed data. Will use `n_jobs=1`.', UserWarning)
        n_jobs = 1
    datasets = Parallel(n_jobs)(
        delayed(_load_parallel)(path, i, preload, is_raw)
        for i in ids_to_load)
    return BaseConcatDataset(datasets)


def _load_parallel(path, i, preload, is_raw):
    sub_dir = os.path.join(path, i)
    file_name_patterns = ['{}-raw.fif', '{}-epo.fif']
    if all([os.path.exists(os.path.join(sub_dir, p.format(i))) for p in file_name_patterns]):
        raise FileExistsError('Found -raw.fif and -epo.fif in directory.')
    fif_name_pattern = file_name_patterns[0] if is_raw else file_name_patterns[1]
    fif_file_name = fif_name_pattern.format(i)
    fif_file_path = os.path.join(sub_dir, fif_file_name)
    signals = _load_signals(fif_file_path, preload, is_raw)
    description_file_path = os.path.join(sub_dir, 'description.json')
    description = pd.read_json(description_file_path, typ='series')
    target_file_path = os.path.join(sub_dir, 'target_name.json')
    target_name = None
    if os.path.exists(target_file_path):
        target_name = json.load(open(target_file_path, "r"))['target_name']

    if is_raw:
        dataset = BaseDataset(signals, description, target_name)
    else:
        window_kwargs = _load_kwargs_json('window_kwargs', sub_dir)
        windows_ds_kwargs = [kwargs[1] for kwargs in window_kwargs if kwargs[0] == 'WindowsDataset']
        windows_ds_kwargs = windows_ds_kwargs[0] if len(windows_ds_kwargs) == 1 else {}
        dataset = WindowsDataset(signals, description,
                                 targets_from=windows_ds_kwargs.get('targets_from', 'metadata'),
                                 last_target_only=windows_ds_kwargs.get('last_target_only', True)
                                 )
        setattr(dataset, 'window_kwargs', window_kwargs)
    for kwargs_name in ['raw_preproc_kwargs', 'window_preproc_kwargs']:
        kwargs = _load_kwargs_json(kwargs_name, sub_dir)
        setattr(dataset, kwargs_name, kwargs)
    return dataset


def _load_kwargs_json(kwargs_name, sub_dir):
    kwargs_file_name = '.'.join([kwargs_name, 'json'])
    kwargs_file_path = os.path.join(sub_dir, kwargs_file_name)
    if os.path.exists(kwargs_file_path):
        kwargs = json.load(open(kwargs_file_path, 'r'))
        kwargs = [tuple(kwarg) for kwarg in kwargs]
        return kwargs


def _is_outdated_saved(path):
    """Data was saved in the old way if there are 'description.json', '-raw.fif'
    or '-epo.fif' in path (no subdirectories) OR if there are more 'fif' files
    than 'description.json' files."""
    description_files = glob(os.path.join(path, '**/description.json'))
    fif_files = glob(os.path.join(path, '**/*-raw.fif')) + glob(os.path.join(path, '**/*-epo.fif'))
    multiple = len(description_files) != len(fif_files)
    kwargs_in_path = any(
        [os.path.exists(os.path.join(path, kwarg_name))
         for kwarg_name in ['raw_preproc_kwargs', 'window_kwargs',
                            'window_preproc_kwargs']])
    return (os.path.exists(os.path.join(path, 'description.json')) or
            os.path.exists(os.path.join(path, '0-raw.fif')) or
            os.path.exists(os.path.join(path, '0-epo.fif')) or
            multiple or
            kwargs_in_path)


def _check_save_dir_empty(save_dir):
    """Make sure a BaseConcatDataset can be saved under a given directory.

    Parameters
    ----------
    save_dir : str
        Directory under which a `BaseConcatDataset` will be saved.

    Raises
    -------
    FileExistsError
        If ``save_dir`` is not a valid directory for saving.
    """
    sub_dirs = [os.path.basename(s).isdigit()
                for s in glob(os.path.join(save_dir, '*'))]
    if any(sub_dirs):
        raise FileExistsError(
            f'Directory {save_dir} already contains subdirectories. Please '
            'select a different directory, set overwrite=True, or resolve '
            'manually.')

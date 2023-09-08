"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os
import json
import pickle
import warnings
from glob import glob
from pathlib import Path

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
    path: pathlib.Path
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
    is_raw = (path / '0-raw.fif').is_file()
    assert not (not is_raw and target_name is not None), (
        'Setting a new target is only supported for raws.')
    is_epochs = (path / '0-epo.fif').is_file()
    paths = [path]
    # assume we have multiple concat datasets to load
    if not (is_raw or is_epochs):
        is_raw = (path / '0' / '0-raw.fif').is_file()
        is_epochs = (path / '0' / '0-epo.fif').is_file()
        paths = path.glob("*/")
        paths = sorted(paths, key=lambda p: int(p.name))
        if ids_to_load is not None:
            paths = [paths[i] for i in ids_to_load]
        ids_to_load = None
    # if we have neither a single nor multiple datasets, something went wrong
    assert is_raw or is_epochs, (
        f'Expect either raw or epo to exist in {path} or in '
        f'{path / "0"}')

    datasets = []
    for path in paths:
        if is_raw and target_name is None:
            target_file_name = path / 'target_name.json'
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
        kwarg_path = path / '.'.join([kwarg_name, 'json'])
        if kwarg_path.exists():
            with open(kwarg_path, 'r') as f:
                kwargs = json.load(f)
            kwargs = [tuple(kwarg) for kwarg in kwargs]
            setattr(concat_ds, kwarg_name, kwargs)
    return concat_ds


def _load_signals_and_description(path, preload, is_raw, ids_to_load=None):
    all_signals = []
    file_name = "{}-raw.fif" if is_raw else "{}-epo.fif"
    description_df = pd.read_json(path / "description.json")
    if ids_to_load is None:
        file_names = path.glob(f"*{file_name.lstrip('{}')}")
        # Extract ids, e.g.,
        # '/home/schirrmr/data/preproced-tuh/all-sensors/11-raw.fif' ->
        # '11-raw.fif' -> 11
        ids_to_load = sorted(
            [int(os.path.split(f)[-1].split('-')[0]) for f in file_names])
    for i in ids_to_load:
        fif_file = path / file_name.format(i)
        all_signals.append(_load_signals(fif_file, preload, is_raw))
    description_df = description_df.iloc[ids_to_load]
    return all_signals, description_df


def _load_signals(fif_file, preload, is_raw):

    # Reading the raw file from pickle if it has been save before.
    # The pickle file only contain the raw object without the data.
    pkl_file = fif_file.with_suffix(".pkl")
    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            signals = pickle.load(f)

        # If the file has been moved together with the pickle file, make sure
        # the path links to correct fif file.
        signals._fname = str(fif_file)
        if preload:
            signals.load_data()
        return signals

    if is_raw:
        signals = mne.io.read_raw_fif(fif_file, preload=preload)
    elif fif_file.name.endswith('-epo.fif'):
        signals = mne.read_epochs(fif_file, preload=preload)
    else:
        raise ValueError('fif_file must end with raw.fif or epo.fif.')

    # Only do this for raw objects. Epoch objects are not picklable as they
    # hold references to open files in `signals._raw[0].fid`.
    if is_raw:
        # Saving the raw file without data into a pickle file, so it can be
        # retrieved faster on the next use of this dataset.
        with open(pkl_file, "wb") as f:
            if preload:
                data = signals._data
                signals._data, signals.preload = None, False
            pickle.dump(signals, f)
            if preload:
                signals._data, signals.preload = data, True

    return signals


def load_concat_dataset(path, preload, ids_to_load=None, target_name=None,
                        n_jobs=1):
    """Load a stored BaseConcatDataset of BaseDatasets or WindowsDatasets from
    files.

    Parameters
    ----------
    path: str | pathlib.Path
        Path to the directory of the .fif / -epo.fif and .json files.
    preload: bool
        Whether to preload the data.
    ids_to_load: list of int | None
        Ids of specific files to load.
    target_name: str | list | None
        Load specific description column as target. If not given, take saved
        target name.
    n_jobs: int
        Number of jobs to be used to read files in parallel.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    # Make sure we always work with a pathlib.Path
    path = Path(path)

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
        ids_to_load = [p.name for p in path.iterdir()]
        ids_to_load = sorted(ids_to_load, key=lambda i: int(i))
    ids_to_load = [str(i) for i in ids_to_load]
    first_raw_fif_path = path / ids_to_load[0] / f'{ids_to_load[0]}-raw.fif'
    is_raw = first_raw_fif_path.exists()

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
        for i in ids_to_load
    )
    return BaseConcatDataset(datasets)


def _load_parallel(path, i, preload, is_raw):
    sub_dir = path / i
    file_name_patterns = ['{}-raw.fif', '{}-epo.fif']
    if all([(sub_dir / p.format(i)).exists() for p in file_name_patterns]):
        raise FileExistsError('Found -raw.fif and -epo.fif in directory.')

    fif_name_pattern = file_name_patterns[0] if is_raw else file_name_patterns[1]
    fif_file_name = fif_name_pattern.format(i)
    fif_file_path = sub_dir / fif_file_name

    signals = _load_signals(fif_file_path, preload, is_raw)

    description_file_path = sub_dir / 'description.json'
    description = pd.read_json(description_file_path, typ='series')

    target_file_path = sub_dir / 'target_name.json'
    target_name = None
    if target_file_path.exists():
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

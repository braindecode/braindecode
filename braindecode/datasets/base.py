"""
Dataset classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import os
import json
from typing import Iterable
import warnings
from glob import glob

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, ConcatDataset


def _create_description(description):
    if description is not None:
        if (not isinstance(description, pd.Series) and
                not isinstance(description, dict)):
            raise ValueError(f"'{description}' has to be either a "
                             f"pandas.Series or a dict.")
        if isinstance(description, dict):
            description = pd.Series(description)
    return description


class BaseDataset(Dataset):
    """Returns samples from an mne.io.Raw object along with a target.

    Dataset which serves samples from an mne.io.Raw object along with a target.
    The target is unique for the dataset, and is obtained through the
    `description` attribute.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous data.
    description : dict | pandas.Series | None
        Holds additional description about the continuous signal / subject.
    target_name : str | tuple | None
        Name(s) of the index in `description` that should be used to provide the
        target (e.g., to be used in a prediction task later on).
    transform : callable | None
        On-the-fly transform applied to the example before it is returned.
    """
    def __init__(self, raw, description=None, target_name=None,
                 transform=None):
        self.raw = raw
        self._description = _create_description(description)
        self.transform = transform

        # save target name for load/save later
        self.target_name = self._target_name(target_name)

    def __getitem__(self, index):
        X = self.raw[:, index][0]
        y = None
        if self.target_name is not None:
            y = self.description[self.target_name]
        if isinstance(y, pd.Series):
            y = y.to_list()
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.raw)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if value is not None and not callable(value):
            raise ValueError('Transform needs to be a callable.')
        self._transform = value

    @property
    def description(self):
        return self._description

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.Series
            Description in the form key: value.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = _create_description(description)
        for key, value in description.items():
            # if the key is already in the existing description, drop it
            if key in self._description:
                assert overwrite, (f"'{key}' already in description. Please "
                                   f"rename or set overwrite to True.")
                self._description.pop(key)
        self._description = pd.concat([self.description, description])

    def _target_name(self, target_name):
        if target_name is None:
            return target_name
        else:
            # convert tuple of names or single name to list
            if isinstance(target_name, tuple):
                target_name = [name for name in target_name]
            else:
                target_name = [target_name]
            # check if target name(s) can be read from description
            for name in target_name:
                if name not in self.description:
                    warnings.warn(f"'{name}' not in description. '__getitem__'"
                                  f"will fail unless an appropriate target is"
                                  f" added to description.", UserWarning)
        # return a list of str if there are multiple targets and a str otherwise
        return target_name if len(target_name) > 1 else target_name[0]


class WindowsDataset(BaseDataset):
    """Returns windows from an mne.Epochs object along with a target.

    Dataset which serves windows from an mne.Epochs object along with their
    target and additional information. The `metadata` attribute of the Epochs
    object must contain a column called `target`, which will be used to return
    the target that corresponds to a window. Additional columns
    `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial` are also
    required to serve information about the windowing (e.g., useful for cropped
    training).
    See `braindecode.datautil.windowers` to directly create a `WindowsDataset`
    from a `BaseDataset` object.

    Parameters
    ----------
    windows : mne.Epochs
        Windows obtained through the application of a windower to a BaseDataset
        (see `braindecode.datautil.windowers`).
    description : dict | pandas.Series | None
        Holds additional info about the windows.
    transform : callable | None
        On-the-fly transform applied to a window before it is returned.
    """
    def __init__(self, windows, description=None, transform=None):
        self.windows = windows
        self._description = _create_description(description)
        self.transform = transform

        self.y = self.windows.metadata.loc[:, 'target'].to_list()
        self.crop_inds = self.windows.metadata.loc[
            :, ['i_window_in_trial', 'i_start_in_trial',
                'i_stop_in_trial']].to_numpy()

    def __getitem__(self, index):
        """Get a window and its target.

        Parameters
        ----------
        index : int
            Index to the window (and target) to return.

        Returns
        -------
        np.ndarray
            Window of shape (n_channels, n_times).
        int
            Target for the windows.
        np.ndarray
            Crop indices.
        """
        X = self.windows.get_data(item=index)[0].astype('float32')
        if self.transform is not None:
            X = self.transform(X)
        y = self.y[index]
        # necessary to cast as list to get list of three tensors from batch,
        # otherwise get single 2d-tensor...
        crop_inds = self.crop_inds[index].tolist()

        return X, y, crop_inds

    def __len__(self):
        return len(self.windows.events)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if value is not None and not callable(value):
            raise ValueError('Transform needs to be a callable.')
        self._transform = value

    @property
    def description(self):
        return self._description

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.Series
            Description in the form key: value.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = _create_description(description)
        for key, value in description.items():
            # if they key is already in the existing description, drop it
            if key in self._description:
                assert overwrite, (f"'{key}' already in description. Please "
                                   f"rename or set overwrite to True.")
                self._description.pop(key)
        self._description = pd.concat([self.description, description])


class BaseConcatDataset(ConcatDataset):
    """A base class for concatenated datasets. Holds either mne.Raw or
    mne.Epoch in self.datasets and has a pandas DataFrame with additional
    description.

    Parameters
    ----------
    list_of_ds : list
        list of BaseDataset, BaseConcatDataset or WindowsDataset
    target_transform : callable | None
        Optional function to call on targets before returning them.
    """
    def __init__(self, list_of_ds, target_transform=None):
        # if we get a list of BaseConcatDataset, get all the individual datasets
        if list_of_ds and isinstance(list_of_ds[0], BaseConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)

        self.target_transform = target_transform

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        """
        if isinstance(idx, Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]
        return item

    def split(self, by=None, property=None, split_ids=None):
        """Split the dataset based on information listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        by : str | list
            If ``by`` is a string, splitting is performed based on the
            description DataFrame column with this name.
            If ``by`` is a (list of) list of integers, the position in the first
            list corresponds to the split id and the integers to the
            datapoints of that split.
        property : str
            Some property which is listed in info DataFrame.
        split_ids : list
            List of indices to be combined in a subset.
            It can be a list of int or a list of list of int.

        Returns
        -------
        splits : dict
            A dictionary with the name of the split (a string) as key and the
            dataset as value.
        """
        args_not_none = [
            by is not None, property is not None, split_ids is not None]
        if sum(args_not_none) != 1:
            raise ValueError("Splitting requires exactly one argument.")

        if property is not None or split_ids is not None:
            warnings.warn("Keyword arguments `property` and `split_ids` "
                          "are deprecated and will be removed in the future. "
                          "Use `by` instead.")
            by = property if property is not None else split_ids
        if isinstance(by, str):
            split_ids = {
                k: list(v)
                for k, v in self.description.groupby(by).groups.items()
            }
        else:
            # assume list(int)
            if not isinstance(by[0], list):
                by = [by]
            # assume list(list(int))
            split_ids = {split_i: split for split_i, split in enumerate(by)}

        return {str(split_name): BaseConcatDataset(
            [self.datasets[ds_ind] for ds_ind in ds_inds])
            for split_name, ds_inds in split_ids.items()}

    def get_metadata(self):
        """Concatenate the metadata and description of the wrapped Epochs.

        Returns
        -------
        metadata : pd.DataFrame
            DataFrame containing as many rows as there are windows in the
            BaseConcatDataset, with the metadata and description information
            for each window.
        """
        if not all([isinstance(ds, WindowsDataset) for ds in self.datasets]):
            raise TypeError('Metadata dataframe can only be computed when all '
                            'datasets are WindowsDataset.')

        all_dfs = list()
        for ds in self.datasets:
            df = ds.windows.metadata
            for k, v in ds.description.items():
                df[k] = v
            all_dfs.append(df)

        return pd.concat(all_dfs)

    @property
    def transform(self):
        return [ds.transform for ds in self.datasets]

    @transform.setter
    def transform(self, fn):
        for i in range(len(self.datasets)):
            self.datasets[i].transform = fn

    @property
    def target_transform(self):
        return self._target_transform

    @target_transform.setter
    def target_transform(self, fn):
        if not (callable(fn) or fn is None):
            raise TypeError('target_transform must be a callable.')
        self._target_transform = fn

    def _outdated_save(self, path, overwrite=False):
        """This is a copy of the old saving function, that had inconsistent
        functionality for BaseDataset and WindowsDataset. It only exists to
        assure backwards compatibility by still being able to run the old tests.

        Save dataset to files.

        Parameters
        ----------
        path : str
            Directory to which .fif / -epo.fif and .json files are stored.
        overwrite : bool
            Whether to delete old files (.json, .fif, -epo.fif) in specified
            directory prior to saving.
        """
        warnings.warn('This function only exists for backwards compatibility '
                      'purposes. DO NOT USE!', UserWarning)
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        if not (hasattr(self.datasets[0], 'raw') or hasattr(
                self.datasets[0], 'windows')):
            raise ValueError("dataset should have either raw or windows "
                             "attribute")
        file_names_ = ["{}-raw.fif", "{}-epo.fif"]
        description_file_name = os.path.join(path, 'description.json')
        target_file_name = os.path.join(path, 'target_name.json')
        if not overwrite:
            if (os.path.exists(description_file_name) or
                    os.path.exists(target_file_name)):
                raise FileExistsError(
                    f'{description_file_name} or {target_file_name} exist in '
                    f'{path}.')
        else:
            for file_name in file_names_:
                file_names = glob(os.path.join(
                    path, f"*{file_name.lstrip('{}')}"))
                _ = [os.remove(f) for f in file_names]
            if os.path.isfile(target_file_name):
                os.remove(target_file_name)
            if os.path.isfile(description_file_name):
                os.remove(description_file_name)
            for kwarg_name in ['raw_preproc_kwargs', 'window_kwargs', 'window_preproc_kwargs']:
                kwarg_path = os.path.join(path, '.'.join([kwarg_name, 'json']))
                if os.path.exists(kwarg_path):
                    os.remove(kwarg_path)

        is_raw = hasattr(self.datasets[0], 'raw')
        file_name = file_names_[0] if is_raw else file_names_[1]
        if is_raw:
            # for checks that all have same target name and for
            # saving later
            target_name = self.datasets[0].target_name
        for i_ds, ds in enumerate(self.datasets):
            full_file_path = os.path.join(path, file_name.format(i_ds))
            if is_raw:
                ds.raw.save(full_file_path, overwrite=overwrite)
                assert ds.target_name == target_name, (
                    "All datasets should have same target name")
            else:
                ds.windows.save(full_file_path, overwrite=overwrite)

        if is_raw:
            json.dump({'target_name': target_name}, open(target_file_name, 'w'))
        self.description.to_json(description_file_name)
        for kwarg_name in ['raw_preproc_kwargs', 'window_kwargs', 'window_preproc_kwargs']:
            if hasattr(self, kwarg_name):
                kwargs_path = os.path.join(path, '.'.join([kwarg_name, 'json']))
                kwargs = getattr(self, kwarg_name)
                if kwargs is not None:
                    json.dump(kwargs, open(kwargs_path, 'w'))

    @property
    def description(self):
        df = pd.DataFrame([ds.description for ds in self.datasets])
        df.reset_index(inplace=True, drop=True)
        return df

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.DataFrame
            Description in the form key: value where the length of the value
            has to match the number of datasets.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = pd.DataFrame(description)
        for key, value in description.items():
            for ds, value_ in zip(self.datasets, value):
                ds.set_description({key: value_}, overwrite=overwrite)

    def save(self, path, overwrite=False, offset=None):
        """Save datasets to files by creating one subdirectory for each dataset:
        path/
            target_name.json (if target_name is not None and dataset is raw)
            0/
                0-raw.fif | 0-epo.fif
                description.json
            1/
                1-raw.fif | 1-epo.fif
                description.json
            ...

        Parameters
        ----------
        path : str
            Directory to which -raw.fif | -epo.fif and .json files are stored.
        overwrite : bool
            Whether to delete old files (target_name.json, description.json,
             -raw.fif, -epo.fif) in specified directory prior to saving.
        offset : int | None
            If provided the integer is added to the id of the datasets in the
            concat. This is useful in the setting of very large datasets, where
            one dataset has to be processed and saved at a time to account for
            its original position.
        """
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        if not (hasattr(self.datasets[0], 'raw') or hasattr(
                self.datasets[0], 'windows')):
            raise ValueError("dataset should have either raw or windows "
                             "attribute")
        description_files = glob(os.path.join(path, '**/description.json'))
        file_name_pattern = ["{}-raw.fif", "{}-epo.fif"]
        # read all -epo.fif and -raw.fif files
        fif_files = []
        for file_name in file_name_pattern:
            fif_files.extend(
                glob(os.path.join(path, f"**/*{file_name.lstrip('{}')}")))
        if overwrite:
            self._clear_up_directory(path, description_files, fif_files)
        is_raw = hasattr(self.datasets[0], 'raw')
        file_name = file_name_pattern[0] if is_raw else file_name_pattern[1]
        self._save_datasets_and_description(
            path, overwrite, file_name, is_raw, offset)
        if is_raw:
            self._save_target_name(path)

    def _save_datasets_and_description(
            self, path, overwrite, file_name, is_raw, offset):
        if offset is None:
            offset = 0
        # save all the datasets and description
        for i_ds, ds in enumerate(self.datasets):
            # create one subdirectory per dataset
            this_path = os.path.join(path, str(i_ds+offset))
            if not os.path.exists(this_path):
                os.makedirs(this_path)
            fif_file_path = os.path.join(
                this_path, file_name.format(i_ds+offset))
            description_file = os.path.join(this_path, 'description.json')
            if not overwrite:
                # we do not accept fif files or description files
                self._raise_error_if_file_exists(fif_file_path)
                self._raise_error_if_file_exists(description_file)
            if is_raw:
                ds.raw.save(fif_file_path)
            else:
                ds.windows.save(fif_file_path)
            ds.description.to_json(description_file)

    def _save_target_name(self, path):
        # make sure we do not have an inconsistency in the target name
        expected_target_name = self.datasets[0].target_name
        assert all([expected_target_name == ds.target_name
                    for ds in self.datasets]), (
            "All datasets should have same target name")
        target_name_file = os.path.join(path, 'target_name.json')
        # always write / overwrite target name. if it already existed, it is
        # not a dealbreaker. code will always crash if overwrite = False and
        # fif files / description files exist. if code did not crash there, we
        # likely have a use case as in plot_tuh_eeg.py where a dataset is split
        # into smaller chunks and saved one dataset at a time. there, the
        # target name is the same for all datasets
        with open(target_name_file, 'w') as f:
            json.dump({'target_name': expected_target_name}, f)

    @staticmethod
    def _clear_up_directory(path, description_files, fif_files):
        # remove all existing "{}-raw.fif" | "{}-epo.fif" and description.json
        # files. afterwards delete subdirectories if empty
        for fif_file, description_file in zip(fif_files, description_files):
            os.remove(fif_file)
            os.remove(description_file)
            subdir_name = os.path.dirname(fif_file)
            # remove subdirectory if empty
            if len(os.listdir(subdir_name)) == 0:
                os.rmdir(subdir_name)
        # remove target name file if it exists
        target_file_name = os.path.join(path, 'target_name.json')
        if os.path.isfile(target_file_name):
            os.remove(target_file_name)

    @staticmethod
    def _raise_error_if_file_exists(file_path):
        if os.path.exists(file_path):
            raise FileExistsError(
                f'Another dataset was already saved to this directory. '
                f'{file_path} exists. '
                'Please clean up manually, set overwrite to True, or save '
                'to another directory.')

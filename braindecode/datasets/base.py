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
    target_name : str | None
        Name of the index in `description` that should be used to provide the
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
        self.target_name = target_name
        if self.target_name is not None and self.target_name not in self.description:
            raise ValueError(f"'{self.target_name}' not in description.")

    def __getitem__(self, index):
        X = self.raw[:, index][0]
        y = None
        if self.target_name is not None:
            y = self.description[self.target_name]
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
            # if they key is already in the existing description, drop it
            if key in self._description:
                assert overwrite, (f"'{key}' already in description. Please "
                                   f"rename or set overwrite to True.")
                self._description.pop(key)
        self._description = pd.concat([self.description, description])


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

        self.y = self.windows.metadata.loc[:, 'target'].to_numpy()
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
    seq_target_transform : callable | None
        Function to call on sequences of targets before returning them. Only
        called if the dataset is indexed with lists of indices to return
        sequences.
    """
    def __init__(self, list_of_ds, seq_target_transform=None):
        # if we get a list of BaseConcatDataset, get all the individual datasets
        if list_of_ds and isinstance(list_of_ds[0], BaseConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)

        self.seq_target_transform = seq_target_transform

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)
        if self.seq_target_transform is not None:
            y = self.seq_target_transform(y)

        return X, y

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated.
        """
        if isinstance(idx, Iterable):  # Sample multiple windows
            return self._get_sequence(idx)
        else:
            return super().__getitem__(idx)

    def split(self, by=None, property=None, split_ids=None):
        """Split the dataset based on information listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        by : str | list
            If ``by`` is a string, splitting is performed based on the description
            DataFrame column with this name.
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
    def seq_target_transform(self):
        return self._seq_target_transform

    @seq_target_transform.setter
    def seq_target_transform(self, fn):
        if not (callable(fn) or fn is None):
            raise TypeError('seq_target_transform must be a callable.')
        self._seq_target_transform = fn

    def save(self, path, overwrite=False):
        """Save dataset to files.

        Parameters
        ----------
        path : str
            Directory to which .fif / -epo.fif and .json files are stored.
        overwrite : bool
            Whether to delete old files (.json, .fif, -epo.fif) in specified directory
            prior to saving.
        """
        assert len(self.datasets) > 0, "Expect at least one dataset"
        assert (hasattr(self.datasets[0], 'raw') + hasattr(
            self.datasets[0], 'windows') == 1), (
            "dataset should have either raw or windows attribute")
        file_names_ = ["{}-raw.fif", "{}-epo.fif"]
        description_file_name = os.path.join(path, 'description.json')
        target_file_name = os.path.join(path, 'target_name.json')
        if not overwrite:
            if (os.path.exists(description_file_name) or
                    os.path.exists(target_file_name)):
                raise FileExistsError(
                    f'{description_file_name} or {target_file_name} exist in {path}.')
        else:
            for file_name in file_names_:
                file_names = glob(os.path.join(path, f"*{file_name.lstrip('{}')}"))
                _ = [os.remove(f) for f in file_names]
            if os.path.isfile(target_file_name):
                os.remove(target_file_name)
            if os.path.isfile(description_file_name):
                os.remove(description_file_name)

        concat_of_raws = hasattr(self.datasets[0], 'raw')
        file_name = file_names_[0] if concat_of_raws else file_names_[1]
        if concat_of_raws:
            # for checks that all have same target name and for
            # saving later
            target_name = self.datasets[0].target_name
        for i_ds, ds in enumerate(self.datasets):
            full_file_path = os.path.join(path, file_name.format(i_ds))
            if concat_of_raws:
                ds.raw.save(full_file_path, overwrite=overwrite)
                assert ds.target_name == target_name, "All datasets should have same target name"
            else:
                ds.windows.save(full_file_path, overwrite=overwrite)

        if concat_of_raws:
            json.dump({'target_name': target_name}, open(target_file_name, 'w'))
        self.description.to_json(description_file_name)

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

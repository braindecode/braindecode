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

import warnings
import glob

import numpy as np
import pandas as pd
import mne

from torch.utils.data import Dataset, ConcatDataset


class BaseDataset(Dataset):
    """A base dataset holds a mne.Raw, and a pandas.DataFrame with additional
    description, such as subject_id, session_id, run_id, or age or gender of
    subjects.

    Parameters
    ----------
    raw: mne.io.Raw
    description: dict | pandas.Series | None
        holds additional description about the continuous signal / subject
    target_name: str | None
        name of the index in `description` that should be use to provide the
        target (e.g., to be used in a prediction task later on).
    """
    def __init__(self, raw, description=None, target_name=None):
        self.raw = raw
        self.description = _create_description(description)

        # save target name for load/save later
        self.target_name = target_name
        if target_name is None:
            self.target = None
        elif target_name in self.description:
            self.target = self.description[target_name]
        else:
            raise ValueError(f"'{target_name}' not in description.")

    def __getitem__(self, index):
        return self.raw[:, index][0], self.target

    def __len__(self):
        return len(self.raw)


def _create_description(description):
    if description is not None:
        if (not isinstance(description, pd.Series)
                and not isinstance(description, dict)):
            raise ValueError(f"'{description}' has to be either a "
                             f"pandas.Series or a dict")
        if isinstance(description, dict):
            description = pd.Series(description)
    return description


class WindowsDataset(BaseDataset):
    """Applies a windower to a base dataset.

    Parameters
    ----------
    windows: mne.Epochs
        windows obtained through the application of a windower to a
        BaseDataset
    description: dict | pandas.Series | None
        holds additional info about the windows
    """
    def __init__(self, windows, description=None):
        self.windows = windows
        self.description = _create_description(description)
        self.y = np.array(self.windows.metadata.loc[:, 'target'])
        self.crop_inds = np.array(self.windows.metadata.loc[:,
                                  ['i_window_in_trial', 'i_start_in_trial',
                                   'i_stop_in_trial']])

    def __getitem__(self, index):
        X = self.windows.get_data(item=index)[0].astype('float32')
        y = self.y[index]
        # necessary to cast as list to get list of
        # three tensors from batch, otherwise get single 2d-tensor...
        crop_inds = list(self.crop_inds[index])
        return X, y, crop_inds

    def __len__(self):
        return len(self.windows.events)


class BaseConcatDataset(ConcatDataset):
    """A base class for concatenated datasets. Holds either mne.Raw or
    mne.Epoch in self.datasets and has a pandas DataFrame with additional
    description.

    Parameters
    ----------
    list_of_ds: list
        list of BaseDataset, BaseConcatDataset or WindowsDataset
    """
    def __init__(self, list_of_ds):
        # if we get a list of BaseConcatDataset, get all the individual datasets
        if list_of_ds and isinstance(list_of_ds[0], BaseConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)
        self.description = pd.DataFrame([ds.description for ds in list_of_ds])
        self.description.reset_index(inplace=True, drop=True)

    def split(self, by=None, property=None, split_ids=None):
        """Split the dataset based on information listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        by: str | list(int) | list(list(int))
            If by is a string, splitting is performed based on the description
            DataFrame column with this name.
            If by is a (list of) list of integers, the position in the first
            list corresponds to the split id and the integers to the
            datapoints of that split.
        property: str
            Some property which is listed in info DataFrame.
        split_ids: list(int) | list(list(int))
            List of indices to be combined in a subset.

        Returns
        -------
        splits: dict{str: BaseConcatDataset}
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
        pd.DataFrame:
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


def read_all_file_names(directory, extension):
    """Read all files with specified extension from given path and sorts them
    based on a given sorting key.

    Parameters
    ----------
    directory: str
        Parent directory to be searched for files of the specified type
    extension: str
        File extension, i.e. ".edf" or ".txt"

    Returns
    -------
    file_paths: list(str)
        A list to all files found in (sub)directories of path.
    """
    assert extension.startswith(".")
    file_paths = glob.glob(directory + "**/*" + extension, recursive=True)
    assert len(file_paths) > 0, (
        f"something went wrong. Found no {extension} files in {directory}")
    return file_paths


class LazyDataset(Dataset):
    """A class that loads stored compute windows when getitem is called.

    Params
    ------
    path: str
        Parent directory of the .fif files serialized with braindecode.datautil.serialization.save_concat_dataset
    """
    def __init__(self, path):
        json_files = read_all_file_names(path, ".json")
        json_description_files = [
            f for f in json_files if f.endswith("description.json")]
        self.description = pd.concat([
            pd.read_json(f) for f in json_description_files],
            ignore_index=True)
        self.use_mne = False
        if self.use_mne:
            self.window_cumsum = self.description.n_windows.to_numpy().cumsum()
            self.file_paths = read_all_file_names(path, ".fif")
        else:
            file_paths = read_all_file_names(path, ".npy")
            self.window_paths = [f for f in file_paths if "x-" in f]
            self.target_paths = [f for f in file_paths if "y-" in f]
            self.ind_paths = [f for f in file_paths if "ind-" in f]
            assert len(self.window_paths) == len(self.target_paths) == len(self.ind_paths)

    def __getitem__(self, idx):
        if self.use_mne:
            # TODO: test the backtracking !
            rec_i = (self.window_cumsum <= idx).sum()
            # if this is not accessing the windows of the very first recording,
            # subtract the count of all previous windows to find the correct
            # intra-recording window index
            if rec_i != 0:
                idx = idx - self.window_cumsum[max(rec_i - 1, 0)]
            epochs = mne.read_epochs(self.file_paths[rec_i], preload=False)
            y = epochs.metadata.loc[idx, 'target']
            crop_inds = list(
                epochs.metadata.loc[idx, [
                    'i_window_in_trial', 'i_start_in_trial', 'i_stop_in_trial']])
            return epochs.get_data(item=idx), y, crop_inds
        else:
            x = np.load(self.window_paths[idx])
            y = np.load(self.target_paths[idx])
            ind = np.load(self.ind_paths[idx])
            return x, y, ind

    def __len__(self):
        if self.use_mne:
            return self.window_cumsum[-1]
        else:
            return len(self.window_paths)

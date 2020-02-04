"""
Dataset base classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, ConcatDataset, Subset


class BaseDataset(Dataset):
    """A base dataset holds a mne.Raw, and a pandas.DataFrame with additional
    description, such as subject_id, session_id, run_id, or age or gender of
    subjects.

    Parameters
    ----------
    raw: mne.Raw
    description: pandas.DataFrame
        holds additional description about the continuous signal / subject
    """
    def __init__(self, raw, description, target_name=None):
        self.raw = raw
        self.description = description
        self.target = target_name
        if target_name is not None:
            assert target_name in self.description, (
                f"'{target_name}' not in description")
            self.target = self.description[target_name].values[0]

    def __getitem__(self, index):
        return self.raw[:, index][0], self.target

    def __len__(self):
        return len(self.raw)


class WindowsDataset(BaseDataset):
    """Applies a windower to a base dataset.

    Parameters
    ----------
    windows: mne.Epochs
        windows/supercrops obtained throiugh application of a windowing
        function to a BaseDataset
    description: pandas.DataFrame
        hold additional info about the windows
    """
    def __init__(self, windows, description):
        self.windows = windows
        self.description = description

    def __getitem__(self, index):
        target = self.windows.metadata["target"]
        keys = ['i_supercrop_in_trial', 'i_start_in_trial', 'i_stop_in_trial']
        supercrop_ind = self.windows.metadata.iloc[index][keys].to_list()
        x = self.windows[index].get_data().squeeze(0)
        return x, target[index], supercrop_ind

    def __len__(self):
        return len(self.windows.events)


class BaseConcatDataset(ConcatDataset):
    """A base class for concatenated datasets. Holds either mne.Raw or
    mne.Epoch in self.datasets and has a pandas DataFrame with additional
    description.

    Parameters
    ----------
    list_of_ds: list
        list of BaseDataset of WindowsDataset to be concatenated.
    """
    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.description = pd.concat(ds.description for ds in list_of_ds)

    def split(self, some_property=None, split_ids=None):
        """Split the dataset based on some property listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        some_property: str
            some property which is listed in info DataFrame
        split_ids: list(int)
            list of indices to be combined in a subset

        Returns
        -------
        splits: dict{split_name: subset}
            mapping of split name based on property or index based on split_ids
            to subset of the data
        """
        assert split_ids is None or some_property is None, (
            "can split either based on ids or based on some property")
        if split_ids is None:
            split_ids = _split_ids(self.description, some_property)
        else:
            split_ids = {split_i: split
                         for split_i, split in enumerate(split_ids)}
        # split_ids are indices for WindowsDatasets
        supercrop_ids = _windows_dataset_ids_to_supercrop_ids(
            split_ids, self.cumulative_sizes)
        return {split_name: Subset(self, split)
                for split_name, split in supercrop_ids.items()}


def _windows_dataset_ids_to_supercrop_ids(dataset_ids, cumulative_sizes):
    supercrop_ids = {}
    for split_name, windows_is in dataset_ids.items():
        this_supercrop_ids = _supercrop_ids_of_windows(
            cumulative_sizes, windows_is)
        supercrop_ids[split_name] = this_supercrop_ids
    return supercrop_ids


def _supercrop_ids_of_windows(cumulative_sizes, windows_is):
    i_stops = cumulative_sizes
    i_starts = np.insert(cumulative_sizes[:-1], 0, [0])
    i_per_window = []
    for i_window in windows_is:
        i_per_window.append(list(range(i_starts[i_window], i_stops[i_window])))
    all_i_windows = np.concatenate(i_per_window)
    return all_i_windows


def _split_ids(df, some_property):
    assert some_property in df
    split_ids = {}
    for group_name, group in df.groupby(some_property):
        split_ids.update({group_name: list(group.index)})
    return split_ids
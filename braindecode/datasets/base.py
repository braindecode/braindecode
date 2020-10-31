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
from typing import Tuple
import numpy as np
import pandas as pd
import bisect
import torch
from torch.utils.data import Dataset, ConcatDataset
from .transform_classes import Transform
from ..util import identity


class Datum:
    """The Datum class is mainly used to provide contextual informations to transforms, when they are applied on a data unitary chunk. For example, the "delay_signal" function needs to know the index of the data to find the data right before and do the shift. Similarly, the "merge_signal" function needs to know the data index, to do the merge with another signal with similar index.
    """
    def __init__(self, X, y) -> None:
        """Initialize one instance
        Args:
            X (Tensor): Tensor containing the signal
            y (Union[Int,float]): Contains the label/value that should be predicted
        """
        self.X = X
        self.y = y
        

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
        if (not isinstance(description, pd.Series) and not isinstance(description, dict)):
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

    def __init__(self, windows, description=None, transform=None):
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
        list of BaseDataset, BaseConcatDataset or WindowsDataset/TransformDataset
    """

    def __init__(self, list_of_ds):
        # if we get a list of BaseConcatDataset, get all the individual datasets
        if isinstance(list_of_ds[0], BaseConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)
        self.description = pd.DataFrame([ds.description for ds in list_of_ds])
        self.description.reset_index(inplace=True, drop=True)

    def split(self, property=None, split_ids=None):
        """Split the dataset based on some property listed in its description
        DataFrame or based on indices.
        Parameters
        ----------
        property: str
            some property which is listed in info DataFrame
        split_ids: list(int)
            list of indices to be combined in a subset
        Returns
        -------
        splits: dict{split_name: BaseConcatDataset}
            mapping of split name based on property or index based on split_ids
            to subset of the data
        """
        if split_ids is None and property is None:
            raise ValueError('Splitting requires defining ids or a property.')
        if split_ids is None:
            if property not in self.description:
                raise ValueError(f'{property} not found in self.description')
            split_ids = {k: list(v) for k, v in self.description.groupby(
                property).groups.items()}
        else:
            split_ids = {split_i: split
                         for split_i, split in enumerate(split_ids)}

        return {split_name: BaseConcatDataset(
            [self.datasets[ds_ind] for ds_ind in ds_inds])
            for split_name, ds_inds in split_ids.items()}

    def tinying_dataset(self, subset_dict):

        def take_dataset_subset(windows_dataset, indice_list):
            windows_dataset.windows = windows_dataset.windows[tuple(indice_list)]
            windows_dataset.y = windows_dataset.y[indice_list]
            windows_dataset.crop_inds = windows_dataset.crop_inds[indice_list]

        for i in subset_dict.keys():
            take_dataset_subset(self.datasets[i], subset_dict[i])
        self.cumulative_sizes = self.cumsum(self.datasets)


class TransformConcatDataset(BaseConcatDataset):
    """A variation of the base class that includes transforms 
    Parameters
    ----------
    list_of_ds: list
        list of TransformDataset
    """

    def __init__(self, list_of_ds):
        # if we get a list of BaseConcatDataset, get all the individual datasets
        def check_equal(iterator):
            return len(set(iterator)) <= 1
        super().__init__(list_of_ds)
        assert check_equal([list_of_ds.subpolicies_list for ds in list_of_ds])
        self.subpolicies_list = list_of_ds[0].subpolicies_list

    def update_augmentation_policy(self, newlist):
        for i in range(len(self.datasets)):
            self.datasets[i].subpolicies_list = newlist
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.subpolicies_list = newlist

    def get_raw_data(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_raw_data(sample_idx)


class TransformDataset(WindowsDataset):

    def __init__(self, windows, description=None, subpolicies_list=Tuple(Transform(identity))):
        super(TransformDataset, self).__init__(windows, description)
        self.subpolicies_list = subpolicies_list

    def __getitem__(self, index):

        img_index = index // len(self.subpolicies_list)
        tf_index = index % len(self.subpolicies_list)
        X = torch.from_numpy(self.windows.get_data(item=img_index)[0].astype('float32'))
        y = self.y[img_index]
        for transform in self.subpolicies_list[tf_index]:
            datum = transform(datum)
        crop_inds = list(self.crop_inds[img_index])
        return datum.X, y, crop_inds  # TODO : modifier getitem de base sur la version gitté

    def __len__(self):
        return len(self.windows.events) * len(self.subpolicies_list)

    def get_raw_data(self, index):
        img_index = index // len(self.subpolicies_list)
        X = torch.from_numpy(self.windows.get_data(item=img_index)[0].astype('float32'))
        y = self.y[img_index]
        crop_inds = list(self.crop_inds[img_index])
        return X, y, crop_inds

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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from .transform_classes import TransformSignal
from ..util import identity


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

    def __init__(self, raw, description=None, target_name=None, transform_list=[[TransformSignal(identity)]]):
        self.raw = raw
        self.description = _create_description(description)
        self.transform_list = transform_list
        # save target name for load/save later
        self.target_name = target_name
        if target_name is None:
            self.target = None
        elif target_name in self.description:
            self.target = self.description[target_name]
        else:
            raise ValueError(f"'{target_name}' not in description.")

    def __getitem__(self, index):
        return self.transform_list.transform(self.raw[:, index][0]), self.target

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
        self.transform = transform

    def __getitem__(self, index):
        X = self.windows.get_data(item=index)[0].astype('float32')
        if self.transform:
            X = self.transform(X)
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
        self.transform_list = list_of_ds[0].transform_list

    def change_transform_list(self, newlist):
        for i in range(len(self.datasets)):
            self.datasets[i].transform_list = newlist
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.transform_list = newlist

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


class TransformDataset(WindowsDataset):

    def __init__(self, windows, description=None, transform_list=[[TransformSignal(identity)]]):
        super(TransformDataset, self).__init__(windows, description)
        self.transform_list = transform_list

    def __getitem__(self, index):

        img_index = index // len(self.transform_list)
        tf_index = index % len(self.transform_list)
        X = torch.from_numpy(self.windows.get_data(item=img_index)[0].astype('float32'))
        y = self.y[img_index]
        for transform in self.transform_list[tf_index]:
            X = transform.transform(X)
        # necessary to cast as list to get list of
        # three tensors from batch, otherwise get single 2d-tensor...
        crop_inds = list(self.crop_inds[img_index])
        return X, y, crop_inds  # TODO : modifier getitem de base sur la version gitté

    def __len__(self):
        return len(self.windows.events) * len(self.transform_list)

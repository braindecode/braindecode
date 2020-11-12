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

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, ConcatDataset
from braindecode.augmentation.transform_class import Transform
from braindecode.augmentation.transforms import identity


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
        if (not isinstance(description, pd.Series) and
                not isinstance(description, dict)):
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
        self.crop_inds = np.array(
            self.windows.metadata.loc[:,
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


class AugmentedDataset(Dataset):

    def __init__(self, ds, list_of_transforms=[Transform(identity)]) -> None:
        self.list_of_transforms = list_of_transforms
        self.ds = ds
        self.required_variables = self.initialize_required_variables()

    def __len__(self):
        return(len(self.ds) * len(self.list_of_transforms))

    def __getitem__(self, index):
        tf_index = index % len(self.list_of_transforms)
        img_index = index // len(self.list_of_transforms)
        X, y, crops_ind = (self.ds[img_index])

        class Datum:
            def __init__(self, X, y, crops_ind, ds, required_variables):
                self.X = X
                self.y = y
                self.crops_ind = crops_ind
                self.ds = ds
                self.required_variables = required_variables

        transf_datum = self.list_of_transforms[tf_index](
            Datum(X, y, crops_ind, self.ds, self.required_variables))
        X, y, crops_ind = transf_datum.X, transf_datum.y, transf_datum.crops_ind

        return X, y, crops_ind

    def initialize_required_variables(self):
        for transform in self.list_of_transforms:
            for key in transform.required_variables.keys():
                self.required_variables[key] = \
                    transform.required_variables[key](self.ds)

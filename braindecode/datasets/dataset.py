"""
Dataset classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    A base dataset.

    Parameters
    ----------
    raw: mne.Raw
    info: pandas.DataFrame
        holds additional information about the raw

    """
    def __init__(self, raw, info):
        """


        """
        self.raw = raw
        self.info = info

    def __getitem__(self, index):
        target = None
        if hasattr(self.info, "target"):
            target = self.info["target"]
        return self.raw, target

    def __len__(self):
        return len(self.raw)


class WindowsDataset(BaseDataset):
    """
    Applies a windower to a base dataset.

    Parameters
    ----------
    dataset: BaseDataset
    windower: braindecode.datautil.windowers.windower
        a windower applied to the dataset to extract windows/supercrops

    """
    def __init__(self, dataset, windower):
        self.windows = windower(dataset)
        self.info = dataset.info

    def __getitem__(self, index):
        target = self.windows.events[:,-1]
        keys = ['i_supercrop_in_trial', 'i_start_in_trial', 'i_stop_in_trial']
        info = self.windows.metadata.iloc[index][keys].to_list()
        return self.windows[index], target[index], info

    def __len__(self):
        return len(self.windows.events)
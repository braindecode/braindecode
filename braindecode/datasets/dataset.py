"""
Dataset classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from torch.utils.data import Dataset


class WindowsDataset(Dataset):
    """
    Dataset wrapper around mne.Epochs

    Parameters
    ----------
    windows : mne.Epochs
        epoched data to wrap
    target : str
        target specified by user to be decoded
    transforms : transform object | list of transform objects | None
        preprocessor transform(s) applied sequentially on windowed data.
    """

    def __init__(self, windows, target="target", transforms=None):
        self.windows = windows
        self.target = target
        if self.target != "target":
            assert self.target in self.windows.info["subject_info"].keys()
        self.transforms = (
            transforms
            if isinstance(transforms, list) or transforms is None
            else [transforms]
        )

        # XXX Handle multitarget case

    def __getitem__(self, index):
        """Return one window and its label

        Parameters
        ----------
        index : int
            index of the window to return

        Returns
        -------
        x : ndarray, shape (n_channels, window_size)
            one window of data
        y : int | float
            window target
        """
        x = np.squeeze(self.windows[index].get_data(), axis=0)
        if self.target == "target":
            y = self.windows.metadata.iloc[index]["target"]
        else:
            y = self.windows.info["subject_info"][self.target]

        if self.transforms is not None:
            for transform in self.transforms:
                x = transform(x)

        return x, y  # robin wants i_trial, i_start, i_stop here

    def __len__(self):
        return self.windows.metadata.shape[0]
        # XXX: The following would fail if data has not been preloaded yet:
        # return len(self.windows)

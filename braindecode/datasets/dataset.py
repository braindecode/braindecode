"""
Dataset classes that handle data
"""

import numpy as np
from torch.utils.data import Dataset


class WindowsDataset(Dataset):
    """Dataset wrapper around mne.Epochs

    Parameters
    ----------
    windows : mne.Epochs
        epoched data to wrap
    target : str
        target specified by user to be decoded
    transformer : list of sklearn.base.TransformerMixin
        preprocessor function applied on windowed data
    """

    def __init__(self, windows, target='target', transformer=None):
        self.windows = windows
        self.target = target
        if self.target != 'target':
            assert self.target in self.windows.info['subject_info'].keys()
        self.transformer = transformer if isinstance(transformer, list)\
            else [transformer]
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
        x = np.squeeze(self.windows[index].get_data())
        if self.target == 'target':
            y = self.windows.metadata.iloc[index]['target']
        else:
            y = self.windows.info['subject_info'][self.target]

        if self.transformer:
            for transformer in self.transformer:
                x = transformer.fit_transform(x)

        return x, y

    def __len__(self):
        return self.windows.metadata.shape[0]
        # XXX: The following would fail if data has not been preloaded yet:
        # return len(self.windows)


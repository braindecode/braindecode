# Authors: Lukas Gemein
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import numpy as np
from torch.utils.data import Dataset

from braindecode.datautil.windowers import _compute_window_inds


class CroppedXyDataset(Dataset):
    def __init__(self, X, y, input_time_length, n_preds_per_input):
        """
        A dataset that creates a cropped dataset from just X,y input.

        Parameters
        ----------
        X: 3darray of float
            Trial data in the form trials x (M/EEG) channels x time
        y: ndarray or list of int
            Trial labels.
        input_time_length: int
            Input time length aka window size in number of samples.
        n_preds_per_input:
            Number of predictions per window (=> will be window stride)
            in number of samples.
        """
        self.X = X
        self.y = y
        # on init we have to generate all the window indices,
        # so we know number of windows (length of the dataset)
        # and we can have a mapping:
        # i_window -> i_trial, start, stop
        i_window_to_idx = []
        for i_trial, trial in enumerate(X):
            idx = _compute_window_inds(
                starts=np.array([0]),
                stops=trial.shape[1],
                start_offset=0,
                stop_offset=0,
                size=input_time_length,
                stride=n_preds_per_input,
                drop_samples=False,
            )
            for _, i_window_in_trial, start, stop in zip(*idx):
                i_window_to_idx.append(
                    dict(
                        i_trial=i_trial,
                        i_window_in_trial=i_window_in_trial,
                        start=start,
                        stop=stop,
                    )
                )
        self.i_window_to_idx = i_window_to_idx

    def __len__(self):
        return len(self.i_window_to_idx)

    def __getitem__(self, i):
        window_idx = self.i_window_to_idx[i]
        X = self.X[
            window_idx["i_trial"],
            :,
            window_idx["start"]: window_idx["stop"],
            ]
        y = self.y[window_idx["i_trial"]]
        returned_idx = (
            window_idx["i_window_in_trial"],
            window_idx["start"],
            window_idx["stop"],
        )
        return X, y, returned_idx

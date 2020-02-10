# Authors: Lukas Gemein
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import numpy as np
from torch.utils.data import Dataset

from braindecode.datautil.windowers import _compute_supercrop_inds


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
            Input time length aka supercrop size in number of samples.
        n_preds_per_input:
            Number of predictions per supercrop (=> will be supercrop stride)
            in number of samples.
        """
        self.X = X
        self.y = y
        # on init we have to generate all the supercrop indices,
        # so we know number of supercrops (length of the dataset)
        # and we can have a mapping:
        # i_supercrop -> i_trial, start, stop
        i_supercrop_to_idx = []
        for i_trial, trial in enumerate(X):
            idx = _compute_supercrop_inds(
                np.array([0]),
                trial.shape[1],
                0,
                0,
                input_time_length,
                n_preds_per_input,
                drop_samples=False,
            )
            for _, i_supercrop_in_trial, start, stop in zip(*idx):
                i_supercrop_to_idx.append(
                    dict(
                        i_trial=i_trial,
                        i_supercrop_in_trial=i_supercrop_in_trial,
                        start=start,
                        stop=stop,
                    )
                )
        self.i_supercrop_to_idx = i_supercrop_to_idx

    def __len__(self):
        return len(self.i_supercrop_to_idx)

    def __getitem__(self, i):
        supercrop_idx = self.i_supercrop_to_idx[i]
        X = self.X[
            supercrop_idx["i_trial"],
            :,
            supercrop_idx["start"]: supercrop_idx["stop"],
            ]
        y = self.y[supercrop_idx["i_trial"]]
        returned_idx = (
            supercrop_idx["i_supercrop_in_trial"],
            supercrop_idx["start"],
            supercrop_idx["stop"],
        )
        return X, y, returned_idx

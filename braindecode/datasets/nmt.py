"""
Dataset classes for the NMT EEG Corpus dataset.

The NMT Scalp EEG Dataset is an open-source annotated dataset of healthy and
pathological EEG recordings for predictive modeling. This dataset contains
2,417 recordings from unique participants spanning almost 625 h.

"""

# Authors: Mohammad Bayazi <mj.darvishi92@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import glob
import os

import mne
import pandas as pd

from braindecode.datasets.base import BaseConcatDataset, BaseDataset


class NMT(BaseConcatDataset):
    """The NMT Scalp EEG Dataset.

    An Open-Source Annotated Dataset of Healthy and Pathological EEG
    Recordings for Predictive Modeling.

    This dataset contains 2,417 recordings from unique participants spanning
    almost 625 h.

    Here, the dataset can be used for three tasks, brain-age, gender prediction,
    abnormality detection.

    The dataset is described in [Khan2022]_.

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later than the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.

    References
    ----------
    .. [Khan2022] Khan, H.A.,Ul Ain, R., Kamboh, A.M., Butt, H.T.,Shafait,S.,
    Alamgir, W., Stricker, D. and Shafait, F., 2022. The NMT scalp EEG dataset:
    an open-source annotated dataset of healthy and pathological EEG recordings
    for predictive modeling. Frontiers in neuroscience, 15, p.755817.
    """

    def __init__(
        self, path, target_name="pathological", recording_ids=None, preload=False
    ):
        file_paths = glob.glob(
            os.path.join(path, "**" + os.sep + "*.edf"), recursive=True
        )
        # sort by subject id
        file_paths = sorted(
            file_paths, key=lambda p: int(os.path.splitext(p)[0].split(os.sep)[-1])
        )
        if recording_ids is not None:
            file_paths = [file_paths[rec_id] for rec_id in recording_ids]

        # read labels and rearrange them to match TUH Abnormal EEG Corpus
        description = pd.read_csv(
            os.path.join(path, "Labels.csv"), index_col="recordname"
        )
        if recording_ids is not None:
            description = description.iloc[recording_ids]
        description.replace(
            {
                "not specified": "X",
                "female": "F",
                "male": "M",
                "abnormal": True,
                "normal": False,
            },
            inplace=True,
        )
        description.rename(columns={"label": "pathological"}, inplace=True)
        description.reset_index(drop=True, inplace=True)
        description["path"] = file_paths
        description = description[["path", "pathological", "age", "gender"]]

        base_datasets = []
        for recording_id, d in description.iterrows():
            raw = mne.io.read_raw_edf(d.path, preload=preload)
            d["n_samples"] = raw.n_times
            d["sfreq"] = raw.info["sfreq"]
            d["train"] = "train" in d.path.split(os.sep)
            base_dataset = BaseDataset(raw, d, target_name)
            base_datasets.append(base_dataset)
        super().__init__(base_datasets)

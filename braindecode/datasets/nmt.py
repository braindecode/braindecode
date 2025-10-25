"""
Dataset classes for the NMT EEG Corpus dataset.

The NMT Scalp EEG Dataset is an open-source annotated dataset of healthy and
pathological EEG recordings for predictive modeling. This dataset contains
2,417 recordings from unique participants spanning almost 625 h.

Note:
    - The signal unit may not be uV and further examination is required.
    - The spectrum shows that the signal may have been band-pass filtered from about 2 - 33Hz,
    which needs to be further determined.

"""

# Authors: Mohammad Bayazi <mj.darvishi92@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from unittest import mock

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.datasets import fetch_dataset

from braindecode.datasets.base import BaseConcatDataset, BaseDataset

NMT_URL = "https://zenodo.org/record/10909103/files/NMT.zip"
NMT_archive_name = "NMT.zip"
NMT_folder_name = "MNE-NMT-eeg-dataset"
NMT_dataset_name = "NMT-EEG-Corpus"

NMT_dataset_params = {
    "dataset_name": NMT_dataset_name,
    "url": NMT_URL,
    "archive_name": NMT_archive_name,
    "folder_name": NMT_folder_name,
    "hash": "77b3ce12bcaf6c6cce4e6690ea89cb22bed55af10c525077b430f6e1d2e3c6bf",
    "config_key": NMT_dataset_name,
}


class NMT(BaseConcatDataset):
    """The NMT Scalp EEG Dataset.

    An Open-Source Annotated Dataset of Healthy and Pathological EEG
    Recordings for Predictive Modeling.

    This dataset contains 2,417 recordings from unique participants spanning
    almost 625 h.

    Here, the dataset can be used for three tasks, brain-age, gender prediction,
    abnormality detection.

    The dataset is described in [Khan2022]_.

    .. versionadded:: 0.9

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
        Can be "pathological", "gender", or "age".
    preload: bool
        If True, preload the data of the Raw objects.

    References
    ----------
    .. [Khan2022] Khan, H.A.,Ul Ain, R., Kamboh, A.M., Butt, H.T.,Shafait,S.,
        Alamgir, W., Stricker, D. and Shafait, F., 2022. The NMT scalp EEG
        dataset: an open-source annotated dataset of healthy and pathological
        EEG recordings for predictive modeling. Frontiers in neuroscience,
        15, p.755817.
    """

    def __init__(
        self,
        path=None,
        target_name="pathological",
        recording_ids=None,
        preload=False,
        n_jobs=1,
    ):
        # correct the path if needed
        if path is not None:
            list_csv = glob.glob(f"{path}/**/Labels.csv", recursive=True)
            if isinstance(list_csv, list) and len(list_csv) > 0:
                path = Path(list_csv[0]).parent

        if path is None or len(list_csv) == 0:
            path = fetch_dataset(
                dataset_params=NMT_dataset_params,
                path=Path(path) if path is not None else None,
                processor="unzip",
                force_update=False,
            )
            # First time we fetch the dataset, we need to move the files to the
            # correct directory.
            path = _correct_path(path)

        # Get all file paths
        file_paths = glob.glob(
            os.path.join(path, "**" + os.sep + "*.edf"), recursive=True
        )

        # sort by subject id
        file_paths = [
            file_path
            for file_path in file_paths
            if os.path.splitext(file_path)[1] == ".edf"
        ]

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

        if n_jobs == 1:
            base_datasets = [
                self._create_dataset(d, target_name, preload)
                for recording_id, d in description.iterrows()
            ]
        else:
            base_datasets = Parallel(n_jobs)(
                delayed(self._create_dataset)(d, target_name, preload)
                for recording_id, d in description.iterrows()
            )

        super().__init__(base_datasets)

    @staticmethod
    def _create_dataset(d, target_name, preload):
        raw = mne.io.read_raw_edf(d.path, preload=preload)
        d["n_samples"] = raw.n_times
        d["sfreq"] = raw.info["sfreq"]
        d["train"] = "train" in d.path.split(os.sep)
        base_dataset = BaseDataset(raw, d, target_name)
        return base_dataset


def _correct_path(path: str):
    """
    Check if the path is correct and rename the file if needed.

    Parameters
    ----------
    path: basestring
        Path to the file.

    Returns
    -------
    path: basestring
        Corrected path.
    """
    if not Path(path).exists():
        unzip_file_name = f"{NMT_archive_name}.unzip"
        if (Path(path).parent / unzip_file_name).exists():
            try:
                os.rename(
                    src=Path(path).parent / unzip_file_name,
                    dst=Path(path),
                )

            except PermissionError:
                raise PermissionError(
                    f"Please rename {Path(path).parent / unzip_file_name}"
                    + f"manually to {path} and try again."
                )
        path = os.path.join(path, "nmt_scalp_eeg_dataset")

    return path


def _get_header(*args):
    all_paths = {**_NMT_PATHS}
    return all_paths[args[0]]


def _fake_pd_read_csv(*args, **kwargs):
    # Create a list of lists to hold the data
    data = [
        ["0000001.edf", "normal", 35, "male", "train"],
        ["0000002.edf", "abnormal", 28, "female", "test"],
        ["0000003.edf", "normal", 62, "male", "train"],
        ["0000004.edf", "abnormal", 41, "female", "test"],
        ["0000005.edf", "normal", 19, "male", "train"],
        ["0000006.edf", "abnormal", 55, "female", "test"],
        ["0000007.edf", "normal", 71, "male", "train"],
    ]

    # Create the DataFrame, specifying column names
    df = pd.DataFrame(data, columns=["recordname", "label", "age", "gender", "loc"])

    return df


def _fake_raw(*args, **kwargs):
    sfreq = 10
    ch_names = [
        "EEG A1-REF",
        "EEG A2-REF",
        "EEG FP1-REF",
        "EEG FP2-REF",
        "EEG F3-REF",
        "EEG F4-REF",
        "EEG C3-REF",
        "EEG C4-REF",
        "EEG P3-REF",
        "EEG P4-REF",
        "EEG O1-REF",
        "EEG O2-REF",
        "EEG F7-REF",
        "EEG F8-REF",
        "EEG T3-REF",
        "EEG T4-REF",
        "EEG T5-REF",
        "EEG T6-REF",
        "EEG FZ-REF",
        "EEG CZ-REF",
        "EEG PZ-REF",
    ]
    duration_min = 6
    data = np.random.randn(len(ch_names), duration_min * sfreq * 60)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data=data, info=info)
    return raw


_NMT_PATHS = {
    # these are actual file paths and edf headers from NMT EEG Corpus
    "nmt_scalp_eeg_dataset/abnormal/train/0000036.edf": b"0       0000036  M 13-May-1951 0000036  Age:32                                          ",
    # noqa E501
    "nmt_scalp_eeg_dataset/abnormal/eval/0000037.edf": b"0       0000037  M 13-May-1951 0000037  Age:32                                          ",
    # noqa E501
    "nmt_scalp_eeg_dataset/abnormal/eval/0000038.edf": b"0       0000038  M 13-May-1951 0000038  Age:32                                          ",
    # noqa E501
    "nmt_scalp_eeg_dataset/normal/train/0000039.edf": b"0       0000039  M 13-May-1951 0000039  Age:32                                          ",
    # noqa E501
    "nmt_scalp_eeg_dataset/normal/eval/0000040.edf": b"0       0000040  M 13-May-1951 0000040  Age:32                                          ",
    # noqa E501
    "nmt_scalp_eeg_dataset/normal/eval/0000041.edf": b"0       0000041  M 13-May-1951 0000041  Age:32                                          ",
    # noqa E501
    "nmt_scalp_eeg_dataset/abnormal/train/0000042.edf": b"0       0000042  M 13-May-1951 0000042  Age:32                                          ",
    # noqa E501
    "Labels.csv": b"0       recordname,label,age,gender,loc       1 0000001.edf,normal,22,not specified,train                                                                      ",
    # noqa E501
}


class _NMTMock(NMT):
    """Mocked class for testing and examples."""

    @mock.patch("glob.glob", return_value=_NMT_PATHS.keys())
    @mock.patch("mne.io.read_raw_edf", new=_fake_raw)
    @mock.patch("pandas.read_csv", new=_fake_pd_read_csv)
    def __init__(
        self,
        mock_glob,
        path,
        recording_ids=None,
        target_name="pathological",
        preload=False,
        n_jobs=1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot save date file")
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                target_name=target_name,
                preload=preload,
                n_jobs=n_jobs,
            )

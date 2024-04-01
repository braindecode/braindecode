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

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")

from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from unittest import mock


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

    def __init__(self, path, target_name='pathological', recording_ids=None, 
                 preload=False, n_jobs=1):
        file_paths = glob.glob(
            os.path.join(path, '**'+os.sep+'*.edf'), recursive=True)
        # sort by subject id
        file_paths = [file_path for file_path in file_paths if os.path.splitext(file_path)[1] == '.edf']
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

def _get_header(*args, **kwargs):
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

    # Display the DataFrame
    # print(df)
    return df

def _fake_raw(*args, **kwargs):
    sfreq = 10
    ch_names = [
        'EEG A1-REF', 'EEG A2-REF',
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
    duration_min = 6
    data = np.random.randn(len(ch_names), duration_min * sfreq * 60)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)
    return raw

_NMT_PATHS = {
    # these are actual file paths and edf headers from NMT EEG Corpus
    'nmt_scalp_eeg_dataset/abnormal/train/0000036.edf': b'0       0000036  M 13-May-1951 0000036  Age:32                                          ',  # noqa E501
    'nmt_scalp_eeg_dataset/abnormal/eval/0000037.edf': b'0       0000037  M 13-May-1951 0000037  Age:32                                          ',  # noqa E501
    'nmt_scalp_eeg_dataset/abnormal/eval/0000038.edf': b'0       0000038  M 13-May-1951 0000038  Age:32                                          ',  # noqa E501
    'nmt_scalp_eeg_dataset/normal/train/0000039.edf': b'0       0000039  M 13-May-1951 0000039  Age:32                                          ',  # noqa E501
    'nmt_scalp_eeg_dataset/normal/eval/0000040.edf': b'0       0000040  M 13-May-1951 0000040  Age:32                                          ',  # noqa E501
    'nmt_scalp_eeg_dataset/normal/eval/0000041.edf': b'0       0000041  M 13-May-1951 0000041  Age:32                                          ',  # noqa E501
    'nmt_scalp_eeg_dataset/abnormal/train/0000042.edf': b'0       0000042  M 13-May-1951 0000042  Age:32                                          ',  # noqa E501
    'Labels.csv': b'0       recordname,label,age,gender,loc       1 0000001.edf,normal,22,not specified,train                                                                      ',  # noqa E501

    # 'tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/078/00007871/s001_2011_07_05/00007871_s001_t001.edf': b'0       00007871 F 01-JAN-1988 00007871 Age:23                                          ',  # noqa E501
    # 'tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/097/00009777/s001_2012_09_17/00009777_s001_t000.edf': b'0       00009777 M 01-JAN-1986 00009777 Age:26                                          ',  # noqa E501
    # 'tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/083/00008393/s002_2012_02_21/00008393_s002_t000.edf': b'0       00008393 M 01-JAN-1960 00008393 Age:52                                          ',  # noqa E501
    # 'tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/012/00001200/s003_2010_12_06/00001200_s003_t000.edf': b'0       00001200 M 01-JAN-1963 00001200 Age:47                                          ',  # noqa E501
    # 'tuh_abnormal_eeg/v2.0.0/edf/eval/abnormal/01_tcp_ar/059/00005932/s004_2013_03_14/00005932_s004_t000.edf': b'0       00005932 M 01-JAN-1963 00005932 Age:50                                          ',  # noqa E501
}

class _NMTMock(NMT):
    """Mocked class for testing and examples."""
    @mock.patch('glob.glob', return_value=_NMT_PATHS.keys())
    @mock.patch('mne.io.read_raw_edf', new=_fake_raw)
    @mock.patch('pandas.read_csv', new=_fake_pd_read_csv)
    @mock.patch('braindecode.datasets.tuh._read_edf_header',
                new=_get_header)
    @mock.patch('braindecode.datasets.tuh._read_physician_report',
                return_value='simple_test')
    def __init__(self, mock_glob, mock_report, path, recording_ids=None,
                 target_name='pathological', preload=False,
                 add_physician_reports=False, n_jobs=1):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Cannot save date file")
            super().__init__(path=path, recording_ids=recording_ids,
                             target_name=target_name, preload=preload,
                            #  add_physician_reports=add_physician_reports,
                             n_jobs=n_jobs)
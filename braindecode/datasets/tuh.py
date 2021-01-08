import re

import numpy as np
import pandas as pd
import mne

from torch.utils.data import Dataset

from .base import BaseDataset, BaseConcatDataset, read_all_file_names


class TUHAbnormal(BaseConcatDataset):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.

    Parameters
    ----------
    path: str
        parent directory of the dataset
    recording_ids: list(int) | int
        (list of) int of recording(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. provide recording_ids in ascending
        order to preserve chronological order)
    target_name: str
        can be "pathological", "gender", or "age"
    preload: bool
        if True, preload the data of the Raw objects.
    add_physician_reports: bool
        if True, the physician reports will be read from disk and added to the
        description
    """
    def __init__(self, path, recording_ids=None, target_name="pathological",
                 preload=False, add_physician_reports=False):
        all_file_paths = read_all_file_names(path, extension=".edf")
        all_file_paths = self.sort_chronologically(all_file_paths)
        if recording_ids is None:
            recording_ids = np.arange(len(all_file_paths))

        all_base_ds = []
        for i, recording_id in enumerate(recording_ids):
            file_path = all_file_paths[recording_id]
            raw = mne.io.read_raw_edf(file_path, preload=preload)
            pathological, train_or_eval, subject_id = (
                self._parse_properties_from_file_path(file_path))
            age, gender = _parse_age_and_gender_from_edf_header(file_path)
            # see https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/_AAREADME.txt
            d = {'age': age, 'pathological': pathological, 'gender': gender,
                 'train_or_eval': train_or_eval, 'subject': subject_id,
                 'recording_id': recording_id}
            if add_physician_reports:
                report_path = "_".join(file_path.split("_")[:-1]) + ".txt"
                with open(report_path, "r", encoding="latin-1") as f:
                    physician_report = f.read()
                d["physician_report"] = physician_report
            description = pd.Series(d, name=i)
            base_ds = BaseDataset(raw, description, target_name=target_name)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)


    @staticmethod
    def sort_chronologically(file_paths):
        """Use pandas groupby to sort the recordings chronologically.

        Parameters
        ----------
        file_paths: list(str)
            a list of all file paths to be sorted
        Returns
        -------
            sorted_file_paths: list(str)
            a list of all file paths sorted chronologically
        """
        # expect filenames as v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/s004_2013_08_15/00000021_s004_t000.edf
        #              version/file type/data_split/label/EEG reference/subset/subject/recording session/file
        # see https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/_AAREADME.txt
        path_splits = [fp.split("/") for fp in file_paths]
        identifiers = [[path_split[-3]] +
                       path_split[-2].split("_") +
                       [path_split[-1].split("_")[-1].split(".")[0]]
                       for path_split in path_splits]
        df = pd.DataFrame(
            identifiers,
            columns=["subject", "session", "year", "month", "day", "token"])
        df = pd.concat([group for name, group in df.groupby(
            ["year", "month", "day", "subject", "session", "token"])])
        return [file_paths[i] for i in df.index]


    @staticmethod
    def _parse_properties_from_file_path(file_path):
        # expect filenames as v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/s004_2013_08_15/00000021_s004_t000.edf
        #              version/file type/data_split/label/EEG reference/subset/subject/recording session/file
        # see https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/_AAREADME.txt
        path_splits = file_path.split("/")
        pathological = path_splits[-6]
        assert pathological in ["abnormal", "normal"]
        train_or_eval = path_splits[-7]
        assert train_or_eval in ["train", "eval"]
        subject_id = path_splits[-3]
        # subject id is string of length 8 with leading zeros
        assert len(subject_id) == 8
        return pathological == "abnormal", train_or_eval, int(subject_id)


def _parse_age_and_gender_from_edf_header(file_path, return_raw_header=False):
    f = open(file_path, "rb")
    content = f.read(88)
    f.close()
    if return_raw_header:
        return content
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = content[8:].decode("ascii")
    assert "F" in patient_id or "M" in patient_id
    assert "Age" in patient_id
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender


class TUHIndexDataset(Dataset):
    """A class to index the EDF files of the TUH EEG Corpus.

    Params
    ------
    path: str
        The parent directory of the edf files.
    target_name: str
        The name of the target. For now can only be 'age' or 'gender'.
    """
    def __init__(self, path, target_name=None):
        file_paths = read_all_file_names(path, ".edf")
        self.description = pd.DataFrame({"path": file_paths})
        # TODO: build up the description based on information accessible without loading, i.e. reference, subject id etc
        self.target_name = target_name

    def __getitem__(self, idx):
        path = self.description.loc[idx, "path"]
        raw = mne.io.read_raw_edf(path)
        age, gender = _parse_age_and_gender_from_edf_header(path)
        description = {**{"age": age, "gender": gender}, **self.description.iloc[idx].to_dict()}
        return BaseConcatDataset([BaseDataset(raw, description, target_name=self.target_name)])

    def __len__(self):
        return len(self.description)

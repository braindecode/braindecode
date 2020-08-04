import re
import glob

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset


class TUHAbnormal(BaseConcatDataset):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.

    Parameters
    ----------
    path: str
        parent directory of the dataset
    recording_ids: list(int)
        list of int of recording(s) to be read
    target_name: str
        can be "pathological", "gender", or "age"
    preload: bool
        if True, preload the data of the Raw objects
    """
    def __init__(self, path, recording_ids=None, target_name="pathological",
                 preload=False):
        all_file_paths = read_all_file_names(path, extension=".edf")
        all_file_paths = self.sort_chronologically(all_file_paths)
        if recording_ids is None:
            recording_ids = np.arange(len(all_file_paths))

        all_base_ds = []
        for recording_id in recording_ids:
            file_path = all_file_paths[recording_id]
            raw = mne.io.read_raw_edf(file_path, preload=preload)
            pathological, train_or_eval, subject_id = (
                self._parse_properties_from_file_path(file_path))
            age, gender = _parse_age_and_gender_from_edf_header(file_path)
            description = pd.Series(
                {"age": age, "pathological": pathological, "gender": gender,
                "train_or_eval": train_or_eval, "subject": subject_id},
                name=recording_id)
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


def read_all_file_names(directory, extension):
    """Read all files with specified extension from given path and sorts them
    based on a given sorting key.

    Parameters
    ----------
    directory: str
        parent directory to be searched for files of the specified type
    extension: str
        file extension, i.e. ".edf" or ".txt"

    Returns
    -------
    file_paths: list(str)
        a list to all files found in (sub)directories of path
    """
    assert extension.startswith(".")
    file_paths = glob.glob(directory + "**/*" + extension, recursive=True)
    assert len(file_paths) > 0, (
        f"something went wrong. Found no {extension} files in {directory}")
    return file_paths


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
    [age] = re.findall(r"Age:(\d+)", patient_id)
    [gender] = re.findall(r"\s([F|M])\s", patient_id)
    return int(age), gender

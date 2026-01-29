"""
Dataset classes for the Temple University Hospital (TUH) EEG Corpus and the.

TUH Abnormal EEG Corpus.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import glob
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal
from unittest import mock

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from braindecode.datasets.base import BaseConcatDataset, RawDataset


class TUH(BaseConcatDataset):
    """Temple University Hospital (TUH) EEG Corpus.

    (www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg).

    Parameters
    ----------
    path : str
        Parent directory of the dataset.
    recording_ids : list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name : str
        Can be 'gender', or 'age'.
    preload : bool
        If True, preload the data of the Raw objects.
    add_physician_reports : bool
        If True, the physician reports will be read from disk and added to the
        description.
    rename_channels : bool
        If True, rename the EEG channels to the standard 10-05 system.
    set_montage : bool
        If True, set the montage to the standard 10-05 system.
    on_missing_files : Literal['warn', 'raise']
        Behavior when the number of files found in the dataset directory
        does not match the expected number of files.
    version: Literal['v1.1.0', 'v1.2.0']
        Version of the TUH EEG Corpus to use. Currently only v1.1.0 and v1.2.0 are supported.
    n_jobs : int
        Number of jobs to be used to read files in parallel.
    """

    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = None,
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        on_missing_files: Literal["warn", "raise"] = "raise",
        version: Literal["v1.1.0", "v1.2.0", "v2.0.1"] = "v2.0.1",
        n_jobs: int = 1,
    ):
        self.version = version
        if on_missing_files not in ["warn", "raise"]:
            raise ValueError(
                "on_missing_files must be either 'warn' or 'raise', "
                f"got {on_missing_files}."
            )
        if set_montage:
            assert rename_channels, (
                "If set_montage is True, rename_channels must be True."
            )
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(
            os.path.join(path, f"{version}/**/*.edf"), recursive=True
        )

        # check files count
        if self._expected_files_count is None:
            warnings.warn(
                f"Could not verify that the number of files in {self.__class__.__name__} "
                "dataset is correct. If you have the dataset completely downloaded, "
                "please open an issue to add the expected number of files for this version."
            )
        elif (files_count := len(file_paths)) != self._expected_files_count:
            msg = (
                f"Expected {self._expected_files_count} files but found "
                f"{files_count} files in {path} for "
                f"{self.__class__.__name__} dataset. The dataset might be "
                "incomplete."
            )
            if on_missing_files == "raise":
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        descriptions = _create_description(file_paths, self.version, self._ds_name)
        # sort the descriptions chronologicaly
        descriptions = _sort_chronologically(descriptions)
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            if not isinstance(recording_ids, Iterable):
                # Assume it is an integer specifying number
                # of recordings to load
                recording_ids = range(recording_ids)
            descriptions = descriptions[recording_ids]

        # workaround to ensure warnings are suppressed when running in parallel
        def create_dataset(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*not in description. '__getitem__'"
                )
                return self._create_dataset(*args, **kwargs)

        # this is the second loop (slow)
        # create datasets gathering more info about the files touching them
        # reading the raws and potentially preloading the data
        # disable joblib for tests. mocking seems to fail otherwise
        if n_jobs == 1:
            base_datasets = [
                create_dataset(
                    descriptions[i],
                    target_name,
                    preload,
                    add_physician_reports,
                    rename_channels,
                    set_montage,
                )
                for i in descriptions.columns
            ]
        else:
            base_datasets = Parallel(n_jobs)(
                delayed(create_dataset)(
                    descriptions[i],
                    target_name,
                    preload,
                    add_physician_reports,
                    rename_channels,
                    set_montage,
                )
                for i in descriptions.columns
            )
        super().__init__(base_datasets)

    @property
    def _expected_files_count(self) -> int | None:
        return None

    @property
    def _ds_name(self) -> str:
        return "tuh"

    @staticmethod
    def _rename_channels(raw):
        """
        Renames the EEG channels using mne conventions and sets their type to 'eeg'.

        See https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/
        """
        # remove ref suffix and prefix:
        # TODO: replace with removesuffix and removeprefix when 3.8 is dropped
        mapping_strip = {
            c: c.replace("-REF", "").replace("-LE", "").replace("EEG ", "")
            for c in raw.ch_names
        }
        raw.rename_channels(mapping_strip)

        montage1005 = mne.channels.make_standard_montage("standard_1005")
        mapping_eeg_names = {
            c.upper(): c for c in montage1005.ch_names if c.upper() in raw.ch_names
        }

        # Set channels whose type could not be inferred (defaulted to "eeg") to "misc":
        non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
        if non_eeg_names:
            non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
            mapping_non_eeg_types = {
                c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
            }
            if mapping_non_eeg_types:
                raw.set_channel_types(mapping_non_eeg_types)

        if mapping_eeg_names:
            # Set 1005 channels type to "eeg":
            raw.set_channel_types(
                {c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore"
            )
            # Fix capitalized EEG channel names:
            raw.rename_channels(mapping_eeg_names)

    @staticmethod
    def _set_montage(raw):
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

    @staticmethod
    def _create_dataset(
        description,
        target_name,
        preload,
        add_physician_reports,
        rename_channels,
        set_montage,
    ):
        file_path = description.loc["path"]

        # parse age and gender information from EDF header
        age, gender = _parse_age_and_gender_from_edf_header(file_path)
        raw = mne.io.read_raw_edf(
            file_path, preload=preload, infer_types=True, verbose="error"
        )
        if rename_channels:
            TUH._rename_channels(raw)
        if set_montage:
            TUH._set_montage(raw)

        meas_date = (
            datetime(1, 1, 1, tzinfo=timezone.utc)
            if raw.info["meas_date"] is None
            else raw.info["meas_date"]
        )
        # if this is old version of the data and the year could be parsed from
        # file paths, use this instead as before
        if "year" in description:
            meas_date = meas_date.replace(*description[["year", "month", "day"]])
        raw.set_meas_date(meas_date)

        d = {
            "age": int(age),
            "gender": gender,
        }
        # if year exists in description = old version
        # if not, get it from meas_date in raw.info and add to description
        # if meas_date is None, create fake one
        if "year" not in description:
            d["year"] = raw.info["meas_date"].year
            d["month"] = raw.info["meas_date"].month
            d["day"] = raw.info["meas_date"].day

        # read info relevant for preprocessing from raw without loading it
        if add_physician_reports:
            physician_report = _read_physician_report(file_path)
            d["report"] = physician_report
        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        base_dataset = RawDataset(raw, description, target_name=target_name)
        return base_dataset


def _create_description(file_paths, version, ds_name):
    descriptions = [
        _parse_description_from_file_path(f, version, ds_name) for f in file_paths
    ]
    descriptions = pd.DataFrame(descriptions)
    return descriptions.T


def _sort_chronologically(descriptions):
    # last resort, we use path to sort (always available):
    sort_cols = ["year", "month", "day", "subject", "session", "segment", "path"]
    available_cols = [col for col in sort_cols if col in descriptions.index]
    descriptions.sort_values(available_cols, axis=1, inplace=True)
    return descriptions


def _read_date(file_path):
    date_path = file_path.replace(".edf", "_date.txt")
    # if date file exists, read it
    if os.path.exists(date_path):
        description = pd.read_json(date_path, typ="series").to_dict()
    # otherwise read edf file, extract date and store to file
    else:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose="error")
        meas_date = raw.info["meas_date"]

        description = (
            {
                "year": meas_date.year,
                "month": meas_date.month,
                "day": meas_date.day,
            }
            if meas_date is not None
            else {}  # saving and returning an empty date to avoid re-reading every time
        )
        # if the txt file storing the recording date does not exist, create it
        try:
            pd.Series(description).to_json(date_path)
        except OSError:
            warnings.warn(
                f"Cannot save date file to {date_path}. "
                f"This might slow down creation of the dataset."
            )
    return description


def _parse_description_from_file_path(
    file_path, version, ds_name: Literal["tuh", "abnormal", "events"]
):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    msg = f"Unexpected filename format '{file_path}'; return minimal description"
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # If an absolute path on POSIX the first token is empty string; drop it
    if tokens and tokens[0] == "":
        tokens = tokens[1:]

    # Extract version number
    v_match = re.match(r"v(\d+)\.\d+\.\d+", version)
    if v_match is None:
        raise ValueError(
            f"Could not parse version number from version string {version}."
        )
    v_number = int(v_match.group(1))

    if (ds_name == "abnormal" and v_number >= 3) or (
        (ds_name == "tuh") and v_number >= 2
    ):
        # New file path structure for versions after december 2022,
        # expect file paths as
        # tuh_eeg/v2.0.0/edf/000/aaaaaaaa/
        #     s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf
        # or for abnormal:
        # tuh_eeg_abnormal/v3.0.0/edf/train/normal/
        #     01_tcp_ar/aaaaaaav_s004_t000.edf
        subject_id = tokens[-1].split("_")[0]
        session = tokens[-1].split("_")[1]
        segment = tokens[-1].split("_")[2].split(".")[0]
        description = _read_date(file_path)
        description.update(
            {
                "path": file_path,
                "version": version,
                "subject": subject_id,
                "session": int(session[1:]),
                "segment": int(segment[1:]),
            }
        )
        if ds_name != "abnormal":
            year, month, day = tokens[-3].split("_")[1:]
            description["year"] = int(year)
            description["month"] = int(month)
            description["day"] = int(day)
        return description
    elif ds_name == "events" and v_number >= 2:
        # Train path example: tuh_eeg_events/v2.0.1/edf/train/aaaaaaar/aaaaaaar_00000001.edf"
        # Eval path example: tuh_eeg_events/v2.0.1/edf/eval/000/bckg_000_a_.edf
        split_name = tokens[-3]
        if split_name not in {"train", "eval"}:
            warnings.warn(msg)
            return {"path": file_path, "version": version}
        base_name = tokens[-1]
        regex = (
            r"(?P<subject_id>[a-z]+)_(?P<session>\d+).edf"
            if split_name == "train"
            else r"(?P<event_prefix>[a-z]+)_(?P<subject_id>\d+?)_a_(?P<run>\d*).edf"
        )
        match = re.match(regex, base_name)
        if match is None:
            warnings.warn(msg)
            return {"path": file_path, "version": version}
        description = _read_date(file_path)
        session = int(match.group("session")) if "session" in match.groupdict() else 1
        event_prefix = (
            match.group("event_prefix") if "event_prefix" in match.groupdict() else None
        )
        run = match.group("run") if "run" in match.groupdict() else 0
        if run is not None:
            run = 0 if run == "" else int(run)
        description.update(
            {
                "path": file_path,
                "version": version,
                "subject": match.group("subject_id"),
                "session": session,
                "split": split_name,
                "event_prefix": event_prefix,
                "run": run,
            }
        )
        return description
    else:  # Old file path structure
        # expect file paths as tuh_eeg/version/file_type/reference/data_split/
        #                          subject/recording session/file
        # e.g.                 tuh_eeg/v1.1.0/edf/01_tcp_ar/027/00002729/
        #                          s001_2006_04_12/00002729_s001.edf
        # or for abnormal
        # version/file type/data_split/pathology status/
        #     reference/subset/subject/recording session/file
        # v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/
        #     s004_2013_08_15/00000021_s004_t000.edf
        subject_id = tokens[-1].split("_")[0]
        session = tokens[-2].split("_")[0]  # string on format 's000'
        # According to the example path in the comment 8 lines above,
        # segment is not included in the file name
        segment = tokens[-1].split("_")[-1].split(".")[0]  # TODO: test with tuh_eeg
        year, month, day = tokens[-2].split("_")[1:]
        return {
            "path": file_path,
            "version": version,
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "subject": int(subject_id),
            "session": int(session[1:]),
            "segment": int(segment[1:]),
        }


def _read_physician_report(file_path):
    directory = os.path.dirname(file_path)
    txt_file = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    # check that there is at most one txt file in the same directory
    assert len(txt_file) in [0, 1]
    report = ""
    if txt_file:
        txt_file = txt_file[0]
        # somewhere in the corpus, encoding apparently changed
        # first try to read as utf-8, if it does not work use latin-1
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                report = f.read()
        except UnicodeDecodeError:
            with open(txt_file, "r", encoding="latin-1") as f:
                report = f.read()
    if not report:
        raise RuntimeError(
            f"Could not read physician report ({txt_file}). "
            f"Disable option or choose appropriate directory."
        )
    return report


def _read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def _parse_age_and_gender_from_edf_header(file_path):
    header = _read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender


class TUHAbnormal(TUH):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.

    see https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml#c_tuab

    Parameters
    ----------
    path : str
        Parent directory of the dataset.
    recording_ids : list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name : str
        Can be 'pathological', 'gender', or 'age'.
    preload : bool
        If True, preload the data of the Raw objects.
    add_physician_reports : bool
        If True, the physician reports will be read from disk and added to the
        description.
    rename_channels : bool
        If True, rename the EEG channels to the standard 10-05 system.
    set_montage : bool
        If True, set the montage to the standard 10-05 system.
    on_missing_files : Literal["warn", "raise"]
        Behavior when the number of files found in the dataset directory
        does not match the expected number of files.
    version: Literal['v2.0.0']
        Version of the TUH Abnormal EEG Corpus to use. Currently only 'v2.0.0' is supported.
    n_jobs : int
        Number of jobs to be used to read files in parallel.
    """

    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = "pathological",
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        on_missing_files: Literal["warn", "raise"] = "raise",
        version: Literal["v2.0.0", "v3.0.1"] = "v3.0.1",
        n_jobs: int = 1,
    ):
        super().__init__(
            path=path,
            recording_ids=recording_ids,
            preload=preload,
            target_name=target_name,
            add_physician_reports=add_physician_reports,
            rename_channels=rename_channels,
            set_montage=set_montage,
            on_missing_files=on_missing_files,
            version=version,  # type: ignore[arg-type]
            n_jobs=n_jobs,
        )
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = self._parse_additional_description_from_file_path(
                file_path
            )
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)
        # expect paths as version/file type/data_split/pathology status/
        #                     reference/subset/subject/recording session/file
        # e.g.            v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/
        #                     s004_2013_08_15/00000021_s004_t000.edf
        assert "abnormal" in tokens or "normal" in tokens, "No pathology labels found."
        assert "train" in tokens or "eval" in tokens, (
            "No train or eval set information found."
        )
        return {
            "version": tokens[-9],
            "train": "train" in tokens,
            "pathological": "abnormal" in tokens,
        }

    @property
    def _expected_files_count(self):
        if self.version == "v3.0.1":
            # dataset.description.groupby(["train", "pathological"]).path.count()
            # train  pathological
            # False  False            150
            #        True             126
            # True   False           1371
            #        True            1346
            return 2993
        return None

    @property
    def _ds_name(self):
        return "abnormal"


# mapping for numeric rec labels
EVENTS_MAP = {
    "1": "spsw",
    "2": "gped",
    "3": "pled",
    "4": "eyem",
    "5": "artf",
    "6": "bckg",
}


class TUHEvents(TUH):
    """Temple University Hospital (TUH) EEG Event Corpus.

    This is a thin wrapper around :class:`TUH` that extracts event annotations
    (from accompanying ``.rec`` or ``.lab`` files when available) and stores
    them in the dataset description under the ``events`` key.

    The dataset layout follows the TUH EEG Event Corpus structure (train/eval
    folders) as documented in the corpus README.

    see https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml#c_tuev

    The event names are mapped as follows:

    - '1' -> 'spsw' (spike and slow wave)
    - '2' -> 'gped' (generalized periodic epileptiform discharge)
    - '3' -> 'pled' (periodic lateralized epileptiform discharge)
    - '4' -> 'eyem' (eye movement)
    - '5' -> 'artf' (artifact)
    - '6' -> 'bckg' (background)

    Parameters
    ----------
    path : str
        Parent directory of the dataset.
    recording_ids : list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name : str | None
        Can be 'gender', 'age', or None. Use None when using events labels as target.
    preload : bool
        If True, preload the data of the Raw objects.
    add_physician_reports : bool
        If True, the physician reports will be read from disk and added to the
        description.
    rename_channels : bool
        If True, rename the EEG channels to the standard 10-05 system.
    set_montage : bool
        If True, set the montage to the standard 10-05 system.
    channel_events : bool
        If True, add channel-specific event annotations to the Raw objects.
        If False, only global events will be added.
    merge_events : bool
        If True, merge consecutive identical events into a single event with
        duration covering the entire span.
    on_missing_files : Literal["warn", "raise"]
        Behavior when the number of files found in the dataset directory
        does not match the expected number of files.
    version: Literal['v2.0.1']
        Version of the TUH Events EEG Corpus to use. Currently only 'v2.0.1' is supported.
    n_jobs : int
        Number of jobs to be used to read files in parallel.
    """

    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = None,
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        channel_events: bool = False,
        merge_events: bool = True,
        on_missing_files: Literal["warn", "raise"] = "raise",
        version: Literal["v2.0.1"] = "v2.0.1",
        n_jobs: int = 1,
    ):
        if merge_events and channel_events:
            raise NotImplementedError(
                "Merging events is not implemented for channel-specific events."
            )
        super().__init__(
            path=path,
            recording_ids=recording_ids,
            preload=preload,
            target_name=target_name,
            add_physician_reports=add_physician_reports,
            rename_channels=rename_channels,
            set_montage=set_montage,
            on_missing_files=on_missing_files,
            version=version,  # type: ignore[arg-type]
            n_jobs=n_jobs,
        )

        for ds in self.datasets:
            file_path = ds.description.loc["path"]
            self._set_annotations(
                file_path,
                ds.raw,
                channel_events=channel_events,
                merge_events=merge_events,
            )

    @property
    def _expected_files_count(self):
        if self.version == "v2.0.1":
            return 518  # 159 (eval) + 359 (train)
        return None

    @property
    def _ds_name(self):
        return "events"

    @staticmethod
    def _set_annotations(
        file_path: str | Path, raw: mne.io.Raw, channel_events: bool, merge_events: bool
    ):
        """Parse event annotations for a single EDF file.
        Then set the annotations in the provided raw object."""
        file_path = Path(file_path)
        rec_path = file_path.parent / (file_path.stem + ".rec")
        ch_names = raw.ch_names

        if not rec_path.exists():
            warnings.warn(
                f"File '{rec_path}' does not exist. No event annotations will be added."
            )
            return None
        onsets = []
        durations = []
        descriptions = []
        ch_name_list: list[list[str]] = []
        lines = rec_path.read_text().splitlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) != 4:
                warnings.warn(
                    f"Unexpected format in '{rec_path}': '{line}'. No event annotations will be added."
                )
                return None
            channel_idx, start, stop, event_code = parts
            if event_code not in EVENTS_MAP:
                warnings.warn(
                    f"Unknown event code '{event_code}' in '{rec_path}'. No event annotations will be added."
                )
                return None
            onsets.append(float(start))
            durations.append(float(stop) - float(start))
            descriptions.append(EVENTS_MAP[event_code])
            ch_name_list.append([ch_names[int(channel_idx)]])

        # remove duplicates
        if not channel_events:
            unique_events = set(zip(onsets, durations, descriptions))
            onsets, durations, descriptions = zip(*unique_events)  # type: ignore[assignment]
        # merge consecutive events
        if merge_events:
            assert not channel_events
            merged_onsets: list[float] = []
            merged_durations: list[float] = []
            merged_descriptions: list[str] = []
            sorted_events = sorted(zip(onsets, durations, descriptions))
            unique_descriptions = set(descriptions)
            for description in unique_descriptions:
                new = True
                for onset, duration, this_description in sorted_events:
                    if this_description != description:
                        continue
                    if new or (onset > merged_onsets[-1] + merged_durations[-1]):
                        # initialize new event
                        merged_onsets.append(onset)
                        merged_durations.append(duration)
                        merged_descriptions.append(this_description)
                    else:
                        # update last event duration
                        merged_durations[-1] = (onset + duration) - merged_onsets[-1]
                    new = False
            onsets = merged_onsets
            durations = merged_durations
            descriptions = merged_descriptions
        # sort by onset time
        sorted_events = sorted(zip(onsets, durations, descriptions))
        onsets, durations, descriptions = zip(*sorted_events)  # type: ignore[assignment]

        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            ch_names=ch_name_list if channel_events else None,
        )
        raw.set_annotations(annotations)


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


def _get_header(*args, **kwargs):
    all_paths = {}
    for paths_dict in [_TUH_EEG_PATHS, _TUH_EEG_ABNORMAL_PATHS, _TUH_EEG_EVENTS_PATHS]:
        for version_dict in paths_dict.values():
            all_paths.update(version_dict)
    return all_paths[args[0]]


_TUH_EEG_PATHS = {
    "v1.1.0": {
        "tuh_eeg/v1.1.0/edf/01_tcp_ar/099/00009932/s004_2014_09_30/00009932_s004_t013.edf": b"0       00009932 F 01-JAN-1961 00009932 Age:53                                          ",
        # noqa E501
        # These are actual file paths and edf headers from the TUH EEG Corpus (v1.1.0 and v1.2.0)
        "tuh_eeg/v1.1.0/edf/01_tcp_ar/000/00000000/s001_2015_12_30/00000000_s001_t000.edf": b"0       00000000 M 01-JAN-1978 00000000 Age:37                                          ",
        # noqa E501
        "tuh_eeg/v1.1.0/edf/02_tcp_le/000/00000058/s001_2003_02_05/00000058_s001_t000.edf": b"0       00000058 M 01-JAN-2003 00000058 Age:0.0109                                      ",
        # noqa E501
        "tuh_eeg/v1.1.0/edf/03_tcp_ar_a/123/00012331/s003_2014_12_14/00012331_s003_t002.edf": b"0       00012331 M 01-JAN-1975 00012331 Age:39                                          ",
        # noqa E501
    },
    "v1.2.0": {
        "tuh_eeg/v1.2.0/edf/03_tcp_ar_a/149/00014928/s004_2016_01_15/00014928_s004_t007.edf": b"0       00014928 F 01-JAN-1933 00014928 Age:83                                          ",
        # noqa E501
    },
}
_TUH_EEG_ABNORMAL_PATHS = {
    "v2.0.0": {
        # these are actual file paths and edf headers from TUH Abnormal EEG Corpus (v2.0.0)
        "tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/078/00007871/s001_2011_07_05/00007871_s001_t001.edf": b"0       00007871 F 01-JAN-1988 00007871 Age:23                                          ",
        # noqa E501
        "tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/097/00009777/s001_2012_09_17/00009777_s001_t000.edf": b"0       00009777 M 01-JAN-1986 00009777 Age:26                                          ",
        # noqa E501
        "tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/083/00008393/s002_2012_02_21/00008393_s002_t000.edf": b"0       00008393 M 01-JAN-1960 00008393 Age:52                                          ",
        # noqa E501
        "tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/012/00001200/s003_2010_12_06/00001200_s003_t000.edf": b"0       00001200 M 01-JAN-1963 00001200 Age:47                                          ",
        # noqa E501
        "tuh_abnormal_eeg/v2.0.0/edf/eval/abnormal/01_tcp_ar/059/00005932/s004_2013_03_14/00005932_s004_t000.edf": b"0       00005932 M 01-JAN-1963 00005932 Age:50                                          ",
        # noqa E501
    }
}
_TUH_EEG_EVENTS_PATHS = {
    "v2.0.1": {
        "tuh_eeg_events/v2.0.1/edf/eval/000/bckg_000_a_.edf": b"0       000 F 01-JAN-0000 000 Age:36                                                    ",  # noaq E501
        "tuh_eeg_events/v2.0.1/edf/train/aaaaaaar/aaaaaaar_00000001.edf": b"0       /aa F 01-JAN-0000 /aa Age:19                                                    ",  # noqa E501
        "tuh_eeg_events/v2.0.1/edf/eval/001/pled_001_a_2.edf": b"0       001 F 01-JAN-0000 001 Age:68                                                    ",  # noqa E501
    }
}


class _MockGlob:
    def __init__(self, paths):
        self.paths = paths

    def __call__(self, pattern, *args, **kwargs):
        # Find the version
        for version in self.paths.keys():
            if version in pattern:
                return list(self.paths[version].keys())
        raise ValueError(
            f"CCould not find a known version in pattern: {pattern=}, known versions: {list(self.paths.keys())}"
        )


class _TUHMock(TUH):
    """Mocked class for testing and examples."""

    @mock.patch("glob.glob", new=_MockGlob(_TUH_EEG_PATHS))
    @mock.patch("mne.io.read_raw_edf", new=_fake_raw)
    @mock.patch("braindecode.datasets.tuh._read_edf_header", new=_get_header)
    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = None,
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        on_missing_files: Literal["warn", "raise"] = "raise",
        version: Literal["v1.1.0", "v1.2.0"] = "v1.2.0",
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot save date file")
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                target_name=target_name,
                preload=preload,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                on_missing_files=on_missing_files,
                version=version,
                n_jobs=n_jobs,
            )

    @property
    def _expected_files_count(self):
        if self.version == "v1.1.0":
            return 4
        if self.version == "v1.2.0":
            return 1
        return None


class _TUHAbnormalMock(TUHAbnormal):
    """Mocked class for testing and examples."""

    @mock.patch("glob.glob", new=_MockGlob(_TUH_EEG_ABNORMAL_PATHS))
    @mock.patch("mne.io.read_raw_edf", new=_fake_raw)
    @mock.patch("braindecode.datasets.tuh._read_edf_header", new=_get_header)
    @mock.patch(
        "braindecode.datasets.tuh._read_physician_report", return_value="simple_test"
    )
    def __init__(
        self,
        mock_report,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = "pathological",
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        on_missing_files: Literal["warn", "raise"] = "raise",
        version: Literal["v2.0.0"] = "v2.0.0",
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot save date file")
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                target_name=target_name,
                preload=preload,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                on_missing_files=on_missing_files,
                version=version,
                n_jobs=n_jobs,
            )

    @property
    def _expected_files_count(self):
        if self.version == "v2.0.0":
            return 5
        return None


class _TUHEventsMock(TUHEvents):
    """Mocked class for testing and examples."""

    @mock.patch("glob.glob", new=_MockGlob(_TUH_EEG_EVENTS_PATHS))
    @mock.patch("mne.io.read_raw_edf", new=_fake_raw)
    @mock.patch("braindecode.datasets.tuh._read_edf_header", new=_get_header)
    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = "pathological",
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        on_missing_files: Literal["warn", "raise"] = "raise",
        version: Literal["v2.0.1"] = "v2.0.1",
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot save date file")
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                target_name=target_name,
                preload=preload,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                on_missing_files=on_missing_files,
                version=version,
                n_jobs=n_jobs,
            )

    @property
    def _expected_files_count(self):
        if self.version == "v2.0.1":
            return 3
        return None

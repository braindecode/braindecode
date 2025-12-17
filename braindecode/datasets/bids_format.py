"""
BIDS-like format utilities for Hub integration.

This module provides BIDS-compatible structures for storing preprocessed EEG
data in the derivatives folder format while maintaining efficiency for training.
It leverages mne_bids for BIDS path handling and metadata generation.

The format follows BIDS derivatives conventions:
- derivatives/<pipeline-name>/
  - dataset_description.json
  - participants.tsv
  - sub-<label>/
    - [ses-<label>/]
      - eeg/
        - sub-<label>_[ses-<label>_]task-<label>_events.tsv
        - sub-<label>_[ses-<label>_]task-<label>_channels.tsv
        - sub-<label>_[ses-<label>_]task-<label>_desc-preproc_eeg.zarr/

References:
- BIDS Derivatives: https://bids-specification.readthedocs.io/en/stable/derivatives/
- BIDS EEG: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Kuntal Kokate
#
# License: BSD (3-clause)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import mne
import mne_bids
import pandas as pd

import braindecode

# Default pipeline name for braindecode derivatives
DEFAULT_PIPELINE_NAME = "braindecode"


def description_to_bids_path(
    description: pd.Series,
    root: Union[str, Path],
    datatype: str = "eeg",
    suffix: str = "eeg",
    extension: str = ".zarr",
    desc_label: str = "preproc",
    check: bool = False,
) -> mne_bids.BIDSPath:
    """
    Create a BIDSPath from a dataset description.

    Uses mne_bids.BIDSPath for proper BIDS path handling.

    Parameters
    ----------
    description : pd.Series
        Dataset description containing BIDS entities.
    root : str | Path
        Root directory of the BIDS dataset.
    datatype : str
        Data type (eeg, meg, etc.).
    suffix : str
        BIDS suffix.
    extension : str
        File extension.
    desc_label : str
        Description label for derivatives.
    check : bool
        Whether to enforce BIDS conformity.

    Returns
    -------
    mne_bids.BIDSPath
        BIDS path object.
    """
    # Extract BIDS entities from description
    entities = _extract_bids_entities(description)

    # Create BIDSPath using mne_bids
    bids_path = mne_bids.BIDSPath(
        root=root,
        subject=entities.get("subject", "unknown"),
        session=entities.get("session"),
        task=entities.get("task", "task"),
        acquisition=entities.get("acquisition"),
        run=entities.get("run"),
        processing=entities.get("processing"),
        recording=entities.get("recording"),
        space=entities.get("space"),
        description=desc_label,
        suffix=suffix,
        extension=extension,
        datatype=datatype,
        check=check,
    )

    return bids_path


def _extract_bids_entities(description: pd.Series) -> dict[str, Any]:
    """
    Extract BIDS entities from a dataset description.

    Parameters
    ----------
    description : pd.Series
        Dataset description containing metadata.

    Returns
    -------
    dict
        Dictionary with BIDS entity keys.
    """
    if description is None or len(description) == 0:
        return {}

    # Common mappings from description keys to BIDS entities
    key_mappings = {
        "subject": "subject",
        "sub": "subject",
        "subject_id": "subject",
        "session": "session",
        "ses": "session",
        "task": "task",
        "run": "run",
        "acquisition": "acquisition",
        "acq": "acquisition",
        "processing": "processing",
        "proc": "processing",
        "recording": "recording",
        "rec": "recording",
        "space": "space",
        "split": "split",
        "description": "description",
        "desc": "description",
    }

    entities = {}
    for key in description.index:
        key_lower = str(key).lower()
        if key_lower in key_mappings:
            bids_key = key_mappings[key_lower]
            value = description[key]
            # Convert to string, handling None and NaN
            if pd.notna(value):
                # Clean up the value for BIDS compatibility
                str_value = str(value)
                # Remove any characters not allowed in BIDS entities
                str_value = "".join(c for c in str_value if c.isalnum() or c in "-_")
                if str_value:
                    entities[bids_key] = str_value

    return entities


def make_dataset_description(
    path: Union[str, Path],
    name: str = "Braindecode Dataset",
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    source_datasets: Optional[list[dict]] = None,
    overwrite: bool = True,
) -> Path:
    """
    Create a BIDS-compliant dataset_description.json for derivatives.

    Uses mne_bids.make_dataset_description for proper BIDS compliance.

    Parameters
    ----------
    path : str | Path
        Path to the derivatives directory.
    name : str
        Name of the dataset.
    pipeline_name : str
        Name of the pipeline that generated the derivatives.
    source_datasets : list of dict | None
        List of source dataset references.
    overwrite : bool
        Whether to overwrite existing file.

    Returns
    -------
    Path
        Path to the created dataset_description.json.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Use mne_bids to create the dataset description
    mne_bids.make_dataset_description(
        path=path,
        name=name,
        dataset_type="derivative",
        generated_by=[
            {
                "Name": "braindecode",
                "Version": braindecode.__version__,
                "CodeURL": "https://github.com/braindecode/braindecode",
            }
        ],
        source_datasets=source_datasets,
        overwrite=overwrite,
    )

    return path / "dataset_description.json"


def create_events_tsv(
    metadata: pd.DataFrame,
    sfreq: float,
    target_column: str = "target",
    extra_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create a BIDS-compliant events.tsv DataFrame from window metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        Window metadata containing i_start_in_trial, i_stop_in_trial, target columns.
    sfreq : float
        Sampling frequency in Hz.
    target_column : str
        Name of the column containing trial type/target information.
    extra_columns : list of str | None
        Additional columns from metadata to include in events.

    Returns
    -------
    pd.DataFrame
        BIDS-compliant events DataFrame.

    Notes
    -----
    The events.tsv file follows BIDS format with columns:
    - onset: Time of event onset in seconds
    - duration: Duration of event in seconds
    - trial_type: Name/label of the event type
    - sample: Sample index of event onset
    - value: Numeric value/target
    """
    events_data: dict[str, list[Any]] = {
        "onset": [],
        "duration": [],
        "trial_type": [],
        "sample": [],
        "value": [],
    }

    # Add extra columns
    extra_data: dict[str, list[Any]] = {col: [] for col in (extra_columns or [])}

    for idx, row in metadata.iterrows():
        # Calculate onset and duration from sample indices
        i_start = row.get("i_start_in_trial", 0)
        i_stop = row.get("i_stop_in_trial", i_start + 1)

        onset = i_start / sfreq
        duration = (i_stop - i_start) / sfreq

        # Get target/trial_type
        target = row.get(target_column, "n/a")
        trial_type = str(target) if pd.notna(target) else "n/a"

        events_data["onset"].append(onset)
        events_data["duration"].append(duration)
        events_data["trial_type"].append(trial_type)
        events_data["sample"].append(int(i_start))
        events_data["value"].append(target if pd.notna(target) else "n/a")

        # Add extra columns
        for col in extra_columns or []:
            extra_data[col].append(row.get(col, "n/a"))

    # Combine all data
    events_data.update(extra_data)

    return pd.DataFrame(events_data)


def create_participants_tsv(
    descriptions: list[pd.Series],
    extra_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create a BIDS-compliant participants.tsv from dataset descriptions.

    Parameters
    ----------
    descriptions : list of pd.Series
        List of dataset descriptions.
    extra_columns : list of str | None
        Additional columns to include from descriptions.

    Returns
    -------
    pd.DataFrame
        BIDS-compliant participants DataFrame.
    """
    participants_data: dict[str, list[Any]] = {
        "participant_id": [],
        "age": [],
        "sex": [],
        "hand": [],
    }

    # Add extra columns
    extra_data: dict[str, list[Any]] = {col: [] for col in (extra_columns or [])}

    seen_subjects = set()

    for desc in descriptions:
        if desc is None:
            continue

        # Get subject ID
        subject = None
        for key in ["subject", "sub", "subject_id"]:
            if key in desc.index and pd.notna(desc[key]):
                subject = str(desc[key])
                break

        if subject is None:
            continue

        # Skip duplicates
        if subject in seen_subjects:
            continue
        seen_subjects.add(subject)

        # Format as BIDS participant_id
        participant_id = f"sub-{subject}" if not subject.startswith("sub-") else subject

        # Get other info
        age = desc.get("age", "n/a")
        sex = desc.get("sex", desc.get("gender", "n/a"))
        hand = desc.get("hand", desc.get("handedness", "n/a"))

        participants_data["participant_id"].append(participant_id)
        participants_data["age"].append(age if pd.notna(age) else "n/a")
        participants_data["sex"].append(sex if pd.notna(sex) else "n/a")
        participants_data["hand"].append(hand if pd.notna(hand) else "n/a")

        # Add extra columns
        for col in extra_columns or []:
            extra_data[col].append(desc.get(col, "n/a"))

    # Combine all data
    participants_data.update(extra_data)

    return pd.DataFrame(participants_data)


def create_channels_tsv(
    info: "mne.Info",
    bad_channels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create a BIDS-compliant channels.tsv from MNE Info.

    Parameters
    ----------
    info : mne.Info
        MNE Info object containing channel information.
    bad_channels : list of str | None
        List of bad channel names.

    Returns
    -------
    pd.DataFrame
        BIDS-compliant channels DataFrame.
    """
    import mne

    bad_channels = bad_channels or info.get("bads", [])

    channels_data: dict[str, list[Any]] = {
        "name": [],
        "type": [],
        "units": [],
        "sampling_frequency": [],
        "low_cutoff": [],
        "high_cutoff": [],
        "status": [],
        "status_description": [],
    }

    # MNE channel type to BIDS type mapping
    type_mapping = {
        "eeg": "EEG",
        "ecg": "ECG",
        "eog": "EOG",
        "emg": "EMG",
        "meg": "MEG",
        "ref_meg": "MEGREF",
        "grad": "MEGGRADAXIAL",
        "mag": "MEGMAG",
        "stim": "TRIG",
        "misc": "MISC",
        "bio": "BIO",
    }

    sfreq = info["sfreq"]

    # Get filter info if available
    highpass = info.get("highpass", "n/a")
    lowpass = info.get("lowpass", "n/a")

    for i, ch_name in enumerate(info["ch_names"]):
        # Get channel type
        ch_type = mne.channel_type(info, i)
        bids_type = type_mapping.get(ch_type.lower(), ch_type.upper())

        # Get units (default to µV for EEG)
        if bids_type == "EEG":
            units = "µV"
        elif bids_type in ("MEG", "MEGMAG", "MEGGRADAXIAL"):
            units = "fT"
        else:
            units = "n/a"

        # Check if bad channel
        is_bad = ch_name in bad_channels
        status = "bad" if is_bad else "good"
        status_desc = "n/a"

        channels_data["name"].append(ch_name)
        channels_data["type"].append(bids_type)
        channels_data["units"].append(units)
        channels_data["sampling_frequency"].append(sfreq)
        channels_data["low_cutoff"].append(highpass if highpass != 0 else "n/a")
        channels_data["high_cutoff"].append(
            lowpass if lowpass != float("inf") else "n/a"
        )
        channels_data["status"].append(status)
        channels_data["status_description"].append(status_desc)

    return pd.DataFrame(channels_data)


def create_eeg_json_sidecar(
    info: "mne.Info",
    task_name: str = "unknown",
    task_description: Optional[str] = None,
    instructions: Optional[str] = None,
    institution_name: Optional[str] = None,
    manufacturer: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """
    Create a BIDS-compliant EEG sidecar JSON.

    Parameters
    ----------
    info : mne.Info
        MNE Info object.
    task_name : str
        Name of the task.
    task_description : str | None
        Description of the task.
    instructions : str | None
        Instructions given to the participant.
    institution_name : str | None
        Name of the institution.
    manufacturer : str | None
        Manufacturer of the EEG equipment.
    extra_metadata : dict | None
        Additional metadata.

    Returns
    -------
    dict
        Sidecar JSON content.
    """
    # Count channels by type
    ch_types = [info["chs"][i]["kind"] for i in range(len(info["ch_names"]))]
    eeg_count = sum(1 for k in ch_types if k == 2)  # EEG kind
    eog_count = sum(1 for k in ch_types if k == 202)  # EOG kind
    ecg_count = sum(1 for k in ch_types if k == 201)  # ECG kind
    emg_count = sum(1 for k in ch_types if k == 302)  # EMG kind

    sidecar = {
        "TaskName": task_name,
        "SamplingFrequency": info["sfreq"],
        "EEGChannelCount": eeg_count,
        "EOGChannelCount": eog_count,
        "ECGChannelCount": ecg_count,
        "EMGChannelCount": emg_count,
        "PowerLineFrequency": info.get("line_freq", "n/a"),
        "SoftwareFilters": {
            "HighpassFilter": {"CutoffFrequency": info.get("highpass", "n/a")},
            "LowpassFilter": {"CutoffFrequency": info.get("lowpass", "n/a")},
        },
    }

    if task_description:
        sidecar["TaskDescription"] = task_description
    if instructions:
        sidecar["Instructions"] = instructions
    if institution_name:
        sidecar["InstitutionName"] = institution_name
    if manufacturer:
        sidecar["Manufacturer"] = manufacturer

    if extra_metadata:
        sidecar.update(extra_metadata)

    return sidecar


def save_bids_sidecar_files(
    bids_path: mne_bids.BIDSPath,
    info: "mne.Info",
    metadata: Optional[pd.DataFrame] = None,
    sfreq: Optional[float] = None,
    task_name: str = "unknown",
) -> dict[str, Path]:
    """
    Save BIDS sidecar files for a recording using mne_bids BIDSPath.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        BIDS path object for the recording.
    info : mne.Info
        MNE Info object.
    metadata : pd.DataFrame | None
        Window metadata for events.tsv.
    sfreq : float | None
        Sampling frequency (if not in info).
    task_name : str
        Task name for sidecar JSON.

    Returns
    -------
    dict
        Dictionary mapping file types to their paths.
    """
    # Ensure directory exists
    bids_path.mkdir(exist_ok=True)

    saved_files = {}
    sfreq = sfreq or info["sfreq"]

    # Get the base path for sidecar files
    base_path = bids_path.copy()

    # Save events.tsv if metadata is available
    if metadata is not None and len(metadata) > 0:
        events_df = create_events_tsv(metadata, sfreq)
        events_path = base_path.copy().update(suffix="events", extension=".tsv")
        events_df.to_csv(events_path.fpath, sep="\t", index=False, na_rep="n/a")
        saved_files["events"] = events_path.fpath

    # Save channels.tsv
    channels_df = create_channels_tsv(info)
    channels_path = base_path.copy().update(suffix="channels", extension=".tsv")
    channels_df.to_csv(channels_path.fpath, sep="\t", index=False, na_rep="n/a")
    saved_files["channels"] = channels_path.fpath

    # Save EEG sidecar JSON
    sidecar = create_eeg_json_sidecar(info, task_name=task_name)
    sidecar_path = base_path.copy().update(suffix="eeg", extension=".json")
    with open(sidecar_path.fpath, "w") as f:
        json.dump(sidecar, f, indent=2)
    saved_files["sidecar"] = sidecar_path.fpath

    return saved_files


class BIDSDerivativesLayout:
    """
    Helper class for creating BIDS-like derivatives folder structure.

    This creates a structure compatible with BIDS derivatives while
    storing data in Zarr format for efficient training.

    Structure:
    derivatives/<pipeline>/
    ├── dataset_description.json
    ├── participants.tsv
    ├── sub-<label>/
    │   └── [ses-<label>/]
    │       └── eeg/
    │           ├── sub-<label>_task-<label>_desc-preproc_events.tsv
    │           ├── sub-<label>_task-<label>_desc-preproc_channels.tsv
    │           ├── sub-<label>_task-<label>_desc-preproc_eeg.json
    │           └── sub-<label>_task-<label>_desc-preproc_eeg.zarr/
    └── dataset.zarr (main data file for efficient loading)
    """

    def __init__(
        self,
        root: Union[str, Path],
        pipeline_name: str = DEFAULT_PIPELINE_NAME,
    ):
        """
        Initialize BIDS derivatives layout.

        Parameters
        ----------
        root : str | Path
            Root directory for derivatives.
        pipeline_name : str
            Name of the processing pipeline.
        """
        self.root = Path(root)
        self.pipeline_name = pipeline_name
        self.derivatives_dir = self.root / "derivatives" / pipeline_name

    def create_structure(self) -> Path:
        """Create the basic derivatives directory structure."""
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)
        return self.derivatives_dir

    def get_bids_path(
        self,
        description: pd.Series,
        suffix: str = "eeg",
        extension: str = ".zarr",
        desc_label: str = "preproc",
    ) -> mne_bids.BIDSPath:
        """
        Get a BIDSPath for a recording based on its description.

        Parameters
        ----------
        description : pd.Series
            Dataset description.
        suffix : str
            BIDS suffix.
        extension : str
            File extension.
        desc_label : str
            Description label.

        Returns
        -------
        mne_bids.BIDSPath
            BIDS path for the recording.
        """
        return description_to_bids_path(
            description=description,
            root=self.derivatives_dir,
            datatype="eeg",
            suffix=suffix,
            extension=extension,
            desc_label=desc_label,
            check=False,
        )

    def save_dataset_description(
        self,
        name: str = "Braindecode Dataset",
        source_datasets: Optional[list[dict]] = None,
    ) -> Path:
        """
        Save dataset_description.json for derivatives.

        Parameters
        ----------
        name : str
            Name of the dataset.
        source_datasets : list of dict | None
            Source dataset references.

        Returns
        -------
        Path
            Path to saved file.
        """
        return make_dataset_description(
            path=self.derivatives_dir,
            name=name,
            pipeline_name=self.pipeline_name,
            source_datasets=source_datasets,
            overwrite=True,
        )

    def save_participants(
        self,
        descriptions: list[pd.Series],
        extra_columns: Optional[list[str]] = None,
    ) -> Path:
        """
        Save participants.tsv file.

        Parameters
        ----------
        descriptions : list of pd.Series
            List of dataset descriptions.
        extra_columns : list of str | None
            Additional columns to include.

        Returns
        -------
        Path
            Path to saved file.
        """
        participants_df = create_participants_tsv(descriptions, extra_columns)
        participants_path = self.derivatives_dir / "participants.tsv"
        participants_df.to_csv(participants_path, sep="\t", index=False, na_rep="n/a")
        return participants_path

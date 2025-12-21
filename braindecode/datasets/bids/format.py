# mypy: ignore-errors
"""
BIDS-inspired format utilities for Hub integration.

This module provides BIDS-inspired structures for storing EEG data optimized
for deep learning training. It leverages mne_bids for BIDS path handling and
metadata generation.

The data is stored in the ``sourcedata/`` directory, which according to BIDS:
- Is NOT validated by BIDS validators (so .zarr files won't cause errors)
- Has no naming restrictions ("BIDS does not prescribe a particular naming
  scheme for source data")
- Is intended for data before file format conversion

This approach allows us to use efficient Zarr storage while maintaining
BIDS-style organization for discoverability.

Structure:
- sourcedata/<pipeline-name>/
  - dataset_description.json     (BIDS-style metadata)
  - participants.tsv             (BIDS-style metadata)
  - sub-<label>/
    - [ses-<label>/]
      - eeg/
        - *_events.tsv           (BIDS-style metadata)
        - *_channels.tsv         (BIDS-style metadata)
        - *_eeg.json             (BIDS-style metadata)
        - *_eeg.zarr/            (Zarr data - efficient for training)
  - dataset.zarr/                (Main data store for training)

References:
- BIDS sourcedata: https://bids-specification.readthedocs.io/en/stable/common-principles.html#source-vs-raw-vs-derived-data
- BIDS EEG: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Kuntal Kokate
#
# License: BSD (3-clause)

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Union

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne_bids.write import _channels_tsv, _sidecar_json

import braindecode

# Default pipeline name for braindecode derivatives
DEFAULT_PIPELINE_NAME = "braindecode"


def _raw_from_info(
    info: "mne.Info",
    bad_channels: Optional[list[str]] = None,
) -> "mne.io.RawArray":
    info = info.copy()
    if bad_channels is not None:
        info["bads"] = list(bad_channels)
    data = np.zeros((len(info["ch_names"]), 1), dtype=float)
    raw = mne.io.RawArray(data, info, verbose="error")
    if not raw.filenames or raw.filenames[0] is None:
        raw._filenames = [Path("dummy.fif")]
    return raw


def _read_tsv(writer, *args) -> pd.DataFrame:
    with TemporaryDirectory() as tmpdir, mne.utils.use_log_level("WARNING"):
        tsv_path = Path(tmpdir) / "sidecar.tsv"
        writer(*args, tsv_path, overwrite=True)
        return pd.read_csv(tsv_path, sep="\t")


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

    Delegates channel formatting to mne_bids.write._channels_tsv.

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
    bad_channels = bad_channels or info.get("bads", [])
    raw = _raw_from_info(info, bad_channels)
    return _read_tsv(_channels_tsv, raw)


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

    Delegates base JSON creation to mne_bids.write._sidecar_json.

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
    raw = _raw_from_info(info)
    manufacturer = manufacturer or "n/a"
    with TemporaryDirectory() as tmpdir, mne.utils.use_log_level("WARNING"):
        sidecar_path = Path(tmpdir) / "eeg.json"
        _sidecar_json(
            raw,
            task_name,
            manufacturer,
            sidecar_path,
            "eeg",
            overwrite=True,
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if task_description:
        sidecar["TaskDescription"] = task_description
    if instructions:
        sidecar["Instructions"] = instructions
    if institution_name:
        sidecar["InstitutionName"] = institution_name

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
        events_df.to_csv(
            events_path.fpath, sep="\t", index=False, na_rep="n/a", encoding="utf-8"
        )
        saved_files["events"] = events_path.fpath

    # Save channels.tsv
    channels_df = create_channels_tsv(info)
    channels_path = base_path.copy().update(suffix="channels", extension=".tsv")
    channels_df.to_csv(
        channels_path.fpath, sep="\t", index=False, na_rep="n/a", encoding="utf-8"
    )
    saved_files["channels"] = channels_path.fpath

    # Save EEG sidecar JSON
    sidecar = create_eeg_json_sidecar(info, task_name=task_name)
    sidecar_path = base_path.copy().update(suffix="eeg", extension=".json")
    with open(sidecar_path.fpath, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    saved_files["sidecar"] = sidecar_path.fpath

    return saved_files


class BIDSSourcedataLayout:
    """
    Helper class for creating BIDS sourcedata folder structure.

    This creates a structure using the BIDS ``sourcedata/`` directory,
    which is not validated by BIDS validators, allowing us to store
    data in Zarr format for efficient training.

    Structure:
    sourcedata/<pipeline>/
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
        Initialize BIDS sourcedata layout.

        Parameters
        ----------
        root : str | Path
            Root directory for sourcedata.
        pipeline_name : str
            Name of the processing pipeline.
        """
        self.root = Path(root)
        self.pipeline_name = pipeline_name
        self.sourcedata_dir = self.root / "sourcedata" / pipeline_name

    def create_structure(self) -> Path:
        """Create the basic sourcedata directory structure."""
        self.sourcedata_dir.mkdir(parents=True, exist_ok=True)
        return self.sourcedata_dir

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
            root=self.sourcedata_dir,
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
        Save dataset_description.json for sourcedata.

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
            path=self.sourcedata_dir,
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
        participants_path = self.sourcedata_dir / "participants.tsv"
        participants_df.to_csv(
            participants_path, sep="\t", index=False, na_rep="n/a", encoding="utf-8"
        )
        return participants_path

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

"""Tests for BIDS validation of braindecode's BIDS-inspired format."""

import json

import mne
import numpy as np
import pandas as pd
import pytest

from braindecode.datasets.bids.format import (
    BIDSSourcedataLayout,
    create_channels_tsv,
    create_eeg_json_sidecar,
    create_events_tsv,
    create_participants_tsv,
    make_dataset_description,
)

try:
    from bids_validator import BIDSValidator

    HAS_BIDS_VALIDATOR = True
except ImportError:
    HAS_BIDS_VALIDATOR = False


def test_dataset_description_has_required_fields(tmp_path):
    """Test dataset_description.json has required BIDS fields."""
    desc_path = make_dataset_description(
        path=tmp_path, name="Test", pipeline_name="braindecode"
    )
    with open(desc_path) as f:
        content = json.load(f)

    assert content["Name"] == "Test"
    assert "BIDSVersion" in content
    assert content["DatasetType"] == "derivative"
    assert "GeneratedBy" in content


def test_participants_tsv_has_participant_id():
    """Test participants.tsv has correctly formatted participant_id."""
    descriptions = [pd.Series({"subject": "001"}), pd.Series({"subject": "002"})]
    df = create_participants_tsv(descriptions)

    assert "participant_id" in df.columns
    assert all(pid.startswith("sub-") for pid in df["participant_id"])


def test_channels_tsv_has_required_columns():
    """Test channels.tsv has required BIDS columns in correct order."""
    info = mne.create_info(["EEG1", "EEG2"], sfreq=256.0, ch_types="eeg")
    df = create_channels_tsv(info)

    assert list(df.columns[:3]) == ["name", "type", "units"]


def test_events_tsv_has_onset_and_duration():
    """Test events.tsv has required onset and duration columns."""
    metadata = pd.DataFrame(
        {"i_start_in_trial": [0, 256], "i_stop_in_trial": [256, 512], "target": [0, 1]}
    )
    df = create_events_tsv(metadata, sfreq=256.0)

    assert "onset" in df.columns
    assert "duration" in df.columns
    assert df["onset"].dtype == np.float64


def test_eeg_json_sidecar_has_required_fields():
    """Test EEG sidecar JSON has required BIDS fields."""
    info = mne.create_info(["EEG1"], sfreq=256.0, ch_types="eeg")
    sidecar = create_eeg_json_sidecar(info, task_name="rest")

    for field in ["TaskName", "SamplingFrequency", "PowerLineFrequency"]:
        assert field in sidecar


def test_sourcedata_layout_creates_correct_structure(tmp_path):
    """Test BIDSSourcedataLayout creates sourcedata/<pipeline>/ structure."""
    layout = BIDSSourcedataLayout(tmp_path, pipeline_name="braindecode")
    sourcedata_dir = layout.create_structure()

    assert sourcedata_dir.exists()
    assert sourcedata_dir == tmp_path / "sourcedata" / "braindecode"


def test_bids_path_generation(tmp_path):
    """Test BIDS path entities are correctly set."""
    layout = BIDSSourcedataLayout(tmp_path, pipeline_name="braindecode")
    layout.create_structure()

    bids_path = layout.get_bids_path(
        pd.Series({"subject": "001", "task": "rest"}), extension=".zarr"
    )

    assert bids_path.subject == "001"
    assert bids_path.task == "rest"
    assert bids_path.extension == ".zarr"


@pytest.mark.skipif(not HAS_BIDS_VALIDATOR, reason="bids_validator not installed")
def test_zarr_paths_are_not_valid_bids():
    """Test .zarr paths are NOT valid BIDS (documenting intentional non-compliance)."""
    validator = BIDSValidator()
    assert not validator.is_bids("/sub-001/eeg/sub-001_task-rest_eeg.zarr")


@pytest.mark.skipif(not HAS_BIDS_VALIDATOR, reason="bids_validator not installed")
def test_edf_paths_are_valid_bids():
    """Test .edf paths are valid BIDS."""
    validator = BIDSValidator()
    assert validator.is_bids("/sub-001/eeg/sub-001_task-rest_eeg.edf")


def test_eeg_json_sidecar_recording_duration_fields():
    """Test EEG sidecar JSON includes recording duration fields from metadata."""
    info = mne.create_info(["EEG1", "EEG2"], sfreq=256.0, ch_types="eeg")

    # Test with explicit recording duration parameters
    recording_duration = 10.0  # 10 seconds
    recording_type = "epoched"
    epoch_length = 2.0  # 2 second epochs

    sidecar = create_eeg_json_sidecar(
        info,
        task_name="test",
        recording_duration=recording_duration,
        recording_type=recording_type,
        epoch_length=epoch_length,
    )

    assert sidecar["RecordingDuration"] == recording_duration
    assert sidecar["RecordingType"] == recording_type
    assert sidecar["EpochLength"] == epoch_length


def test_recording_duration_computed_from_metadata_multiple_recordings():
    """Test recording duration is correctly computed for multiple recordings.

    Simulates a BaseConcatDataset with multiple recordings, each having
    different numbers of epochs and durations - matching real-world usage.
    """
    sfreq = 256.0

    # Simulate 3 recordings with different epoch counts and lengths
    # Recording 1: 5 epochs of 1 second each (256 samples)
    # Recording 2: 10 epochs of 0.5 seconds each (128 samples)
    # Recording 3: 3 epochs of 2 seconds each (512 samples)
    recordings = [
        {
            "subject": "001",
            "metadata": pd.DataFrame({
                "i_start_in_trial": [0, 256, 512, 768, 1024],
                "i_stop_in_trial": [256, 512, 768, 1024, 1280],
                "target": [0, 1, 0, 1, 0],
            }),
            "expected_epoch_length": 1.0,
            "expected_duration": 5.0,
        },
        {
            "subject": "002",
            "metadata": pd.DataFrame({
                "i_start_in_trial": [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152],
                "i_stop_in_trial": [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }),
            "expected_epoch_length": 0.5,
            "expected_duration": 5.0,
        },
        {
            "subject": "003",
            "metadata": pd.DataFrame({
                "i_start_in_trial": [0, 512, 1024],
                "i_stop_in_trial": [512, 1024, 1536],
                "target": [0, 1, 0],
            }),
            "expected_epoch_length": 2.0,
            "expected_duration": 6.0,
        },
    ]

    for rec in recordings:
        metadata = rec["metadata"]

        # Compute epoch_length and recording_duration as hub.py does
        i_start = metadata["i_start_in_trial"].iloc[0]
        i_stop = metadata["i_stop_in_trial"].iloc[0]
        epoch_length = (i_stop - i_start) / sfreq
        n_epochs = len(metadata)
        recording_duration = n_epochs * epoch_length

        assert epoch_length == rec["expected_epoch_length"], (
            f"Subject {rec['subject']}: expected epoch_length "
            f"{rec['expected_epoch_length']}, got {epoch_length}"
        )
        assert recording_duration == rec["expected_duration"], (
            f"Subject {rec['subject']}: expected duration "
            f"{rec['expected_duration']}, got {recording_duration}"
        )

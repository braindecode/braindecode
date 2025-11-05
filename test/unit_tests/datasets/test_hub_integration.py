# Authors: Kuntal Kokate
#
# License: BSD-3

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.preprocessing import create_windows_from_events

# MOABB fake dataset configuration
bnci_kwargs = {
    "n_sessions": 1,
    "n_runs": 1,
    "n_subjects": 1,
    "paradigm": "imagery",
    "duration": 50.0,  # Short for fast tests
    "sfreq": 250,
    "event_list": ("feet", "left_hand"),
    "channels": ("C3", "Cz"),
}


@pytest.fixture()
def setup_concat_windows_dataset():
    """Create a small windowed dataset for testing."""
    dataset = MOABBDataset(
        dataset_name="FakeDataset", subject_ids=[1], dataset_kwargs=bnci_kwargs
    )
    return create_windows_from_events(
        concat_ds=dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,  # Required for .windows attribute
    )


# =============================================================================
# Hub Mixin Integration Tests
# =============================================================================


def test_hub_mixin_available(setup_concat_windows_dataset):
    """Test that Hub mixin methods are available on BaseConcatDataset."""
    dataset = setup_concat_windows_dataset

    # Check that methods exist
    assert hasattr(dataset, "push_to_hub")
    assert hasattr(dataset, "from_pretrained")
    assert callable(dataset.push_to_hub)
    assert callable(BaseConcatDataset.from_pretrained)


def test_push_to_hub_requires_huggingface_hub(
    setup_concat_windows_dataset, monkeypatch
):
    """Test that push_to_hub raises error when huggingface-hub not installed."""
    # Mock HF_HUB_AVAILABLE as False
    monkeypatch.setattr(
        "braindecode.datasets.hub_mixin.HF_HUB_AVAILABLE", False
    )

    dataset = setup_concat_windows_dataset

    with pytest.raises(ImportError, match="huggingface-hub is not installed"):
        dataset.push_to_hub("test-repo")


def test_from_pretrained_requires_huggingface_hub(monkeypatch):
    """Test that from_pretrained raises error when huggingface-hub not installed."""
    # Mock HF_HUB_AVAILABLE as False
    monkeypatch.setattr(
        "braindecode.datasets.hub_mixin.HF_HUB_AVAILABLE", False
    )

    with pytest.raises(ImportError, match="huggingface-hub is not installed"):
        BaseConcatDataset.from_pretrained("test-repo")


@patch("braindecode.datasets.hub_mixin.create_repo")
@patch("braindecode.datasets.hub_mixin.upload_folder")
def test_push_to_hub_zarr_format(
    mock_upload, mock_create_repo, setup_concat_windows_dataset
):
    """Test push_to_hub with Zarr format (mocked)."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset

    mock_upload.return_value = "https://huggingface.co/datasets/test-repo/blob/main"

    try:
        dataset.push_to_hub(
            repo_id="test-user/test-dataset",
            commit_message="Test upload",
        )
    except Exception:
        # Expected to fail due to missing actual Hub connection
        pass

    # Verify create_repo was called
    mock_create_repo.assert_called_once()


@patch("braindecode.datasets.hub_mixin.create_repo")
@patch("braindecode.datasets.hub_mixin.upload_folder")
def test_push_to_hub_custom_compression(
    mock_upload, mock_create_repo, setup_concat_windows_dataset
):
    """Test push_to_hub with custom compression settings."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset

    mock_upload.return_value = "https://huggingface.co/datasets/test-repo/blob/main"

    try:
        dataset.push_to_hub(
            repo_id="test-user/test-dataset",
            compression="blosc",
            compression_level=9,
            commit_message="Test upload with custom compression",
        )
    except Exception:
        pass

    mock_create_repo.assert_called_once()


def test_dataset_card_generation(setup_concat_windows_dataset):
    """Test that dataset card (README) is generated correctly."""
    pytest.importorskip("huggingface_hub")

    dataset = setup_concat_windows_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Call the internal method to generate dataset card
        dataset._save_dataset_card(tmp_path)

        readme_path = tmp_path / "README.md"
        assert readme_path.exists()

        # Check content
        content = readme_path.read_text()
        assert "braindecode" in content
        assert "Number of recordings" in content
        assert "Sampling frequency" in content
        assert "Usage" in content
        assert "zarr" in content.lower()  # Should mention Zarr format


# =============================================================================
# Local Round-Trip Tests (without actual Hub interaction)
# =============================================================================


def test_local_save_and_load_zarr(setup_concat_windows_dataset):
    """Test local save in Zarr format and load using file path."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zarr")

    from braindecode.datautil.hub_formats import convert_to_zarr, load_from_zarr

    dataset = setup_concat_windows_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as Zarr
        zarr_path = Path(tmpdir) / "dataset.zarr"
        convert_to_zarr(dataset, zarr_path)

        # Load back
        loaded = load_from_zarr(zarr_path, preload=True)

        # Verify
        assert len(loaded.datasets) == len(dataset.datasets)
        assert len(loaded) == len(dataset)


# =============================================================================
# Mock Hub Download Tests
# =============================================================================


@patch("huggingface_hub.snapshot_download")
def test_from_pretrained_zarr_mock(mock_snapshot):
    """Test from_pretrained with Zarr format (mocked Hub download)."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zarr")

    # Create a temporary Zarr directory to return from mock
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset
        from braindecode.datautil.hub_formats import convert_to_zarr

        mock_dataset = MOABBDataset(
            dataset_name="FakeDataset", subject_ids=[1], dataset_kwargs=bnci_kwargs
        )
        mock_windows = create_windows_from_events(
            concat_ds=mock_dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            use_mne_epochs=True,
        )

        zarr_path = Path(tmpdir) / "dataset.zarr"
        convert_to_zarr(mock_windows, zarr_path)

        format_info_path = Path(tmpdir) / "format_info.json"
        with open(format_info_path, "w") as f:
            json.dump({"format": "zarr"}, f)

        # Mock snapshot_download to return our temp directory
        mock_snapshot.return_value = tmpdir

        # Test from_pretrained
        loaded = BaseConcatDataset.from_pretrained("mock-user/mock-dataset")

        assert isinstance(loaded, BaseConcatDataset)
        assert len(loaded.datasets) > 0


@patch("huggingface_hub.snapshot_download")
def test_from_pretrained_wrong_format(mock_snapshot):
    """Test from_pretrained with non-zarr format raises error."""
    pytest.importorskip("huggingface_hub")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create format_info with wrong format
        format_info_path = Path(tmpdir) / "format_info.json"
        with open(format_info_path, "w") as f:
            json.dump({"format": "hdf5"}, f)

        mock_snapshot.return_value = tmpdir

        with pytest.raises(RuntimeError, match="only 'zarr' format is supported"):
            BaseConcatDataset.from_pretrained("mock-user/mock-dataset")


@patch("huggingface_hub.snapshot_download")
def test_from_pretrained_not_found(mock_snapshot):
    """Test from_pretrained with non-existent repository."""
    pytest.importorskip("huggingface_hub")

    from huggingface_hub.utils import HfHubHTTPError

    # Mock 404 error
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_snapshot.side_effect = HfHubHTTPError("Not found", response=mock_response)

    with pytest.raises(FileNotFoundError, match="not found on Hugging Face Hub"):
        BaseConcatDataset.from_pretrained("nonexistent/repo")


# =============================================================================
# Integration with Preprocessing Tests
# =============================================================================


def test_hub_integration_preserves_preprocessing_kwargs(setup_concat_windows_dataset):
    """Test that preprocessing kwargs are preserved in Hub format."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zarr")

    from braindecode.datautil.hub_formats import convert_to_zarr, load_from_zarr

    dataset = setup_concat_windows_dataset

    # Add some preprocessing kwargs
    dataset.window_kwargs = {"test_param": "test_value"}

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "dataset.zarr"
        convert_to_zarr(dataset, zarr_path)

        loaded = load_from_zarr(zarr_path)

        # Check that kwargs are preserved
        assert hasattr(loaded, "window_kwargs")
        assert loaded.window_kwargs == {"test_param": "test_value"}


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_push_to_hub_create_repo_failure(setup_concat_windows_dataset):
    """Test handling of repository creation failure."""
    pytest.importorskip("huggingface_hub")

    dataset = setup_concat_windows_dataset

    with patch("braindecode.datasets.hub_mixin.create_repo") as mock_create:
        mock_create.side_effect = Exception("Failed to create repo")

        with pytest.raises(RuntimeError, match="Failed to create repository"):
            dataset.push_to_hub("test-repo")


def test_push_to_hub_upload_failure(setup_concat_windows_dataset):
    """Test handling of upload failure."""
    pytest.importorskip("huggingface_hub")

    dataset = setup_concat_windows_dataset

    with patch("braindecode.datasets.hub_mixin.create_repo"):
        with patch("braindecode.datasets.hub_mixin.upload_folder") as mock_upload:
            mock_upload.side_effect = Exception("Upload failed")

            with pytest.raises(RuntimeError, match="Failed to upload dataset"):
                dataset.push_to_hub("test-repo")


def test_from_pretrained_missing_zarr_directory(tmpdir):
    """Test from_pretrained when zarr directory is missing."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zarr")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create format_info but no zarr directory
        format_info_path = Path(tmpdir) / "format_info.json"
        with open(format_info_path, "w") as f:
            json.dump({"format": "zarr"}, f)

        with patch("huggingface_hub.snapshot_download") as mock:
            mock.return_value = tmpdir

            with pytest.raises(RuntimeError, match="Zarr dataset not found"):
                BaseConcatDataset.from_pretrained("test-user/test-dataset")

# Authors: Kuntal Kokate
#
# License: BSD-3

"""Simple tests for Zarr format converters."""

import pytest
from pathlib import Path

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.datautil.hub_formats import convert_to_zarr, load_from_zarr, get_format_info


# MOABB fake dataset configuration
bnci_kwargs = {
    "n_sessions": 1,
    "n_runs": 1,
    "n_subjects": 1,
    "paradigm": "imagery",
    "duration": 50.0,
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
        use_mne_epochs=True,
    )


def test_zarr_round_trip(setup_concat_windows_dataset, tmp_path):
    """Test saving and loading dataset in Zarr format."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    zarr_path = tmp_path / "dataset.zarr"

    # Save to Zarr
    convert_to_zarr(dataset, zarr_path, compression="blosc", compression_level=5)

    assert zarr_path.exists()
    assert (zarr_path / ".zgroup").exists()

    # Load from Zarr
    loaded = load_from_zarr(zarr_path, preload=True)

    assert len(loaded.datasets) == len(dataset.datasets)
    assert loaded.datasets[0].windows.info["sfreq"] == dataset.datasets[0].windows.info["sfreq"]


def test_get_format_info(setup_concat_windows_dataset):
    """Test getting format information from dataset."""
    dataset = setup_concat_windows_dataset

    info = get_format_info(dataset)

    assert "n_recordings" in info
    assert "total_samples" in info
    assert "total_size_mb" in info
    assert info["n_recordings"] == len(dataset.datasets)
    assert info["total_samples"] > 0
    assert info["total_size_mb"] > 0


def test_no_lazy_imports_in_hub_formats():
    """Verify that hub_formats module has global imports only."""
    from braindecode.datautil import hub_formats
    import inspect

    # Get all functions in the hub_formats module
    functions = [
        obj for name, obj in inspect.getmembers(hub_formats)
        if inspect.isfunction(obj) and obj.__module__ == hub_formats.__name__
    ]

    # Check that no functions have lazy imports
    for func in functions:
        source = inspect.getsource(func)
        # Functions should not have 'from ..datasets' imports inside them
        lines = source.split('\n')
        for line in lines:
            # Skip the function definition line
            if line.strip().startswith('def '):
                continue
            # Check for import statements inside function body
            if 'from ..datasets' in line and 'import' in line:
                # This should only be at module level, not in functions
                assert False, f"{func.__name__} has lazy import: {line.strip()}"

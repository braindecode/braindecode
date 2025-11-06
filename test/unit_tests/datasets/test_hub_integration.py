# Authors: Kuntal Kokate
#
# License: BSD-3

"""Simple tests for Hugging Face Hub integration."""

import pytest
from pathlib import Path

from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.preprocessing import create_windows_from_events


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


def test_hub_mixin_methods_exist(setup_concat_windows_dataset):
    """Test that Hub mixin methods are available on BaseConcatDataset."""
    dataset = setup_concat_windows_dataset

    # Check that methods exist
    assert hasattr(dataset, "push_to_hub")
    assert hasattr(dataset, "from_pretrained")
    assert callable(dataset.push_to_hub)
    assert callable(dataset.from_pretrained)


def test_dataset_card_generation(setup_concat_windows_dataset, tmp_path):
    """Test that dataset card (README) is generated correctly."""
    pytest.importorskip("huggingface_hub")

    dataset = setup_concat_windows_dataset

    # Call the internal method to generate dataset card
    dataset._save_dataset_card(tmp_path)

    # Check that README.md was created
    readme_path = tmp_path / "README.md"
    assert readme_path.exists()

    # Check content
    content = readme_path.read_text()
    assert "braindecode" in content.lower()
    assert "zarr" in content.lower()
    assert str(len(dataset.datasets)) in content


def test_zarr_save_and_load(setup_concat_windows_dataset, tmp_path):
    """Test basic Zarr save and load functionality."""
    pytest.importorskip("zarr")

    dataset = setup_concat_windows_dataset
    zarr_path = tmp_path / "dataset.zarr"

    # Test via internal methods (avoiding HuggingFace Hub)
    dataset._convert_to_zarr_inline(
        zarr_path,
        compression="blosc",
        compression_level=5,
    )

    assert zarr_path.exists()

    # Load it back
    loaded = dataset._load_from_zarr_inline(zarr_path, preload=True)

    assert isinstance(loaded, BaseConcatDataset)
    assert len(loaded.datasets) == len(dataset.datasets)


def test_no_lazy_imports_in_hub_module():
    """Verify that hub module uses registry pattern instead of lazy imports."""
    from braindecode.datasets import hub
    import inspect

    # Get all functions in the hub module
    functions = [
        obj for name, obj in inspect.getmembers(hub)
        if inspect.isfunction(obj) and obj.__module__ == hub.__name__
    ]

    # Check that no functions have 'from .base import' (which would be circular)
    for func in functions:
        source = inspect.getsource(func)
        # No function should have 'from .base import' or 'from ..datasets.base import'
        assert "from .base import" not in source, f"{func.__name__} has circular import"
        assert "from ..datasets.base import" not in source, f"{func.__name__} has circular import"


def test_registry_pattern_works():
    """Test that registry pattern allows access to dataset classes without circular imports."""
    from braindecode.datasets.registry import get_dataset_class, get_dataset_type
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import create_windows_from_events

    # Get classes from registry
    WindowsDataset = get_dataset_class("WindowsDataset")
    EEGWindowsDataset = get_dataset_class("EEGWindowsDataset")
    BaseConcatDataset = get_dataset_class("BaseConcatDataset")

    # Verify classes are actual types
    assert isinstance(WindowsDataset, type)
    assert isinstance(EEGWindowsDataset, type)
    assert isinstance(BaseConcatDataset, type)

    # Test get_dataset_type with actual dataset
    dataset = MOABBDataset(
        dataset_name="FakeDataset", subject_ids=[1], dataset_kwargs=bnci_kwargs
    )
    windowed = create_windows_from_events(
        concat_ds=dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        use_mne_epochs=True,
    )

    # Check type detection works
    assert get_dataset_type(windowed.datasets[0]) == "WindowsDataset"
    assert get_dataset_type(windowed) == "BaseConcatDataset"

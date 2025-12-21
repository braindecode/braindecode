# Authors: Mohammad Bayazi <mj.darvishi92@gmail.com>
#
# License: BSD-3
import os
import platform
from unittest import mock

import pandas as pd
import pytest

from braindecode.datasets.nmt import NMT_archive_name, _NMTMock
from braindecode.datasets.utils import _correct_dataset_path


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_nmt():
    nmt = _NMTMock(
        path="",
        n_jobs=1,  # required for test to work. mocking seems to fail otherwise
    )
    assert len(nmt.datasets) == 7
    assert nmt.description.shape == (7, 7)
    # assert nmt.description.version.to_list() == [
    #     'v2.0.0', 'v2.0.0', 'v2.0.0', 'v2.0.0', 'v2.0.0']
    assert nmt.description.pathological.to_list() == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert nmt.description.train.to_list() == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    x, y = nmt[0]
    assert x.shape == (21, 1)
    assert y is False
    x, y = nmt[-1]
    assert y is False

    nmt = _NMTMock(
        path="",
        target_name="age",
        n_jobs=1,
    )
    x, y = nmt[-1]
    assert y == 71
    for ds in nmt.datasets:
        ds.target_name = "gender"
    x, y = nmt[0]
    assert y == "M"


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)
def test_nmt_recording_ids_metadata_alignment():
    """Test that metadata is correctly aligned when using recording_ids.
    
    This test ensures that when recording_ids is provided, metadata is matched
    by record name rather than positional index, preventing misalignment when
    the CSV order differs from the sorted file order.
    """
    # Test with recording_ids to ensure correct metadata alignment
    # Using recording_ids [0, 2, 4] should select files 0000036, 0000038, 0000040
    # and their corresponding metadata by name match, not by position
    nmt = _NMTMock(
        path="",
        recording_ids=[0, 2, 4],
        n_jobs=1,
    )
    
    # Should have exactly 3 datasets corresponding to recording_ids [0, 2, 4]
    assert len(nmt.datasets) == 3
    
    # Verify the correct files are selected
    # Files should be 0000036.edf, 0000038.edf, 0000040.edf after sorting
    paths = nmt.description["path"].tolist()
    assert any("0000036.edf" in p for p in paths), f"Expected 0000036.edf in paths: {paths}"
    assert any("0000038.edf" in p for p in paths), f"Expected 0000038.edf in paths: {paths}"
    assert any("0000040.edf" in p for p in paths), f"Expected 0000040.edf in paths: {paths}"
    
    # Verify metadata alignment by checking the age values
    # From _fake_pd_read_csv (now updated to match file IDs):
    # 0000036.edf has age 35
    # 0000038.edf has age 62  
    # 0000040.edf has age 19
    ages = nmt.description["age"].tolist()
    assert ages == [35, 62, 19], f"Expected [35, 62, 19] but got {ages}"
    
    # Verify pathological labels
    # 0000036.edf is "normal" -> False
    # 0000038.edf is "normal" -> False
    # 0000040.edf is "normal" -> False
    pathological = nmt.description["pathological"].tolist()
    assert pathological == [False, False, False]
    
    # Verify gender
    # All three in the mock data are "male" -> "M"
    genders = nmt.description["gender"].tolist()
    assert genders == ["M", "M", "M"]


# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)
def test_nmt_recording_ids_with_unordered_csv():
    """Test metadata alignment when CSV rows are in different order than sorted files.
    
    This test creates a scenario where the CSV is deliberately in a different order
    than the sorted file paths to verify that metadata matching works by record name
    rather than positional index.
    """
    # Create a custom mock CSV with rows in reverse order
    def _fake_pd_read_csv_reverse(*args, **kwargs):
        # CSV rows in REVERSE order compared to sorted files
        data = [
            ["0000042.edf", "normal", 71, "male", "train"],
            ["0000041.edf", "abnormal", 55, "female", "test"],
            ["0000040.edf", "normal", 19, "male", "train"],
            ["0000039.edf", "abnormal", 41, "female", "test"],
            ["0000038.edf", "normal", 62, "male", "train"],
            ["0000037.edf", "abnormal", 28, "female", "test"],
            ["0000036.edf", "normal", 35, "male", "train"],
        ]
        df = pd.DataFrame(data, columns=["recordname", "label", "age", "gender", "loc"])
        df.set_index("recordname", inplace=True)
        return df
    
    # Patch the CSV mock with reversed order
    with mock.patch("braindecode.datasets.nmt._fake_pd_read_csv", new=_fake_pd_read_csv_reverse):
        nmt = _NMTMock(
            path="",
            recording_ids=[0, 2, 4],  # Select files 0000036, 0000038, 0000040
            n_jobs=1,
        )
    
    # Even though CSV is in reverse order, metadata should still match correctly
    # File 0000036.edf should have age 35 (not 71 from positional index 0 in reversed CSV)
    # File 0000038.edf should have age 62 (not 19 from positional index 2 in reversed CSV)
    # File 0000040.edf should have age 19 (not 55 from positional index 4 in reversed CSV)
    ages = nmt.description["age"].tolist()
    assert ages == [35, 62, 19], f"Expected [35, 62, 19] but got {ages}"


@pytest.fixture
def setup_file_structure(tmp_path):
    """Sets up a temporary file structure for testing."""
    original_path = tmp_path / "original"
    original_path.mkdir()
    (original_path / f"{NMT_archive_name}.unzip").touch()
    return tmp_path


def test_path_exists(tmp_path):
    """Test if the function returns the same path when it already exists."""
    # Create a dummy file to simulate an existing path
    test_file = tmp_path / "test_file"
    test_file.mkdir()
    (test_file / "nmt_scalp_eeg_dataset").mkdir()

    # Assert that the path returned is the same since it exists
    assert (
        _correct_dataset_path(
            str(test_file), NMT_archive_name, "nmt_scalp_eeg_dataset"
        )
        == str(test_file / "nmt_scalp_eeg_dataset")
    )


def test_path_does_not_exist_but_unzip_file_does(setup_file_structure):
    """
    Renames the unzip file correctly when the path does not exist
    and appends the dataset directory name to the path.

    """
    # Path before renaming
    original_path = setup_file_structure / "original"
    unzip_file_path = original_path / f"{NMT_archive_name}.unzip"
    test_path = original_path / "original"

    # Call the function
    corrected_path = _correct_dataset_path(
        str(test_path),
        NMT_archive_name,
        "nmt_scalp_eeg_dataset",
    )
    # Path that does not exist to trigger renaming

    # Assert the unzip file has been renamed to the original path
    # The function renames the .unzip file to the expected path, then
    # checks if subfolder exists. Since we don't create the subfolder,
    # it should return the renamed path
    assert corrected_path == str(test_path)
    assert not unzip_file_path.exists()
    # Ensuring the unzip file was renamed (implied by the disappearance)


def test_permission_error_when_renaming_unzip_file(
    monkeypatch, setup_file_structure
):
    """Test if the function raises a PermissionError
    when renaming is not possible.
    """

    def mock_rename(src, dst):
        raise PermissionError("Mock permission error.")

    monkeypatch.setattr(os, "rename", mock_rename)

    # Path setup
    original_path = setup_file_structure / "original"
    expected_new_path = original_path / "original"

    # Assert that PermissionError is raised
    with pytest.raises(PermissionError):
        _correct_dataset_path(
            str(expected_new_path),
            NMT_archive_name,
            "nmt_scalp_eeg_dataset",
        )

# Authors: Mohammad Bayazi <mj.darvishi92@gmail.com>
#
# License: BSD-3
import os
import platform

import pytest

from braindecode.datasets.nmt import NMT_archive_name, _correct_path, _NMTMock


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
    assert _correct_path(str(test_file)) == str(test_file)


def test_path_does_not_exist_but_unzip_file_does(setup_file_structure):
    """
    Renames the unzip file correctly when the path does not exist
    and appends the dataset directory name to the path.

    """
    # Path before renaming
    original_path = setup_file_structure / "original"
    unzip_file_path = original_path / f"{NMT_archive_name}.unzip"
    expected_new_path = original_path / "original" / "nmt_scalp_eeg_dataset"
    # Updated expectation based on function's behavior

    # Call the function
    corrected_path = _correct_path(str(original_path / "original"))
    # Path that does not exist to trigger renaming

    # Assert the unzip file has been renamed to the original path
    # (not directly checking here because rename is not creating a folder,
    # just renaming the path)
    assert corrected_path == str(expected_new_path)
    assert not unzip_file_path.exists()
    # Ensuring the unzip file was renamed (implied by the disappearance)


def test_permission_error_when_renaming_unzip_file(monkeypatch,
                                                   setup_file_structure):
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
        _correct_path(str(expected_new_path))

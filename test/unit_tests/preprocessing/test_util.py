# Authors: Christian Kothe <christian.kothe@intheon.io>
# License: BSD-3

import copy

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from braindecode.preprocessing.util import mne_load_metadata, mne_store_metadata

expected_tag = "braindecode-meta:"


@pytest.fixture
def dummy_raw():
    """Create a simple dummy MNE Raw object for testing."""
    n_channels = 2
    n_times = 1000
    sfreq = 100
    data = np.random.randn(n_channels, n_times)
    info = mne.create_info(ch_names=[f"ch{i}" for i in range(n_channels)], sfreq=sfreq)
    raw = mne.io.RawArray(data, info)
    return raw


# Basic Functionality Tests


def test_mne_store_metadata_basic(dummy_raw):
    # Store a simple payload
    raw = copy.deepcopy(dummy_raw)
    payload = {"value": 42, "name": "test"}
    mne_store_metadata(raw, payload, key="test_key")

    # Verify it's in the description field
    assert raw.info["description"] is not None
    assert expected_tag in raw.info["description"]


def test_mne_store_and_load_roundtrip(dummy_raw):
    # Store complex payload with nested structures
    raw = copy.deepcopy(dummy_raw)
    payload = {
        "nested": {"dict": {"with": "values"}},
        "list": [1, 2, 3, {"nested": "item"}],
        "number": 3.14159,
        "bool": True,
        "null": None,
    }
    mne_store_metadata(raw, payload, key="complex")

    # Load it back and verify integrity
    loaded = mne_load_metadata(raw, key="complex")
    assert loaded == payload


def test_mne_load_metadata_nonexistent_key(dummy_raw):
    # Try to load a key that doesn't exist
    raw = copy.deepcopy(dummy_raw)
    loaded = mne_load_metadata(raw, key="nonexistent")
    assert loaded is None


# Multiple Keys Tests


def test_mne_store_multiple_keys(dummy_raw):
    # Store multiple payloads under different keys
    raw = copy.deepcopy(dummy_raw)
    payload1 = {"data": "first"}
    payload2 = {"data": "second"}
    payload3 = {"data": "third"}

    mne_store_metadata(raw, payload1, key="key1")
    mne_store_metadata(raw, payload2, key="key2")
    mne_store_metadata(raw, payload3, key="key3")

    # Load each independently
    assert mne_load_metadata(raw, key="key1") == payload1
    assert mne_load_metadata(raw, key="key2") == payload2
    assert mne_load_metadata(raw, key="key3") == payload3


def test_mne_store_update_existing_key(dummy_raw):
    # Store a payload, then update it
    raw = copy.deepcopy(dummy_raw)
    original_payload = {"version": 1}
    updated_payload = {"version": 2}

    mne_store_metadata(raw, original_payload, key="data")
    mne_store_metadata(raw, updated_payload, key="data", no_overwrite=False)

    # Verify it's updated
    loaded = mne_load_metadata(raw, key="data")
    assert loaded == updated_payload



# no_overwrite Parameter Tests


def test_mne_store_no_overwrite_true(dummy_raw):
    # Store a payload, then try to overwrite with no_overwrite=True
    raw = copy.deepcopy(dummy_raw)
    original_payload = {"value": "original"}
    new_payload = {"value": "new"}

    mne_store_metadata(raw, original_payload, key="data")
    mne_store_metadata(raw, new_payload, key="data", no_overwrite=True)

    # Verify original data preserved
    loaded = mne_load_metadata(raw, key="data")
    assert loaded == original_payload


# delete Parameter Tests


def test_mne_load_delete_true(dummy_raw):
    # Store a payload, load with delete=True
    raw = copy.deepcopy(dummy_raw)
    payload = {"value": "test"}

    mne_store_metadata(raw, payload, key="data")
    loaded = mne_load_metadata(raw, key="data", delete=True)

    # Verify data returned
    assert loaded == payload

    # Verify key removed
    loaded_again = mne_load_metadata(raw, key="data")
    assert loaded_again is None


def test_mne_load_delete_false(dummy_raw):
    # Store a payload, load with delete=False
    raw = copy.deepcopy(dummy_raw)
    payload = {"value": "test"}

    mne_store_metadata(raw, payload, key="data")
    loaded = mne_load_metadata(raw, key="data", delete=False)

    # Verify data returned
    assert loaded == payload

    # Verify data persists
    loaded_again = mne_load_metadata(raw, key="data")
    assert loaded_again == payload


def test_mne_load_delete_last_key(dummy_raw):
    # Store one key, load with delete=True
    raw = copy.deepcopy(dummy_raw)
    payload = {"value": "test"}

    mne_store_metadata(raw, payload, key="data")
    loaded = mne_load_metadata(raw, key="data", delete=True)

    # Verify data returned
    assert loaded == payload

    # Verify entire marker removed from description
    description = raw.info.get("description") or ""
    assert expected_tag not in description


# Edge Cases Tests


def test_mne_store_empty_numpy_array(dummy_raw):
    # Store an empty numpy array (sentinel object)
    raw = copy.deepcopy(dummy_raw)
    payload = {"empty": np.array([])}

    mne_store_metadata(raw, payload, key="data")
    loaded = mne_load_metadata(raw, key="data")

    # Verify special encoding/decoding works
    assert "empty" in loaded
    assert isinstance(loaded["empty"], np.ndarray)
    assert loaded["empty"].size == 0


def test_mne_store_numpy_arrays_comprehensive(dummy_raw):
    # Store various numpy arrays with different dtypes, shapes, and structures
    raw = copy.deepcopy(dummy_raw)
    payload = {
        "int32_1d": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "float64_2d": np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
        "float32_3d": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        "bool_array": np.array([True, False, True, False], dtype=bool),
        "uint8_array": np.array([0, 127, 255], dtype=np.uint8),
        "int16_array": np.array([-32768, 0, 32767], dtype=np.int16),
        "single_element": np.array([42], dtype=np.int64),
        "empty_array": np.array([], dtype=np.float32),
        "zeros": np.zeros((3, 3), dtype=np.float64),
        "object_array": np.array([{"a": 1}, {"b": 2}], dtype=object),
    }

    mne_store_metadata(raw, payload, key="arrays")
    loaded = mne_load_metadata(raw, key="arrays")

    # Verify all arrays are correctly restored with proper dtype, shape, and content
    for key, original_array in payload.items():
        loaded_array = loaded[key]

        # Verify it's a numpy array
        assert isinstance(loaded_array, np.ndarray), f"{key} is not a numpy array"

        # Verify dtype matches
        assert loaded_array.dtype == original_array.dtype, \
            f"{key} dtype mismatch: {loaded_array.dtype} != {original_array.dtype}"

        # Verify shape matches
        assert loaded_array.shape == original_array.shape, \
            f"{key} shape mismatch: {loaded_array.shape} != {original_array.shape}"

        # Verify content matches (using appropriate comparison for object arrays)
        if original_array.dtype == object:
            # For object arrays, compare element by element
            assert len(loaded_array) == len(original_array)
            for i, (loaded_elem, orig_elem) in enumerate(zip(loaded_array, original_array)):
                assert loaded_elem == orig_elem, \
                    f"{key}[{i}] content mismatch: {loaded_elem} != {orig_elem}"
        else:
            # For numeric arrays, use numpy testing
            assert_array_equal(loaded_array, original_array,
                             err_msg=f"{key} content mismatch")


def test_mne_store_jagged_arrays(dummy_raw):
    # Store jagged arrays (numpy arrays containing numpy arrays of different lengths)
    raw = copy.deepcopy(dummy_raw)
    payload = {
        "jagged_1d": np.array([
            np.array([1, 2, 3]),
            np.array([4, 5]),
            np.array([6, 7, 8, 9]),
        ], dtype=object),
        "jagged_2d": np.array([
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6, 7]]),
            np.array([[8]]),
        ], dtype=object),
        "mixed_dtypes": np.array([
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4.5, 5.5], dtype=np.float64),
            np.array([True, False, True], dtype=bool),
        ], dtype=object),
    }

    mne_store_metadata(raw, payload, key="jagged")
    loaded = mne_load_metadata(raw, key="jagged")

    # Verify jagged arrays are correctly restored
    for key, original_jagged in payload.items():
        loaded_jagged = loaded[key]

        # Verify it's a numpy array with object dtype
        assert isinstance(loaded_jagged, np.ndarray), f"{key} is not a numpy array"
        assert loaded_jagged.dtype == object, f"{key} dtype should be object"
        assert loaded_jagged.shape == original_jagged.shape, \
            f"{key} shape mismatch: {loaded_jagged.shape} != {original_jagged.shape}"

        # Verify each sub-array
        assert len(loaded_jagged) == len(original_jagged)
        for i, (loaded_sub, orig_sub) in enumerate(zip(loaded_jagged, original_jagged)):
            assert isinstance(loaded_sub, np.ndarray), \
                f"{key}[{i}] should be a numpy array"
            assert loaded_sub.dtype == orig_sub.dtype, \
                f"{key}[{i}] dtype mismatch: {loaded_sub.dtype} != {orig_sub.dtype}"
            assert loaded_sub.shape == orig_sub.shape, \
                f"{key}[{i}] shape mismatch: {loaded_sub.shape} != {orig_sub.shape}"
            assert_array_equal(loaded_sub, orig_sub,
                             err_msg=f"{key}[{i}] content mismatch")


def test_mne_store_with_existing_description(dummy_raw):
    # Store metadata when description field has existing content
    # using some whitespace and newlines for good measure
    raw = copy.deepcopy(dummy_raw)
    existing_description = "Line 1\n  Line 2 with indent\n\nLine 4 after blank"
    raw.info["description"] = existing_description

    payload = {"value": "test"}
    mne_store_metadata(raw, payload, key="data")

    # Verify both preserved
    description = raw.info["description"]
    assert existing_description in description
    assert expected_tag in description

    # Verify data can be loaded
    loaded = mne_load_metadata(raw, key="data")
    assert loaded == payload


def test_mne_store_with_empty_description(dummy_raw):
    # Store metadata when description is empty or None
    raw = copy.deepcopy(dummy_raw)
    raw.info["description"] = None

    payload = {"value": "test"}
    mne_store_metadata(raw, payload, key="data")

    # Verify metadata stored
    loaded = mne_load_metadata(raw, key="data")
    assert loaded == payload

    # Test with empty string
    raw2 = copy.deepcopy(dummy_raw)
    raw2.info["description"] = ""

    mne_store_metadata(raw2, payload, key="data")
    loaded2 = mne_load_metadata(raw2, key="data")
    assert loaded2 == payload


def test_mne_load_corrupted_data(dummy_raw):
    # Manually corrupt the marker in description
    raw = copy.deepcopy(dummy_raw)
    payload = {"value": "test"}

    mne_store_metadata(raw, payload, key="data")

    # Corrupt the base64 data by truncating it significantly
    description = raw.info["description"]
    # Find the marker and corrupt the encoded data
    start_idx = description.find(expected_tag) + len(expected_tag)
    end_idx = description.find("-->", start_idx)
    # Replace the base64 data with invalid content
    corrupted = (
        description[:start_idx] + " !!!CORRUPTED!!! " + description[end_idx:]
    )
    raw.info["description"] = corrupted

    # Verify load returns None gracefully
    loaded = mne_load_metadata(raw, key="data")
    assert loaded is None


def test_mne_store_nasty_payload(dummy_raw):
    # Store payload with nested structures, special characters, unicode
    raw = copy.deepcopy(dummy_raw)
    payload = {
        "unicode": "Hello 世界 🌍",
        "special_chars": "!@#$%^&*()_+-=[]{}|;:',.<>?/`~",
        "nested": {
            "level1": {"level2": {"level3": "deep"}},
            "array": [1, 2, 3, [4, 5, 6]],
        },
        "numbers": [1, 2.5, -3, 1e10, float("inf"), float("-inf")],
        "bools": [True, False],
    }

    mne_store_metadata(raw, payload, key="complex")
    loaded = mne_load_metadata(raw, key="complex")

    # Verify roundtrip
    assert loaded["unicode"] == payload["unicode"]
    assert loaded["special_chars"] == payload["special_chars"]
    assert loaded["nested"] == payload["nested"]
    assert loaded["numbers"] == payload["numbers"]
    assert loaded["bools"] == payload["bools"]


def test_mne_store_nan_values(dummy_raw):
    # Store NaN values both as Python floats and in numpy arrays
    # NaN requires special treatment since nan != nan
    raw = copy.deepcopy(dummy_raw)
    payload = {
        "python_nan": float("nan"),
        "python_list_with_nan": [1.0, float("nan"), 3.0],
        "numpy_with_nan": np.array([1.0, np.nan, 3.0, np.inf, -np.inf]),
        "numpy_all_nan": np.array([np.nan, np.nan]),
        "nested_nan": {
            "inner": [float("nan"), 2.0],
            "array": np.array([np.nan, 1.0]),
        },
    }

    mne_store_metadata(raw, payload, key="nan_test")
    loaded = mne_load_metadata(raw, key="nan_test")

    # Verify Python NaN
    import math
    assert math.isnan(loaded["python_nan"])

    # Verify Python list with NaN
    assert loaded["python_list_with_nan"][0] == 1.0
    assert math.isnan(loaded["python_list_with_nan"][1])
    assert loaded["python_list_with_nan"][2] == 3.0

    # Verify numpy array with NaN and infinities
    numpy_with_nan = loaded["numpy_with_nan"]
    assert numpy_with_nan[0] == 1.0
    assert np.isnan(numpy_with_nan[1])
    assert numpy_with_nan[2] == 3.0
    assert np.isinf(numpy_with_nan[3]) and numpy_with_nan[3] > 0
    assert np.isinf(numpy_with_nan[4]) and numpy_with_nan[4] < 0

    # Verify numpy array with all NaN
    numpy_all_nan = loaded["numpy_all_nan"]
    assert np.all(np.isnan(numpy_all_nan))

    # Verify nested structure with NaN
    assert math.isnan(loaded["nested_nan"]["inner"][0])
    assert loaded["nested_nan"]["inner"][1] == 2.0
    assert np.isnan(loaded["nested_nan"]["array"][0])
    assert loaded["nested_nan"]["array"][1] == 1.0


def test_mne_store_complex_dtype_rejected(dummy_raw):
    # Verify that complex-valued numpy arrays are explicitly rejected
    raw = copy.deepcopy(dummy_raw)

    # Test with complex128
    payload_complex128 = {"complex_array": np.array([1 + 2j, 3 + 4j], dtype=np.complex128)}
    with pytest.raises(TypeError, match="Cannot serialize numpy array with complex dtype"):
        mne_store_metadata(raw, payload_complex128, key="test")

    # Test with complex64
    payload_complex64 = {"complex_array": np.array([1 + 2j], dtype=np.complex64)}
    with pytest.raises(TypeError, match="Cannot serialize numpy array with complex dtype"):
        mne_store_metadata(raw, payload_complex64, key="test")

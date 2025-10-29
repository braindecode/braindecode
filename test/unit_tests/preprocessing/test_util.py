# Authors: Christian Kothe <christian.kothe@intheon.io>
# License: BSD-3

import copy

import mne
import numpy as np
import pytest

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
        "numbers": [1, 2.5, -3, 1e10, float("inf")],
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

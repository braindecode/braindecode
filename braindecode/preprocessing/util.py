"""Utilities for preprocessing functionality in Braindecode."""

# Authors: Christian Kothe <christian.kothe@intheon.io>
#
# License: BSD-3

import base64
import json
import re
from typing import Any

import numpy as np
from mne.io.base import BaseRaw

__all__ = ["mne_store_metadata", "mne_load_metadata"]


# Use a unique marker for embedding structured data in info['description']
_MARKER_PATTERN = re.compile(r"<!-- braindecode-meta:\s*(\S+)\s*-->", re.DOTALL)
_MARKER_START = "<!-- braindecode-meta:"
_MARKER_END = "-->"

# Marker key for numpy arrays
_NP_ARRAY_TAG = "__numpy_array__"


def _numpy_decoder(dct):
    """Internal JSON decoder hook to handle numpy arrays."""
    if dct.get(_NP_ARRAY_TAG):
        arr = np.array(dct["data"], dtype=dct["dtype"])
        return arr.reshape(dct["shape"])
    return dct


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder hook to handle numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Reject complex-valued dtypes as they're not JSON serializable
            if np.issubdtype(obj.dtype, np.complexfloating):
                raise TypeError(
                    f"Cannot serialize numpy array with complex dtype {obj.dtype}. "
                    "Complex dtypes are not supported."
                )
            return {
                _NP_ARRAY_TAG: True,
                "dtype": obj.dtype.str,
                "shape": obj.shape,
                "data": obj.flatten().tolist(),
            }
        return super().default(obj)


def _encode_payload(data: dict) -> str:
    """Serializes, encodes, and formats data into a marker string."""
    json_str = json.dumps(data, cls=NumpyEncoder)
    encoded = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
    return f"{_MARKER_START} {encoded} {_MARKER_END}"


def mne_store_metadata(
    raw: BaseRaw, payload: Any, *, key: str, no_overwrite: bool = False
) -> None:
    """Embed a JSON-serializable metadata payload in an MNE BaseRaw dataset
    under a specified key.

    This will encode the payload as a base64-encoded JSON string and store it
    in the `info['description']` field of the Raw object while preserving any
    existing content. Note this is not particularly efficient and should not
    be used for very large payloads.

    Parameters
    ----------
    raw : BaseRaw
        The MNE Raw object to store data in.
    payload : Any
        The JSON-serializable data to store.
    key : str
        The key under which to store the payload.
    no_overwrite : bool
        If True, will not overwrite an existing entry with the same key.

    """
    # the description is apparently the only viable place where custom metadata may be
    # stored in MNE Raw objects that persists through saving/loading
    description = raw.info.get("description") or ""

    # Try to find existing eegprep data
    if match := _MARKER_PATTERN.search(description):
        # Parse existing data
        try:
            decoded = base64.b64decode(match.group(1)).decode("utf-8")
            existing_data = json.loads(decoded, object_hook=_numpy_decoder)
        except (ValueError, json.JSONDecodeError):
            existing_data = {}
        # Check no_overwrite condition
        if no_overwrite and key in existing_data:
            return
        # Update data
        existing_data[key] = payload
        new_marker = _encode_payload(existing_data)
        # Replace the old marker with updated one
        new_description = _MARKER_PATTERN.sub(new_marker, description, count=1)
    else:
        # No existing data, append new marker
        data = {key: payload}
        new_marker = _encode_payload(data)
        # Append with spacing if description exists
        if description.strip():
            new_description = f"{description.rstrip()}\n{new_marker}"
        else:
            new_description = new_marker

    raw.info["description"] = new_description


def mne_load_metadata(raw: BaseRaw, *, key: str, delete: bool = False) -> Any | None:
    """Retrieves data that was previously stored using mne_store_metadata from an MNE
    BaseRaw dataset.

    This function can retrieve data from an MNE Raw object that was stored
    using `mne_store_metadata`. It decodes the base64-encoded JSON string from the
    `info['description']` field and extracts the payload associated with the
    specified key.

    Parameters
    ----------
    raw : BaseRaw
        The MNE Raw object to retrieve data from.
    key : str
        The key under which the payload was stored.
    delete : bool
        If True, removes the key from the stored data after retrieval.

    Returns
    -------
    Any | None
        The retrieved payload, or None if not found.
    """
    description = raw.info.get("description") or ""
    match = _MARKER_PATTERN.search(description)
    if not match:
        return None

    try:
        decoded = base64.b64decode(match.group(1)).decode("utf-8")
        data = json.loads(decoded, object_hook=_numpy_decoder)
    except (ValueError, json.JSONDecodeError):
        return None

    result = data.get(key)

    if delete and key in data:
        # Remove the key from data
        del data[key]
        if data:
            # Still have other keys, update the marker
            new_marker = _encode_payload(data)
            new_description = _MARKER_PATTERN.sub(new_marker, description, count=1)
        else:
            # No more keys, remove the entire marker
            new_description = _MARKER_PATTERN.sub("", description, count=1).rstrip()
        raw.info["description"] = new_description

    return result

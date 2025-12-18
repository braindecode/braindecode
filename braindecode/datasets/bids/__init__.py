"""
BIDS (Brain Imaging Data Structure) integration for braindecode.

This subpackage provides:
- BIDS dataset loading (BIDSDataset, BIDSEpochsDataset)
- BIDS-like format utilities for Hub integration
- Hugging Face Hub push/pull functionality
- Validation utilities for Hub operations

More information on BIDS can be found at https://bids.neuroimaging.io
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Pierre Guetschel <pierre.guetschel@gmail.com>
#          Kuntal Kokate
#
# License: BSD (3-clause)

# Only import hub at init level (needed by base.py)
# Other imports are deferred to avoid circular imports
from .hub import HubDatasetMixin


def __getattr__(name):
    """Lazy imports to avoid circular dependencies."""
    if name in ("BIDSDataset", "BIDSEpochsDataset"):
        from .datasets import BIDSDataset, BIDSEpochsDataset

        return {"BIDSDataset": BIDSDataset, "BIDSEpochsDataset": BIDSEpochsDataset}[
            name
        ]
    elif name in (
        "BIDSDerivativesLayout",
        "create_channels_tsv",
        "create_eeg_json_sidecar",
        "create_events_tsv",
        "create_participants_tsv",
        "description_to_bids_path",
        "make_dataset_description",
    ):
        from . import format

        return getattr(format, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Dataset classes
    "BIDSDataset",
    "BIDSEpochsDataset",
    # Format utilities
    "BIDSDerivativesLayout",
    "create_channels_tsv",
    "create_eeg_json_sidecar",
    "create_events_tsv",
    "create_participants_tsv",
    "description_to_bids_path",
    "make_dataset_description",
    # Hub integration
    "HubDatasetMixin",
]

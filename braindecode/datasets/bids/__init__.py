# mypy: ignore-errors
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

from typing import TYPE_CHECKING

# Only import hub at init level (needed by base.py)
# Other imports are deferred to avoid circular imports
from .hub import HubDatasetMixin

# For static type checkers (mypy), provide explicit imports
if TYPE_CHECKING:
    from .datasets import BIDSDataset, BIDSEpochsDataset
    from .iterable import BIDSIterableDataset


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    if name in ("BIDSDataset", "BIDSEpochsDataset"):
        from .datasets import BIDSDataset, BIDSEpochsDataset

        return {"BIDSDataset": BIDSDataset, "BIDSEpochsDataset": BIDSEpochsDataset}[
            name
        ]
    elif name == "BIDSIterableDataset":
        from .iterable import BIDSIterableDataset

        return BIDSIterableDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Dataset classes
    "BIDSDataset",
    "BIDSEpochsDataset",
    "BIDSIterableDataset",
    # Hub integration
    "HubDatasetMixin",
]

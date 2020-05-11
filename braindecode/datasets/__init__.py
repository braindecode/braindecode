"""
Loader code for some datasets.
"""
from .base import WindowsDataset, BaseDataset, BaseConcatDataset
from .moabb import MOABBDataset
from .tuh import TUHAbnormal
from .mne import create_from_mne_raw, create_from_mne_epochs
from .xy import create_from_X_y

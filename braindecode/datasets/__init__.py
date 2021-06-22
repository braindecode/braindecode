"""
Loader code for some datasets.
"""
from .base import (
	WindowsDataset, SequenceWindowsDataset, BaseDataset, BaseConcatDataset,
	create_sequence_dataset)
from .moabb import MOABBDataset
from .mne import create_from_mne_raw, create_from_mne_epochs
from .tuh import TUH, TUHAbnormal
from .sleep_physionet import SleepPhysionet
from .xy import create_from_X_y

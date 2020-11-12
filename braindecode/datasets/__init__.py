"""
Loader code for some datasets.
"""
from .base import WindowsDataset, BaseDataset, BaseConcatDataset, AugmentedDataset
from .moabb import MOABBDataset
from .tuh import TUHAbnormal
from .sleep_physionet import SleepPhysionet

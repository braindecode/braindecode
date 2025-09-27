"""
Loader code for some datasets.
"""

from .base import BaseConcatDataset, BaseDataset, WindowsDataset
from .bcicomp import BCICompetitionIVDataset4
from .bids import BIDSDataset, BIDSEpochsDataset
from .mne import create_from_mne_epochs, create_from_mne_raw
from .moabb import BNCI2014001, HGD, MOABBDataset
from .nmt import NMT
from .sleep_physio_challe_18 import SleepPhysionetChallenge2018
from .sleep_physionet import SleepPhysionet
from .tuh import TUH, TUHAbnormal
from .xy import create_from_X_y

__all__ = [
    "WindowsDataset",
    "BaseDataset",
    "BaseConcatDataset",
    "BIDSDataset",
    "BIDSEpochsDataset",
    "MOABBDataset",
    "HGD",
    "BNCI2014001",
    "create_from_mne_raw",
    "create_from_mne_epochs",
    "TUH",
    "TUHAbnormal",
    "NMT",
    "SleepPhysionet",
    "SleepPhysionetChallenge2018",
    "create_from_X_y",
    "BCICompetitionIVDataset4",
]

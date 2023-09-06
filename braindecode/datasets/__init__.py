"""
Loader code for some datasets.
"""
# flake8: noqa
from .base import BaseConcatDataset, BaseDataset, WindowsDataset
from .bcicomp import BCICompetitionIVDataset4
from .mne import create_from_mne_epochs, create_from_mne_raw
from .moabb import BNCI2014001, HGD, MOABBDataset
from .sleep_physionet import SleepPhysionet
from .tuh import TUH, TUHAbnormal
from .xy import create_from_X_y

"""
Loader code for some datasets.
"""
from .base import WindowsDataset, BaseDataset, BaseConcatDataset
from .moabb import MOABBDataset, HGD, BNCI2014001
from .mne import create_from_mne_raw, create_from_mne_epochs
from .tuh import TUH, TUHAbnormal
from .sleep_physionet import SleepPhysionet
from .xy import create_from_X_y
from .bcicomp import BCICompetitionIVDataset4


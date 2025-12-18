"""
Loader code for some datasets.
"""

from .base import (
    BaseConcatDataset,
    EEGWindowsDataset,
    RawDataset,
    RecordDataset,
    WindowsDataset,
)
from .bcicomp import BCICompetitionIVDataset4
from .bids import (
    BIDSDataset,
    BIDSDerivativesLayout,
    BIDSEpochsDataset,
    create_channels_tsv,
    create_eeg_json_sidecar,
    create_events_tsv,
    create_participants_tsv,
    description_to_bids_path,
    make_dataset_description,
)
from .chb_mit import CHBMIT
from .mne import create_from_mne_epochs, create_from_mne_raw
from .moabb import BNCI2014_001, HGD, MOABBDataset
from .nmt import NMT
from .siena import SIENA
from .sleep_physio_challe_18 import SleepPhysionetChallenge2018
from .sleep_physionet import SleepPhysionet
from .tuh import TUH, TUHAbnormal
from .xy import create_from_X_y

__all__ = [
    "WindowsDataset",
    "EEGWindowsDataset",
    "RecordDataset",
    "RawDataset",
    "BaseConcatDataset",
    "BIDSDataset",
    "BIDSEpochsDataset",
    "BIDSDerivativesLayout",
    "create_events_tsv",
    "create_participants_tsv",
    "create_channels_tsv",
    "create_eeg_json_sidecar",
    "make_dataset_description",
    "description_to_bids_path",
    "MOABBDataset",
    "HGD",
    "BNCI2014_001",
    "create_from_mne_raw",
    "create_from_mne_epochs",
    "TUH",
    "TUHAbnormal",
    "SIENA",
    "NMT",
    "CHBMIT",
    "SleepPhysionet",
    "SleepPhysionetChallenge2018",
    "create_from_X_y",
    "BCICompetitionIVDataset4",
]

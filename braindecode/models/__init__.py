"""
Some predefined network architectures for EEG decoding.
"""

from .atcnet import ATCNet
from .attentionbasenet import AttentionBaseNet
from .attn_sleep import AttnSleep
from .base import EEGModuleMixin
from .biot import BIOT
from .contrawr import ContraWR
from .ctnet import CTNet
from .deep4 import Deep4Net
from .deepsleepnet import DeepSleepNet
from .eegconformer import EEGConformer
from .eeginception_erp import EEGInceptionERP
from .eeginception_mi import EEGInceptionMI
from .eegitnet import EEGITNet
from .eegminer import EEGMiner
from .eegnet import EEGNet, EEGNetv4
from .eegnex import EEGNeX
from .eegsimpleconv import EEGSimpleConv
from .eegtcnet import EEGTCNet
from .fbcnet import FBCNet
from .fblightconvnet import FBLightConvNet
from .fbmsnet import FBMSNet
from .hybrid import HybridNet
from .ifnet import IFNet
from .labram import Labram
from .msvtnet import MSVTNet
from .sccnet import SCCNet
from .shallow_fbcsp import ShallowFBCSPNet
from .signal_jepa import (
    SignalJEPA,
    SignalJEPA_Contextual,
    SignalJEPA_PostLocal,
    SignalJEPA_PreLocal,
)
from .sinc_shallow import SincShallowNet
from .sleep_stager_blanco_2020 import SleepStagerBlanco2020
from .sleep_stager_chambon_2018 import SleepStagerChambon2018
from .sparcnet import SPARCNet
from .syncnet import SyncNet
from .tcn import BDTCN, TCN
from .tidnet import TIDNet
from .tsinception import TSception
from .usleep import USleep
from .util import _init_models_dict, models_mandatory_parameters

# Call this last in order to make sure the dataset list is populated with
# the models imported in this file.
_init_models_dict()

__all__ = [
    "ATCNet",
    "AttnSleep",
    "AttentionBaseNet",
    "EEGModuleMixin",
    "BIOT",
    "ContraWR",
    "CTNet",
    "Deep4Net",
    "DeepSleepNet",
    "EEGConformer",
    "EEGInceptionERP",
    "EEGInceptionMI",
    "EEGITNet",
    "EEGMiner",
    "EEGNet",
    "EEGNetv4",
    "EEGNeX",
    "EEGSimpleConv",
    "EEGTCNet",
    "FBCNet",
    "FBLightConvNet",
    "FBMSNet",
    "HybridNet",
    "IFNet",
    "Labram",
    "MSVTNet",
    "SCCNet",
    "ShallowFBCSPNet",
    "SignalJEPA",
    "SignalJEPA_Contextual",
    "SignalJEPA_PostLocal",
    "SignalJEPA_PreLocal",
    "SincShallowNet",
    "SleepStagerBlanco2020",
    "SleepStagerChambon2018",
    "SPARCNet",
    "SyncNet",
    "BDTCN",
    "TCN",
    "TIDNet",
    "TSception",
    "USleep",
    "_init_models_dict",
    "models_mandatory_parameters",
]

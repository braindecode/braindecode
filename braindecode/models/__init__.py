"""Some predefined network architectures for EEG decoding."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type checkers and IDEs see the classes; runtime uses _LAZY
    from .atcnet import ATCNet
    from .attentionbasenet import AttentionBaseNet
    from .attn_sleep import AttnSleep
    from .axon import AXON
    from .barista import BaRISTA
    from .base import EEGModuleMixin
    from .bendr import BENDR
    from .biot import BIOT
    from .brainbert import BrainBERT
    from .brainmodule import BrainModule
    from .brainomni import BrainOmni, BrainTokenizer
    from .brant import Brant
    from .cbramod import CBraMod
    from .codebrain import CodeBrain
    from .contrawr import ContraWR
    from .csbrain import CSBrain
    from .ctnet import CTNet
    from .dance import DANCE
    from .deep4 import Deep4Net
    from .deepsleepnet import DeepSleepNet
    from .dgcnn import DGCNN
    from .diver1 import DIVER1
    from .duin import DuIN
    from .eeg_clip import EEGCLIP
    from .eegconformer import EEGConformer
    from .eegdino import EEGDINO
    from .eeginception_erp import EEGInceptionERP
    from .eeginception_mi import EEGInceptionMI
    from .eegitnet import EEGITNet
    from .eegminer import EEGMiner
    from .eegnet import EEGNet
    from .eegnex import EEGNeX
    from .eegpt import EEGPT
    from .eegsimpleconv import EEGSimpleConv
    from .eegsym import EEGSym
    from .eegtcnet import EEGTCNet
    from .emg2qwerty import EMG2QwertyNet
    from .epint import EpiNT
    from .fbcnet import FBCNet
    from .fblightconvnet import FBLightConvNet
    from .fbmsnet import FBMSNet
    from .gcnsnet import GCNsNet
    from .guetschel2026 import Guetschel2026
    from .hybrid import HybridNet
    from .ifnet import IFNet
    from .labram import Labram
    from .luna import LUNA
    from .mapa import MAPA
    from .medformer import MEDFormer
    from .meta_neuromotor import MetaNeuromotorHand
    from .mirepnet import MIRepNet
    from .mscformer import MSCFormer
    from .msvtnet import MSVTNet
    from .mvpformer import MVPFormer
    from .neuropose import NeuroPose
    from .neurorvq import NeuroRVQ
    from .neurorvq_tokenizer import NeuroRVQTokenizer
    from .patchedtransformer import PBT
    from .popt import PopulationTransformer
    from .reve import REVE
    from .sccnet import SCCNet
    from .seizure_transformer import SeizureTransformer
    from .sensingdynamics import SensingDynamics
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
    from .sleepfm import SleepFM, SleepFMStager
    from .sparcnet import SPARCNet
    from .sstdpn import SSTDPN
    from .steegformer import STEEGFormer
    from .syncnet import SyncNet
    from .tcformer import TCFormer
    from .tcn import BDTCN, TCN
    from .tfm_tokenizer import TFMTokenizer, TFMTokenizerOutput
    from .tidnet import TIDNet
    from .tmsanet import TMSANet
    from .tsinception import TSception
    from .usleep import USleep
    from .util import (
        _init_models_dict,
        build_model_config,
        extract_channel_locations_from_chs_info,
        models_mandatory_parameters,
        positions_from_chs_info,
    )
    from .vemg2pose import VEMG2Pose
    from .zuna import ZUNA


# Public name -> submodule that defines it. Submodules are imported on first
# attribute access (PEP 562), so ``import braindecode`` does not load every model.
_LAZY = {
    "ATCNet": "atcnet",
    "AttentionBaseNet": "attentionbasenet",
    "AttnSleep": "attn_sleep",
    "AXON": "axon",
    "BaRISTA": "barista",
    "EEGModuleMixin": "base",
    "BENDR": "bendr",
    "BIOT": "biot",
    "BrainBERT": "brainbert",
    "BrainModule": "brainmodule",
    "BrainOmni": "brainomni",
    "BrainTokenizer": "brainomni",
    "Brant": "brant",
    "CBraMod": "cbramod",
    "CodeBrain": "codebrain",
    "ContraWR": "contrawr",
    "CSBrain": "csbrain",
    "CTNet": "ctnet",
    "DANCE": "dance",
    "Deep4Net": "deep4",
    "DeepSleepNet": "deepsleepnet",
    "DGCNN": "dgcnn",
    "DIVER1": "diver1",
    "DuIN": "duin",
    "EEGCLIP": "eeg_clip",
    "EEGConformer": "eegconformer",
    "EEGDINO": "eegdino",
    "EEGInceptionERP": "eeginception_erp",
    "EEGInceptionMI": "eeginception_mi",
    "EEGITNet": "eegitnet",
    "EEGMiner": "eegminer",
    "EEGNet": "eegnet",
    "EEGNeX": "eegnex",
    "EEGPT": "eegpt",
    "EEGSimpleConv": "eegsimpleconv",
    "EEGSym": "eegsym",
    "EEGTCNet": "eegtcnet",
    "EMG2QwertyNet": "emg2qwerty",
    "EpiNT": "epint",
    "FBCNet": "fbcnet",
    "FBLightConvNet": "fblightconvnet",
    "FBMSNet": "fbmsnet",
    "GCNsNet": "gcnsnet",
    "Guetschel2026": "guetschel2026",
    "HybridNet": "hybrid",
    "IFNet": "ifnet",
    "Labram": "labram",
    "LUNA": "luna",
    "MAPA": "mapa",
    "MEDFormer": "medformer",
    "MetaNeuromotorHand": "meta_neuromotor",
    "MIRepNet": "mirepnet",
    "MSCFormer": "mscformer",
    "MSVTNet": "msvtnet",
    "MVPFormer": "mvpformer",
    "NeuroPose": "neuropose",
    "NeuroRVQ": "neurorvq",
    "NeuroRVQTokenizer": "neurorvq_tokenizer",
    "PBT": "patchedtransformer",
    "PopulationTransformer": "popt",
    "REVE": "reve",
    "SCCNet": "sccnet",
    "SeizureTransformer": "seizure_transformer",
    "SensingDynamics": "sensingdynamics",
    "ShallowFBCSPNet": "shallow_fbcsp",
    "SignalJEPA": "signal_jepa",
    "SignalJEPA_Contextual": "signal_jepa",
    "SignalJEPA_PostLocal": "signal_jepa",
    "SignalJEPA_PreLocal": "signal_jepa",
    "SincShallowNet": "sinc_shallow",
    "SleepStagerBlanco2020": "sleep_stager_blanco_2020",
    "SleepStagerChambon2018": "sleep_stager_chambon_2018",
    "SleepFM": "sleepfm",
    "SleepFMStager": "sleepfm",
    "SPARCNet": "sparcnet",
    "SSTDPN": "sstdpn",
    "STEEGFormer": "steegformer",
    "SyncNet": "syncnet",
    "TCFormer": "tcformer",
    "BDTCN": "tcn",
    "TCN": "tcn",
    "TFMTokenizer": "tfm_tokenizer",
    "TFMTokenizerOutput": "tfm_tokenizer",
    "TIDNet": "tidnet",
    "TMSANet": "tmsanet",
    "TSception": "tsinception",
    "USleep": "usleep",
    "_init_models_dict": "util",
    "build_model_config": "util",
    "extract_channel_locations_from_chs_info": "util",
    "models_mandatory_parameters": "util",
    "positions_from_chs_info": "util",
    "VEMG2Pose": "vemg2pose",
    "ZUNA": "zuna",
}


if not TYPE_CHECKING:  # hidden from checkers so misspelled names stay errors

    def __getattr__(name):
        if name in _LAZY:
            value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        elif name in _LAZY.values():
            value = importlib.import_module(f".{name}", __name__)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        globals()[name] = value
        return value

    def __dir__():
        return sorted(set(globals()) | set(__all__))


__all__ = [
    "ATCNet",
    "AXON",
    "AttnSleep",
    "AttentionBaseNet",
    "BaRISTA",
    "EEGModuleMixin",
    "BIOT",
    "BENDR",
    "BrainBERT",
    "BrainOmni",
    "BrainTokenizer",
    "CBraMod",
    "CodeBrain",
    "ContraWR",
    "CSBrain",
    "CTNet",
    "DANCE",
    "Deep4Net",
    "DeepSleepNet",
    "DIVER1",
    "DuIN",
    "BrainModule",
    "Brant",
    "EEGCLIP",
    "EEGConformer",
    "EEGDINO",
    "EEGPT",
    "EEGInceptionERP",
    "EEGInceptionMI",
    "EEGITNet",
    "EEGMiner",
    "EEGNet",
    "EEGNeX",
    "EEGSym",
    "EEGSimpleConv",
    "EEGTCNet",
    "DGCNN",
    "EMG2QwertyNet",
    "EpiNT",
    "NeuroPose",
    "SensingDynamics",
    "VEMG2Pose",
    "FBCNet",
    "FBLightConvNet",
    "FBMSNet",
    "GCNsNet",
    "Guetschel2026",
    "MetaNeuromotorHand",
    "HybridNet",
    "IFNet",
    "Labram",
    "LUNA",
    "extract_channel_locations_from_chs_info",
    "positions_from_chs_info",
    "MAPA",
    "MEDFormer",
    "MIRepNet",
    "NeuroRVQ",
    "MSCFormer",
    "NeuroRVQTokenizer",
    "MSVTNet",
    "MVPFormer",
    "PBT",
    "PopulationTransformer",
    "REVE",
    "SCCNet",
    "SeizureTransformer",
    "ShallowFBCSPNet",
    "SignalJEPA",
    "SignalJEPA_Contextual",
    "SignalJEPA_PostLocal",
    "SignalJEPA_PreLocal",
    "SincShallowNet",
    "SSTDPN",
    "SleepFM",
    "SleepFMStager",
    "SleepStagerBlanco2020",
    "SleepStagerChambon2018",
    "SPARCNet",
    "STEEGFormer",
    "SyncNet",
    "BDTCN",
    "TFMTokenizer",
    "TFMTokenizerOutput",
    "TCFormer",
    "TCN",
    "TIDNet",
    "TMSANet",
    "TSception",
    "USleep",
    "ZUNA",
    "build_model_config",
    "_init_models_dict",
    "models_mandatory_parameters",
]

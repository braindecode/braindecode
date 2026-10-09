"""Some predefined network architectures for EEG decoding."""

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
from .fbcnet import FBCNet
from .fblightconvnet import FBLightConvNet
from .fbmsnet import FBMSNet
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

# Call this last in order to make sure the dataset list is populated with
# the models imported in this file.
_init_models_dict()

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
    "NeuroPose",
    "SensingDynamics",
    "VEMG2Pose",
    "FBCNet",
    "FBLightConvNet",
    "FBMSNet",
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

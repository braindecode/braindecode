"""Some predefined network architectures for EEG decoding."""

import ast
import importlib
from typing import TYPE_CHECKING

_LAZY: dict[str, str]  # public name -> submodule, read below from the imports

# A new model needs one import line here. It gives type checkers the class and,
# read back at runtime, the lazy name -> submodule map and ``__all__``; so these
# imports are used although no code names them.
# ruff: noqa: F401
if TYPE_CHECKING:
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
else:  # hidden from checkers so misspelled names stay errors

    def _type_checking_imports():
        # ponytail: needs this file's source, as get_summary_table needs summary.csv.
        with open(__file__, "rb") as source:
            tree = ast.parse(source.read())
        block = next(node for node in tree.body if isinstance(node, ast.If))
        return {alias.name: imp.module for imp in block.body for alias in imp.names}

    # Each submodule is imported on first attribute access (PEP 562), so
    # ``import braindecode`` does not load every model.
    _LAZY = _type_checking_imports()
    __all__ = list(_LAZY)

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

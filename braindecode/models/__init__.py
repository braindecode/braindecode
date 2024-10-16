"""
Some predefined network architectures for EEG decoding.
"""

from .attentionbasenet import AttentionBaseNet
from .base import EEGModuleMixin
from .biot import BIOT
from .eegconformer import EEGConformer
from .eegitnet import EEGITNet
from .deep4 import Deep4Net
from .deepsleepnet import DeepSleepNet
from .eegnet import EEGNetv4, EEGNetv1
from .hybrid import HybridNet
from .shallow_fbcsp import ShallowFBCSPNet
from .eegresnet import EEGResNet
from .eeginception_erp import EEGInceptionERP
from .eeginception_mi import EEGInceptionMI
from .atcnet import ATCNet
from .tcn import TCN
from .sleep_stager_chambon_2018 import SleepStagerChambon2018
from .sleep_stager_blanco_2020 import SleepStagerBlanco2020
from .sleep_stager_eldele_2021 import SleepStagerEldele2021
from .tidnet import TIDNet
from .usleep import USleep
from .util import get_output_shape, to_dense_prediction_model
from .modules import TimeDistributed
from .util import _init_models_dict, models_mandatory_parameters
from .labram import Labram
from .eegsimpleconv import EEGSimpleConv
from .sparcnet import SPARCNet
from .contrawr import ContraWR
from .eegnex import EEGNeX
from .tsinception import TSceptionV1
from .eegtcnet import EEGTCNet
from .syncnet import SyncNet
from .fbeegnet import FBEEGNet
from .sinc_shallow import SincShallowNet
from .lmda import LMDANet
from .sccnet import SCCNet
from .fbcnet import FBCNet
from .ifnet import IFNetV2
from .eegminer import EEGMiner
from .fblightconvnet import FBLightConvNet
from .fbmsnet import FBMSNet
from .ctnet import CTNet


# Call this last in order to make sure the dataset list is populated with
# the models imported in this file.
_init_models_dict()

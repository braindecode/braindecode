"""
Some predefined network architectures for EEG decoding.
"""
from .atcnet import ATCNet
from .deep4 import Deep4Net
from .deepsleepnet import DeepSleepNet
# flake8: noqa
from .eegconformer import EEGConformer
from .eeginception import EEGInception
from .eeginception_erp import EEGInceptionERP
from .eeginception_mi import EEGInceptionMI
from .eegitnet import EEGITNet
from .eegnet import EEGNetv1, EEGNetv4
from .eegresnet import EEGResNet
from .hybrid import HybridNet
from .modules import TimeDistributed
from .shallow_fbcsp import ShallowFBCSPNet
from .sleep_stager_blanco_2020 import SleepStagerBlanco2020
from .sleep_stager_chambon_2018 import SleepStagerChambon2018
from .sleep_stager_eldele_2021 import SleepStagerEldele2021
from .tcn import TCN
from .tidnet import TIDNet
from .usleep import USleep
from .util import get_output_shape, to_dense_prediction_model

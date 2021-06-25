"""
Some predefined network architectures for EEG decoding.
"""

from .deep4 import Deep4Net
from .eegnet import EEGNetv4, EEGNetv1
from .hybrid import HybridNet
from .shallow_fbcsp import ShallowFBCSPNet
from .eegresnet import EEGResNet
from .tcn import TCN
from .sleep_stager_chambon_2018 import SleepStagerChambon2018
from .util import get_output_shape
from .usleep import USleep

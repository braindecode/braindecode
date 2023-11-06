# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from ..datasets.mne import *  # noqa: F401,F403
from warnings import warn

warn('datautil.mne module is deprecated and is now under '
     'datasets.mne, please use from import braindecode.datasets.mne')

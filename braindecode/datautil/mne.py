# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from warnings import warn

from ..datasets.mne import *  # noqa: F401,F403

warn('datautil.mne module is deprecated and is now under '
     'datasets.mne, please use from import braindecode.datasets.mne')

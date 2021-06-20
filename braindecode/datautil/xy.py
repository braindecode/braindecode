# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from warnings import warn
from ..datasets.xy import *  # noqa: F401,F403

warn('datautil.xy module is deprecated and is now under '
     'datasets.xy, please use from import braindecode.datasets.xy')

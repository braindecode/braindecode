# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

# To be removed in future versions

from ..preprocessing.preprocess import *  # noqa: F401,F403
from warnings import warn

warn('datautil.preprocess module is deprecated and is now under '
     'preprocessing.preprocess, please use from import '
     'braindecode.preprocessing.preprocess')

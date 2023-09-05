# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

# To be removed in future versions

from warnings import warn

from ..preprocessing.preprocess import *  # noqa: F401,F403

warn('datautil.preprocess module is deprecated and is now under '
     'preprocessing.preprocess, please use from import '
     'braindecode.preprocessing.preprocess')

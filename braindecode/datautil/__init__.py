"""
Utilities for data manipulation.
"""

from .windowers import create_windows_from_events, create_fixed_length_windows
from .preprocess import zscore, scale, exponential_moving_demean, exponential_moving_standardize, filterbank
from .transform_side_funcs import mask_along_axis
from .xy import create_from_X_y
from .mne import create_from_mne_raw, create_from_mne_epochs
from .serialization import save_concat_dataset, load_concat_dataset

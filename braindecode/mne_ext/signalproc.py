import logging
from copy import deepcopy

import numpy as np
from mne.io.base import concatenate_raws
import mne

log = logging.getLogger(__name__)


def concatenate_raws_with_events(raws):
    """
    Concatenates `mne.io.RawArray` objects, respects `info['events']` attributes
    and concatenates them correctly. Also does not modify `raws[0]` inplace
    as the :func:`concatenate_raws` function of MNE does.
    
    Parameters
    ----------
    raws: list of `mne.io.RawArray`

    Returns
    -------
    concatenated_raw: `mne.io.RawArray`
    """
    # prevent in-place modification of raws[0]
    raws[0] = deepcopy(raws[0])
    event_lists = [r.info["events"] for r in raws]
    new_raw, new_events = concatenate_raws(raws, events_list=event_lists)
    new_raw.info["events"] = new_events
    return new_raw


def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.
    
    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)


def common_average_reference_cnt(cnt,):
    """
    Common average reference, subtract average over electrodes at each timestep.

    Parameters
    ----------
    cnt: `mne.io.RawArray`

    Returns
    -------
    car_cnt: cnt: `mne.io.RawArray`
        Same data after common average reference.
    """

    return mne_apply(lambda a: a - np.mean(a, axis=0, keepdim=True), cnt)

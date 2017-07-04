import logging
from copy import deepcopy

import resampy
from mne.io.base import concatenate_raws
import mne

log = logging.getLogger(__name__)


def concatenate_raws_with_events(raws):
    """
    Concatenates `Raw` mne objects, respects info['events'] attributes
    and concatenates them correctly. Also does not modify raws[0] inplace
    as the concatenate_raws function of mne does
    Parameters
    ----------
    raws: list of `Raw` mne objects

    Returns
    -------
    concatenated_raw: `Raw` mne object
    """
    # prevent in-place modification of raws[0]
    raws[0] = deepcopy(raws[0])
    event_lists = [r.info['events'] for r in raws]
    new_raw, new_events = concatenate_raws(raws, events_list=event_lists)
    new_raw.info['events'] = new_events
    return new_raw


def resample_cnt(cnt, new_fs):
    if new_fs == cnt.info['sfreq']:
        log.info(
            "Just copying data, no resampling, since new sampling rate same.")
        return deepcopy(cnt)
    log.warning("This is not causal, uses future data....")
    log.info("Resampling from {:f} to {:f} Hz.".format(
        cnt.info['sfreq'], new_fs
    ))

    data = cnt.get_data().T

    new_data = resampy.resample(data, cnt.info['sfreq'],
                                new_fs, axis=0, filter='kaiser_fast').T
    old_fs = cnt.info['sfreq']
    new_info = deepcopy(cnt.info)
    new_info['sfreq'] = new_fs
    events = new_info['events']
    event_samples_old = cnt.info['events'][:, 0]
    event_samples = event_samples_old * new_fs / float(old_fs)
    events[:, 0] = event_samples
    return mne.io.RawArray(new_data, new_info)


def mne_apply(func, raw, verbose='WARNING'):
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)

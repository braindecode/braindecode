import logging

from braindecode.mne_ext.signalproc import resample_cnt, concatenate_raws_with_events

log = logging.getLogger(__name__)


class MultipleSetLoader(object):
    """
    Class to load multiple sets.
    
    Resamples individual sets down to lowest common sampling frequency, 
    if necessary.

    Parameters
    ----------
    set_loaders: object with load method: () -> MNE Raw object
        The individual set loaders.
    """

    def __init__(self, set_loaders):
        self.set_loaders = set_loaders

    def load(self):
        cnt = self.set_loaders[0].load()
        for loader in self.set_loaders[1:]:
            next_cnt = loader.load()
            # always sample down to lowest common denominator
            if next_cnt.fs > cnt.fs:
                log.warning(
                    "Next set has larger sampling rate ({:d}) "
                    "than before ({:d}), resampling next set".format(
                        next_cnt.fs, cnt.fs
                    )
                )
                next_cnt = resample_cnt(next_cnt, cnt.fs)
            if next_cnt.fs < cnt.fs:
                log.warning(
                    "Next set has smaller sampling rate ({:d}) "
                    "than before ({:d}), resampling set so far".format(
                        next_cnt.fs, cnt.fs
                    )
                )
                cnt = resample_cnt(cnt, next_cnt.fs)
            cnt = concatenate_raws_with_events(cnt, next_cnt)
        return cnt

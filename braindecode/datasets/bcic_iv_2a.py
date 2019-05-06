import numpy as np
import mne
from scipy.io import loadmat


class BCICompetition4Set2A(object):
    def __init__(self, filename, load_sensor_names=None, labels_filename=None):
        assert load_sensor_names is None
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info["events"] = events
        cnt.info["artifact_trial_mask"] = artifact_trial_mask
        return cnt

    def extract_data(self):
        raw_edf = mne.io.read_raw_edf(self.filename, stim_channel="auto")
        raw_edf.load_data()
        # correct nan values

        data = raw_edf.get_data()

        # do not correct stimulus channel
        assert raw_edf.ch_names[-1] == "STI 014"
        for i_chan in range(data.shape[0] - 1):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean
        gdf_events = raw_edf.find_edf_events()
        raw_edf = mne.io.RawArray(data, raw_edf.info, verbose="WARNING")
        # remember gdf events
        raw_edf.info["gdf_events"] = gdf_events
        return raw_edf

    def extract_events(self, raw_edf):
        # all events
        events = np.array(
            list(zip(raw_edf.info["gdf_events"][1], raw_edf.info["gdf_events"][2]))
        )

        # only trial onset events
        trial_mask = [ev_code in [769, 770, 771, 772, 783] for ev_code in events[:, 1]]
        trial_events = events[trial_mask]
        assert len(trial_events) == 288, "Got {:d} markers".format(len(trial_events))
        # event markers 769,770 -> 1,2
        trial_events[:, 1] = trial_events[:, 1] - 768
        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            trial_events[:, 1] = classes
        unique_classes = np.unique(trial_events[:, 1])
        assert np.array_equal(
            [1, 2, 3, 4], unique_classes
        ), "Expect 1,2,3,4 as class labels, got {:s}".format(str(unique_classes))
        # now also create 0-1 vector for rejected trials
        trial_start_events = events[events[:, 1] == 768]
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 1] == 1023]
        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        # mne expects events with 3 ints each:
        events = np.zeros((len(trial_events), 3), dtype=np.int32)
        events[:, 0] = trial_events[:, 0]
        events[:, 2] = trial_events[:, 1]

        return events, artifact_trial_mask

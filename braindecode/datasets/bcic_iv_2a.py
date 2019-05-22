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
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto")
        raw_gdf.load_data()
        # correct nan values

        data = raw_gdf.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean

        gdf_events = mne.events_from_annotations(raw_gdf)
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="WARNING")
        # remember gdf events
        raw_gdf.info["gdf_events"] = gdf_events
        return raw_gdf

    def extract_events(self, raw_gdf):
        # all events
        events, name_to_code = raw_gdf.info["gdf_events"]

        if "class1, Left hand - cue onset (BCI experiment)" in name_to_code:
            train_set = True
        else:
            train_set = False
            assert (
                "cue unknown/undefined (used for BCI competition) "
                in name_to_code
            )

        if train_set:
            trial_codes = [4, 5, 6, 7]  # the 4 classes
        else:
            trial_codes = [4]  # "unknown" class

        trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
        trial_events = events[trial_mask]
        assert len(trial_events) == 288, "Got {:d} markers".format(
            len(trial_events)
        )
        trial_events[:, 2] = trial_events[:, 2] - 3
        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            [1, 2, 3, 4], unique_classes
        ), "Expect 1,2,3,4 as class labels, got {:s}".format(
            str(unique_classes)
        )

        # now also create 0-1 vector for rejected trials
        trial_start_events = events[events[:, 2] == 2]
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]

        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        return trial_events, artifact_trial_mask

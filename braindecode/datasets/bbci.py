import logging
import re
from glob import glob
import os.path

import numpy as np
import h5py
import mne

log = logging.getLogger(__name__)


class BBCIDataset(object):
    """
    Loader class for files created by saving BBCI files in matlab (make
    sure to save with '-v7.3' in matlab, see
    https://de.mathworks.com/help/matlab/import_export/mat-file-versions.html#buk6i87
    )

    Parameters
    ----------
    filename: str
    load_sensor_names: list of str, optional
        Also speeds up loading if you only load some sensors.
        None means load all sensors.
    check_class_names: bool, optional
        check if the class names are part of some known class names at
        Translational NeuroTechnology Lab, AG Ball, Freiburg, Germany.
    """

    def __init__(self, filename, load_sensor_names=None, check_class_names=False):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt = self._load_continuous_signal()
        cnt = self._add_markers(cnt)
        return cnt

    def _load_continuous_signal(self):
        wanted_chan_inds, wanted_sensor_names = self._determine_sensors()
        fs = self._determine_samplingrate()
        with h5py.File(self.filename, "r") as h5file:
            samples = int(h5file["nfo"]["T"][0, 0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.ones(cnt_signal_shape, dtype=np.float32) * np.nan
            for chan_ind_arr, chan_ind_set in enumerate(wanted_chan_inds):
                # + 1 because matlab/this hdf5-naming logic
                # has 1-based indexing
                # i.e ch1,ch2,....
                chan_set_name = "ch" + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][
                    :
                ].squeeze()  # already load into memory
                continuous_signal[:, chan_ind_arr] = chan_signal
            assert not np.any(np.isnan(continuous_signal)), "No NaNs expected in signal"

        if self.load_sensor_names is None:
            ch_types = ["EEG"] * len(wanted_chan_inds)
        else:
            # Assume we cant know channel type here automatically
            ch_types = ["misc"] * len(wanted_chan_inds)
        info = mne.create_info(
            ch_names=wanted_sensor_names, sfreq=fs, ch_types=ch_types
        )

        cnt = mne.io.RawArray(continuous_signal.T, info)
        return cnt

    def _determine_sensors(self):
        all_sensor_names = self.get_all_sensors(self.filename, pattern=None)
        if self.load_sensor_names is None:

            # if no sensor names given, take all EEG-chans
            eeg_sensor_names = all_sensor_names
            eeg_sensor_names = filter(
                lambda s: not s.startswith("BIP"), eeg_sensor_names
            )
            eeg_sensor_names = filter(lambda s: not s.startswith("E"), eeg_sensor_names)
            eeg_sensor_names = filter(
                lambda s: not s.startswith("Microphone"), eeg_sensor_names
            )
            eeg_sensor_names = filter(
                lambda s: not s.startswith("Breath"), eeg_sensor_names
            )
            eeg_sensor_names = filter(
                lambda s: not s.startswith("GSR"), eeg_sensor_names
            )
            eeg_sensor_names = list(eeg_sensor_names)
            assert (
                len(eeg_sensor_names) == 128
                or len(eeg_sensor_names) == 64
                or len(eeg_sensor_names) == 32
                or len(eeg_sensor_names) == 16
            ), "Recheck this code if you have different sensors..."
            self.load_sensor_names = eeg_sensor_names
        chan_inds = self._determine_chan_inds(all_sensor_names, self.load_sensor_names)
        return chan_inds, self.load_sensor_names

    def _determine_samplingrate(self):
        with h5py.File(self.filename, "r") as h5file:
            fs = h5file["nfo"]["fs"][0, 0]
            assert isinstance(fs, int) or fs.is_integer()
            fs = int(fs)
        return fs

    @staticmethod
    def _determine_chan_inds(all_sensor_names, sensor_names):
        assert sensor_names is not None
        chan_inds = [all_sensor_names.index(s) for s in sensor_names]
        assert len(chan_inds) == len(sensor_names), "All" "sensors should be there."
        assert len(set(chan_inds)) == len(chan_inds), "No" "duplicated sensors wanted."
        return chan_inds

    @staticmethod
    def get_all_sensors(filename, pattern=None):
        """
        Get all sensors that exist in the given file.
        
        Parameters
        ----------
        filename: str
        pattern: str, optional
            Only return those sensor names that match the given pattern.

        Returns
        -------
        sensor_names: list of str
            Sensor names that match the pattern or all sensor names in the file.
        """
        with h5py.File(filename, "r") as h5file:
            clab_set = h5file["nfo"]["clab"][:].squeeze()
            all_sensor_names = [
                "".join(chr(c) for c in h5file[obj_ref]) for obj_ref in clab_set
            ]
            if pattern is not None:
                all_sensor_names = filter(
                    lambda sname: re.search(pattern, sname), all_sensor_names
                )
        return all_sensor_names

    def _add_markers(self, cnt):
        with h5py.File(self.filename, "r") as h5file:
            event_times_in_ms = h5file["mrk"]["time"][:].squeeze()
            event_classes = h5file["mrk"]["event"]["desc"][:].squeeze().astype(np.int64)

            # Check whether class names known and correct order
            class_name_set = h5file["nfo"]["className"][:].squeeze()
            all_class_names = [
                "".join(chr(c) for c in h5file[obj_ref]) for obj_ref in class_name_set
            ]

            if self.check_class_names:
                _check_class_names(all_class_names, event_times_in_ms, event_classes)

        event_times_in_samples = event_times_in_ms * cnt.info["sfreq"] / 1000.0
        event_times_in_samples = np.uint32(np.round(event_times_in_samples))

        # Check if there are markers at the same time
        previous_i_sample = -1
        for i_event, (i_sample, id_class) in enumerate(
            zip(event_times_in_samples, event_classes)
        ):
            if i_sample == previous_i_sample:
                log.warning(
                    "Same sample has at least two markers.\n"
                    "{:d}: ({:.0f} and {:.0f}).\n".format(
                        i_sample, event_classes[i_event - 1], event_classes[i_event]
                    )
                    + "Marker codes will be summed."
                )
            previous_i_sample = i_sample

        # Now create stim chan
        stim_chan = np.zeros_like(cnt.get_data()[0])
        for i_sample, id_class in zip(event_times_in_samples, event_classes):
            stim_chan[i_sample] += id_class
        info = mne.create_info(
            ch_names=["STI 014"], sfreq=cnt.info["sfreq"], ch_types=["stim"]
        )
        stim_cnt = mne.io.RawArray(stim_chan[None], info, verbose="WARNING")
        cnt = cnt.add_channels([stim_cnt])
        event_arr = [
            event_times_in_samples,
            [0] * len(event_times_in_samples),
            event_classes,
        ]
        cnt.info["events"] = np.array(event_arr).T
        return cnt


def _check_class_names(all_class_names, event_times_in_ms, event_classes):
    """
    Checks if the class names are part of some known class names used in
    translational neurotechnology lab, AG Ball, Freiburg.
    
    Logs warning in case class names are not known. 
    
    Parameters
    ----------
    all_class_names: list of str
    event_times_in_ms: list of number
    event_classes: list of number

    """
    if all_class_names == ["Right Hand", "Left Hand", "Rest", "Feet"]:
        pass
    elif (
        (
            all_class_names
            == [
                "1",
                "10",
                "11",
                "111",
                "12",
                "13",
                "150",
                "2",
                "20",
                "22",
                "3",
                "30",
                "33",
                "4",
                "40",
                "44",
                "99",
            ]
        )
        or (
            all_class_names
            == [
                "1",
                "10",
                "11",
                "12",
                "13",
                "150",
                "2",
                "20",
                "22",
                "3",
                "30",
                "33",
                "4",
                "40",
                "44",
                "99",
            ]
        )
        or (all_class_names == ["1", "2", "3", "4"])
    ):
        pass  # Semantic classes
    elif all_class_names == ["Rest", "Feet", "Left Hand", "Right Hand"]:
        # Have to swap from
        # ['Rest', 'Feet', 'Left Hand', 'Right Hand']
        # to
        # ['Right Hand', 'Left Hand', 'Rest', 'Feet']
        right_mask = event_classes == 4
        left_mask = event_classes == 3
        rest_mask = event_classes == 1
        feet_mask = event_classes == 2
        event_classes[right_mask] = 1
        event_classes[left_mask] = 2
        event_classes[rest_mask] = 3
        event_classes[feet_mask] = 4
        log.warn(
            "Swapped  class names {:s}... might cause problems...".format(
                all_class_names
            )
        )
    elif all_class_names == [
        "Right Hand Start",
        "Left Hand Start",
        "Rest Start",
        "Feet Start",
        "Right Hand End",
        "Left Hand End",
        "Rest End",
        "Feet End",
    ]:
        pass
    elif all_class_names == [
        "Right Hand",
        "Left Hand",
        "Rest",
        "Feet",
        "Face",
        "Navigation",
        "Music",
        "Rotation",
        "Subtraction",
        "Words",
    ]:
        pass  # robot hall 10 class decoding
    elif all_class_names == [
        "RightHand",
        "Feet",
        "Rotation",
        "Words",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "RightHand_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Feet_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Rotation_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Words_End",
    ] or all_class_names == [
        "RightHand",
        "Feet",
        "Rotation",
        "Words",
        "Rest",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "RightHand_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Feet_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Rotation_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Words_End",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "\x00\x00",
        "Rest_End",
    ]:
        pass  # weird stuff when we recorded cursor in robot hall
        # on 2016-09-14 and 2016-09-16 :D

    elif all_class_names == [
        "0004",
        "0016",
        "0032",
        "0056",
        "0064",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == ["0004", "0056", "0088", "0120"]:
        pass
    elif all_class_names == [
        "0004",
        "0016",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == ["0004", "0016", "0056", "0088", "0120", "__"]:
        pass
    elif all_class_names == ["0004", "0056", "0088", "0120", "__"]:
        pass
    elif all_class_names == [
        "0004",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
        "__",
    ]:
        pass
    elif all_class_names == ["0004", "0056", "0080", "0088", "0096", "0120", "__"]:
        pass
    elif all_class_names == [
        "0004",
        "0032",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == [
        "0004",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0120",
    ]:
        pass
    elif all_class_names == [
        "0004",
        "0016",
        "0032",
        "0048",
        "0056",
        "0064",
        "0080",
        "0088",
        "0095",
        "0096",
        "0120",
    ]:
        pass
    elif all_class_names == ["4", "16", "32", "56", "64", "88", "95", "120"]:
        pass
    elif all_class_names == ["4", "56", "88", "120"]:
        pass
    elif all_class_names == [
        "4",
        "16",
        "32",
        "48",
        "56",
        "64",
        "80",
        "88",
        "95",
        "120",
    ]:
        pass
    elif all_class_names == ["0", "4", "56", "88", "120"]:
        pass
    elif all_class_names == ["0", "4", "16", "56", "88", "120"]:
        pass
    elif all_class_names == ["0", "4", "32", "48", "56", "64", "80", "88", "95", "120"]:
        pass
    elif all_class_names == ["0", "4", "56", "80", "88", "96", "120"]:
        pass
    elif all_class_names == ["4", "32", "56", "64", "80", "88", "95", "120"]:
        pass
    elif all_class_names == ["One", "Two", "Three", "Four"]:
        pass
    elif all_class_names == ["1", "10", "11", "12", "2", "20", "3", "30", "4", "40"]:
        pass
    elif all_class_names == ["1", "10", "12", "13", "2", "20", "3", "30", "4", "40"]:
        pass
    elif all_class_names == ["1", "10", "13", "2", "20", "3", "30", "4", "40", "99"]:
        pass
    elif all_class_names == [
        "1",
        "10",
        "11",
        "14",
        "18",
        "20",
        "21",
        "24",
        "251",
        "252",
        "28",
        "30",
        "4",
        "8",
    ]:
        pass
    elif all_class_names == [
        "1",
        "10",
        "11",
        "14",
        "18",
        "20",
        "21",
        "24",
        "252",
        "253",
        "28",
        "30",
        "4",
        "8",
    ]:
        pass
    elif len(event_times_in_ms) == len(all_class_names):
        pass  # weird neuroone(?) logic where class names have event classes
    elif all_class_names == [
        "Right_hand_stimulus_onset",
        "Feet_stimulus_onset",
        "Rotation_stimulus_onset",
        "Words_stimulus_onset",
        "Right_hand_stimulus_offset",
        "Feet_stimulus_offset",
        "Rotation_stimulus_offset",
        "Words_stimulus_offset",
    ]:
        pass
    else:
        # remove this whole if else stuffs?
        log.warn("Unknown class names {:s}".format(all_class_names))


def load_bbci_sets_from_folder(folder, runs="all"):
    """
    Load bbci datasets from files in given folder.
    
    Parameters
    ----------
    folder: str
        Folder with .BBCI.mat files inside
    runs: list of int 
        If you only want to load specific runs.
        Assumes filenames with such kind of part: S001R02 for Run 2.
        Tries to match this regex: ``'S[0-9]{3,3}R[0-9]{2,2}_'``.

    Returns
    -------

    """
    bbci_mat_files = sorted(glob(os.path.join(folder, "*.BBCI.mat")))
    if runs != "all":
        file_run_numbers = [
            int(re.search("S[0-9]{3,3}R[0-9]{2,2}_", f).group()[5:7])
            for f in bbci_mat_files
        ]
        indices = [file_run_numbers.index(num) for num in runs]

        wanted_files = np.array(bbci_mat_files)[indices]
    else:
        wanted_files = bbci_mat_files
    cnts = []
    for f in wanted_files:
        log.info("Loading {:s}".format(f))
        cnts.append(BBCIDataset(f).load())
    return cnts

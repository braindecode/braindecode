"""
=================
Creating Datasets
=================

"""
# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict
import mne
from moabb.datasets import BNCI2014001
from mne import find_events

# preprocessing
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)

###############################################################################
# Fetching and downloading data with MOABB
class BNCI2014001Dataset():
    """
    """
    def __init__(self, subject, path=None, force_update=False):
        """
        0- Check whether files exist given path 
        1- Download files if they don't exist
        """

        ival = [-500, 4000]
        input_time_length = 1000
        max_epochs = 800
        max_increase_epochs = 80
        batch_size = 60
        low_cut_hz = 4
        high_cut_hz = 38
        factor_new = 1e-3
        init_block_size = 1000
        valid_set_fraction = 0.2

        self.subject = [subject] if isinstance(subject, int) else subject
        data = BNCI2014001().get_data(self.subject)

        state_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                                  ('Foot', [3]), ('Tongue', [4])])

        base_datasets = list()
        for subj_id, subj_data in data.items():
            for sess_id, sess_data in subj_data.items():
                for run_id, raw in sess_data.items():
                    # 0 - Get events and remove stim channel
                    raw.info['session'] = sess_id
                    raw.info['run'] = run_id
                    raw.info['events'] = find_events(raw)
                    raw.drop_channels(['EOG1', 'EOG2', 'EOG3', 'stim'])

                    # 1- Apply preprocessing
                    raw.apply_function(lambda a: a * 1e6)
                    raw.apply_function(lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, raw.info['sfreq'],
                                                              filt_order=3, axis=1))
                    raw.apply_function(lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                                                 init_block_size=init_block_size,
                                                                                 eps=1e-4))
                    # 2- Epoch
                    # 3- Create BaseDataset

                    pass

        # Concatenate datasets
        ConcatDataset(...)
        

# ###############################################################################
# # Let's create a dataset!

bnci = BNCI2014001Dataset([1, 2])



# data_path = sample.data_path()
# raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# raw = mne.io.read_raw_fif(raw_fname).crop(120, 240).load_data()

# ###############################################################################
# # Since downsampling reduces the timing precision of events, we recommend
# # first extracting epochs and downsampling the Epochs object:
# events = mne.find_events(raw)
# epochs = mne.Epochs(raw, events, event_id=2, tmin=-0.1, tmax=0.8, preload=True)

# # Downsample to 100 Hz
# print('Original sampling rate:', epochs.info['sfreq'], 'Hz')
# epochs_resampled = epochs.copy().resample(100, npad='auto')
# print('New sampling rate:', epochs_resampled.info['sfreq'], 'Hz')

# # Plot a piece of data to see the effects of downsampling
# plt.figure(figsize=(7, 3))

# n_samples_to_plot = int(0.5 * epochs.info['sfreq'])  # plot 0.5 seconds of data
# plt.plot(epochs.times[:n_samples_to_plot],
#          epochs.get_data()[0, 0, :n_samples_to_plot], color='black')

# n_samples_to_plot = int(0.5 * epochs_resampled.info['sfreq'])
# plt.plot(epochs_resampled.times[:n_samples_to_plot],
#          epochs_resampled.get_data()[0, 0, :n_samples_to_plot],
#          '-o', color='red')

# plt.xlabel('time (s)')
# plt.legend(['original', 'downsampled'], loc='best')
# plt.title('Effect of downsampling')
# mne.viz.tight_layout()


# ###############################################################################
# # When resampling epochs is unwanted or impossible, for example when the data
# # doesn't fit into memory or your analysis pipeline doesn't involve epochs at
# # all, the alternative approach is to resample the continuous data. This
# # can only be done on loaded or pre-loaded data.

# # Resample to 300 Hz
# raw_resampled_300 = raw.copy().resample(300, npad='auto')



# # Resample to 100 Hz (suppress the warning that would be emitted)
# raw_resampled_100 = raw.copy().resample(100, npad='auto', verbose='error')
# print('Number of events after resampling:',
#       len(mne.find_events(raw_resampled_100)))

# # To avoid losing events, jointly resample the data and event matrix
# events = mne.find_events(raw)
# raw_resampled, events_resampled = raw.copy().resample(
#     100, npad='auto', events=events)
# print('Number of events after resampling:', len(events_resampled))
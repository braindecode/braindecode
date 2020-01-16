"""
Trialwise Decoding
==================

In this example, we will use a convolutional neural network on the
[Physiobank EEG Motor Movement/Imagery Dataset](https://www.physionet.org/physiobank/database/eegmmidb/) to decode two classes:

1. Executed and imagined opening and closing of both hands
2. Executed and imagined opening and closing of both feet

.. warning::

    We use only one subject (with 90 trials) in this tutorial for demonstration
    purposes. A more interesting decoding task with many more trials would be
    to do cross-subject decoding on the same dataset.

"""

##############################################################################
# Load data
# ---------

# You can load and preprocess your EEG dataset in any way,
# Braindecode only expects a 3darray (trials, channels, timesteps) of input
# signals `X` and a vector of labels `y` later (see below). In this tutorial,
# we will use the [MNE](https://www.martinos.org/mne/stable/index.html)
# library to load an EEG motor imagery/motor execution dataset. For a
# tutorial from MNE using Common Spatial Patterns to decode this data, see
# [here](http://martinos.org/mne/stable/auto_examples/decoding/plot_decoding_csp_eeg.html).
# For another library useful for loading EEG data, take a look at
# [Neo IO](https://pythonhosted.org/neo/io.html).

import mne
from mne.io import concatenate_raws

subject_id = 22  # carefully cherry-picked to give nice results on such limited data :)
event_codes = [5, 6, 9, 10, 13, 14]  # codes for executed and imagined hands/feet

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(
    subject_id, event_codes, update_path=False)

# Load each of the files
raws = [mne.io.read_raw_edf(path, preload=False, stim_channel='auto',
                            verbose='WARNING')
        for path in physionet_paths]

# Concatenate them
raw = concatenate_raws(raws)
del raws  # to save memory

# Find the events in this dataset
events, _ = mne.events_from_annotations(raw)

# Use only EEG channels
eeg_channel_inds = \
    mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Extract trials, only using EEG channels
epoched = mne.Epochs(raw, events, dict(hands_or_left=2, feet_or_right=3),
                     tmin=1, tmax=4.1, proj=False, picks=eeg_channel_inds,
                     baseline=None, preload=True)

##############################################################################
# Convert data to Braindecode format
# ----------------------------------

# Braindecode has a minimalistic ```SignalAndTarget``` class, with
# attributes `X` for the signal and `y` for the labels. `X` should have
# these dimensions: trials x channels x timesteps. `y` should have one
# label per trial.

import numpy as np

# Convert data from volt to millivolt
# Pytorch expects float32 for input and int64 for labels.
X = (epoched.get_data() * 1e6).astype(np.float32)
y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1


# We use the first 40 trials for training and the next 30 trials for
# validation. The validation accuracies can be used to tune hyperparameters
# such as learning rate etc. The final 20 trials are split apart so we have
# a final hold-out evaluation set that is not part of any hyperparameter
# optimization. As mentioned before, this dataset is dangerously small to
# get any meaningful results and only used here for quick demonstration
# purposes.

from braindecode.datautil.signal_target import SignalAndTarget

train_set = SignalAndTarget(X[:40], y=y[:40])
valid_set = SignalAndTarget(X[40:70], y=y[40:70])


##############################################################################
# Create the model
# ----------------

# Braindecode comes with some predefined convolutional neural network
# architectures for raw time-domain EEG. Here, we use the shallow ConvNet
# model from [Deep learning with convolutional neural networks for EEG
# decoding and visualization](https://arxiv.org/abs/1703.05051).

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds  # XXX : move to braindecode.util

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = False
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 2
in_chans = train_set.X.shape[1]

# final_conv_length = auto ensures we only get a single output in the time dimension
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto')
if cuda:
    model.cuda()


# We use [AdamW](https://arxiv.org/abs/1711.05101) to optimize the parameters of
# our network together with [Cosine Annealing](https://arxiv.org/abs/1608.03983)
# of the learning rate. We supply some default parameters that we have found to
# work well for motor decoding, however we strongly encourage you to perform
# your own hyperparameter optimization using cross validation on your training
# data.

##############################################################################
# .. warning::
#
#   We will now use the Braindecode model class directly to perform the
#   training in a few lines of code. If you instead want to use your own
#   training loop, have a look at the
#   `Trialwise Low-Level Tutorial <./TrialWise_LowLevel.html>`_.

from torch.optim import AdamW
import torch.nn.functional as F
# optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1)

##############################################################################
# Run the training
# ----------------

n_epochs = 4
model.fit(train_set.X, train_set.y, n_epochs=n_epochs, batch_size=64,
          scheduler='cosine', validation_data=(valid_set.X, valid_set.y))


# The monitored values are also stored into a pandas dataframe:

model.epochs_df

# Eventually, we arrive at 83.4% accuracy, so 25 from 30 trials are correctly
# predicted. In the `Cropped Decoding Tutorial <./Cropped_Decoding.html>`_,
# we can learn how to achieve higher accuracies using cropped training.

##############################################################################
# Evaluation
# ----------

# Once we have all our hyperparameters and architectural choices done, we
# can evaluate the accuracies to report in our publication by evaluating on
# the test set:

test_set = SignalAndTarget(X[70:], y=y[70:])

model.evaluate(test_set.X, test_set.y)

# We can also retrieve predicted labels per trial as such:

y_pred = model.predict(test_set.X)

##############################################################################
# We can also retrieve the raw network outputs per trial as such:
#
# .. warning::
#
#    Note these are log-softmax outputs, so to get probabilities one would
#    have to exponentiate them using `th.exp`.

model.predict_outs(test_set.X)

##############################################################################
# .. warning::
#
#   If you want to try cross-subject decoding, changing the loading code to
#   the following will perform cross-subject decoding on imagined left vs
#   right hand closing, with 50 training and 5 validation subjects
#   (Warning, might be very slow if you are on CPU):

from braindecode.datautil import SignalAndTarget

# First 50 subjects as train
physionet_paths = [
    mne.datasets.eegbci.load_data(
        sub_id, [4, 8, 12], update_path=False)
    for sub_id in range(1, 51)]
physionet_paths = np.concatenate(physionet_paths)
raws = [mne.io.read_raw_edf(path, preload=False, stim_channel='auto')
        for path in physionet_paths]

raw = concatenate_raws(raws)

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                       eog=False, exclude='bads')

# Find the events in this dataset
events, _ = mne.events_from_annotations(raw)

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=1,
                     tmax=4.1, proj=False, picks=picks,
                     baseline=None, preload=True)

# 51-55 as validation subjects
physionet_paths_valid = [
    mne.datasets.eegbci.load_data(
        sub_id, [4, 8, 12], update_path=False)
    for sub_id in range(51, 56)]
physionet_paths_valid = np.concatenate(physionet_paths_valid)
raws_valid = [mne.io.read_raw_edf(path, preload=False, stim_channel='auto')
              for path in physionet_paths_valid]
raw_valid = concatenate_raws(raws_valid)
del raws_valid  # save memory

picks_valid = mne.pick_types(raw_valid.info, meg=False, eeg=True,
                             stim=False, eog=False,
                             exclude='bads')

events_valid, _ = mne.events_from_annotations(raw_valid)

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epoched_valid = mne.Epochs(raw_valid, events_valid, dict(hands=2, feet=3),
                           tmin=1, tmax=4.1, proj=False, picks=picks_valid,
                           baseline=None, preload=True)

train_X = (epoched.get_data() * 1e6).astype(np.float32)
train_y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1
valid_X = (epoched_valid.get_data() * 1e6).astype(np.float32)
valid_y = (epoched_valid.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1
train_set = SignalAndTarget(train_X, y=train_y)
valid_set = SignalAndTarget(valid_X, y=valid_y)


##############################################################################
# References
# ----------
#
#  This dataset was created and contributed to PhysioNet by the developers of the [BCI2000](http://www.schalklab.org/research/bci2000) instrumentation system, which they used in making these recordings. The system is described in:
#
#      Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043.
#
# [PhysioBank](https://physionet.org/physiobank/) is a large and growing archive of well-characterized digital recordings of physiologic signals and related data for use by the biomedical research community and further described in:
#
#     Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220.

"""
Amplitude Perturbation Visualization
====================================

In this tutorial, we show how to use perturbations of the input
amplitudes to learn something about the trained convolutional
networks. For more background, see
[Deep learning with convolutional neural networks for EEG decoding
and visualization](https://arxiv.org/abs/1703.05051), Section A.5.2.

First we will do some cross-subject decoding, again using the [Physiobank EEG Motor Movement/Imagery Dataset](https://www.physionet.org/physiobank/database/eegmmidb/), this time to decode imagined left hand vs. imagined right hand movement.

.. warning::

    This tutorial might be very slow if you are not using a GPU.
"""

##############################################################################
# Load data
# ---------

import mne
import numpy as np
from mne.io import concatenate_raws

from braindecode.datautil import SignalAndTarget

# First 50 subjects as train
physionet_paths = [
    mne.datasets.eegbci.load_data(sub_id, [4, 8, 12], update_path=False)
    for sub_id in range(1, 51)
]

physionet_paths = np.concatenate(physionet_paths)
raws = [
    mne.io.read_raw_edf(path, preload=False, stim_channel="auto")
    for path in physionet_paths
]

raw = concatenate_raws(raws)
del raws

picks = mne.pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# Find the events in this dataset
events, _ = mne.events_from_annotations(raw)

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = mne.Epochs(
    raw,
    events,
    dict(hands=2, feet=3),
    tmin=1,
    tmax=4.1,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
)

# 51-55 as validation subjects
physionet_paths_valid = [
    mne.datasets.eegbci.load_data(sub_id, [4, 8, 12], update_path=False)
    for sub_id in range(51, 56)
]
physionet_paths_valid = np.concatenate(physionet_paths_valid)
raws_valid = [
    mne.io.read_raw_edf(path, preload=False, stim_channel="auto")
    for path in physionet_paths_valid
]
raw_valid = concatenate_raws(raws_valid)

picks_valid = mne.pick_types(
    raw_valid.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# Find the events in this dataset
events_valid, _ = mne.events_from_annotations(raw_valid)

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs_valid = mne.Epochs(
    raw_valid,
    events_valid,
    dict(hands=2, feet=3),
    tmin=1,
    tmax=4.1,
    proj=False,
    picks=picks_valid,
    baseline=None,
    preload=True,
)

train_X = (epochs.get_data() * 1e6).astype(np.float32)
train_y = (epochs.events[:, 2] - 2).astype(np.int64)  # 2, 3 -> 0, 1
valid_X = (epochs_valid.get_data() * 1e6).astype(np.float32)
valid_y = (epochs_valid.events[:, 2] - 2).astype(np.int64)  # 2, 3 -> 0, 1
train_set = SignalAndTarget(train_X, y=train_y)
valid_set = SignalAndTarget(valid_X, y=valid_y)

##############################################################################
# Create the model
# ----------------
#
# We use the deep ConvNet from [Deep learning with convolutional neural
# networks for EEG decoding and visualization](https://arxiv.org/abs/1703.05051) (Section 2.4.2).

from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.util import set_random_seeds

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
cuda = False
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_window_samples = 450
# final_conv_length determines the size of the receptive field of the ConvNet
model = Deep4Net(
    in_chans=64,
    n_classes=2,
    input_window_samples=input_window_samples,
    filter_length_3=5,
    filter_length_4=5,
    pool_time_stride=2,
    stride_before_pool=True,
    final_conv_length=1,
)
if cuda:
    model.cuda()

from torch.optim import AdamW
import torch.nn.functional as F

optimizer = AdamW(
    model.parameters(), lr=0.01, weight_decay=0.5 * 0.001
)  # these are good values for the deep model
model.compile(
    loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True
)

##############################################################################
# Run the training
# ----------------

input_window_samples = 450
model.fit(
    train_set.X,
    train_set.y,
    n_epochs=30,
    batch_size=64,
    scheduler="cosine",
    input_window_samples=input_window_samples,
    validation_data=(valid_set.X, valid_set.y),
)


##############################################################################
# Compute correlation: amplitude perturbation - prediction change
# ---------------------------------------------------------------
#
# First collect all batches and concatenate them into one array of examples:

from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.torch_ext.util import np_to_var

test_input = np_to_var(np.ones((2, 64, input_window_samples, 1), dtype=np.float32))
if cuda:
    test_input = test_input.cuda()
out = model.network(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
iterator = CropsFromTrialsIterator(
    batch_size=32,
    input_window_samples=input_window_samples,
    n_preds_per_input=n_preds_per_input,
)

train_batches = list(iterator.get_batches(train_set, shuffle=False))
train_X_batches = np.concatenate(list(zip(*train_batches))[0])

# Next, create a prediction function that wraps the model prediction
# function and returns the predictions as numpy arrays. We use the predition
# before the softmax, so we create a new module with all the layers of the
# old until before the softmax.

from braindecode.util import var_to_np
import torch

new_model = nn.Sequential()
for name, module in model.network.named_children():
    if name == "softmax":
        break
    new_model.add_module(name, module)

new_model.eval()


def pred_fn(x):
    return var_to_np(
        torch.mean(
            new_model(np_to_var(x).cuda())[:, :, :, 0], dim=2, keepdim=False
        )
    )


from braindecode.visualization.perturbation import (
    compute_amplitude_prediction_correlations,
)

amp_pred_corrs = compute_amplitude_prediction_correlations(
    pred_fn, train_X_batches, n_iterations=12, batch_size=30
)

##############################################################################
# Plot correlations
# -----------------
#
# Pick out one frequency range and mean correlations within that frequency
# range to make a scalp plot. Here we use the alpha frequency range.

print(amp_pred_corrs.shape)

fs = epochs.info["sfreq"]
freqs = np.fft.rfftfreq(train_X_batches.shape[2], d=1.0 / fs)
start_freq = 7
stop_freq = 14

i_start = np.searchsorted(freqs, start_freq)
i_stop = np.searchsorted(freqs, stop_freq) + 1

freq_corr = np.mean(amp_pred_corrs[:, i_start:i_stop], axis=1)


# Now get approximate positions of the channels in the 10-20 system.

from braindecode.datasets.sensor_positions import (
    get_channelpos,
    CHANNEL_10_20_APPROX,
)

ch_names = [s.strip(".") for s in epochs.ch_names]
positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
positions = np.array(positions)

##############################################################################
# Plot with MNE
# -------------

import matplotlib.pyplot as plt
from matplotlib import cm

max_abs_val = np.max(np.abs(freq_corr))

fig, axes = plt.subplots(1, 2)
class_names = ["Left Hand", "Right Hand"]
for i_class in range(2):
    ax = axes[i_class]
    mne.viz.plot_topomap(
        freq_corr[:, i_class],
        positions,
        vmin=-max_abs_val,
        vmax=max_abs_val,
        contours=0,
        cmap=cm.coolwarm,
        axes=ax,
        show=False,
    )
    ax.set_title(class_names[i_class])

##############################################################################
# Plot with Braindecode
# ---------------------

from braindecode.visualization.plot import ax_scalp

fig, axes = plt.subplots(1, 2)
class_names = ["Left Hand", "Right Hand"]
for i_class in range(2):
    ax = axes[i_class]
    ax_scalp(
        freq_corr[:, i_class],
        ch_names,
        chan_pos_list=CHANNEL_10_20_APPROX,
        cmap=cm.coolwarm,
        vmin=-max_abs_val,
        vmax=max_abs_val,
        ax=ax,
    )
    ax.set_title(class_names[i_class])

# From these plots we can see the ConvNet clearly learned to use the
# lateralized response in the alpha band. Note that the positive correlations
# for the left hand on the left side do not imply an increase of alpha
# activity for the left hand in the data, see  [Deep learning with
# convolutional neural networks for EEG decoding and
# visualization](https://arxiv.org/abs/1703.05051) Result 12 for some
# notes on interpretability.

##############################################################################
# Dataset references
# ------------------

#  This dataset was created and contributed to PhysioNet by the developers of the [BCI2000](http://www.schalklab.org/research/bci2000) instrumentation system, which they used in making these recordings. The system is described in:
#
#      Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043.
#
# [PhysioBank](https://physionet.org/physiobank/) is a large and growing archive of well-characterized digital recordings of physiologic signals and related data for use by the biomedical research community and further described in:
#
#     Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220.

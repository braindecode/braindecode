"""
Sleep staging on the Sleep Physionet dataset using Eldele2021
=============================================================

This tutorial shows how to train and test a sleep staging neural network with
Braindecode. We use the attention-based model from [1]_ with the time distributed approach of [2]_
to learn on sequences of EEG windows using the openly accessible Sleep Physionet dataset [3]_ [4]_.

References
----------
.. [1] E. Eldele et al., "An Attention-Based Deep Learning Approach for Sleep Stage
        Classification With Single-Channel EEG," in IEEE Transactions on Neural Systems and
        Rehabilitation Engineering, vol. 29, pp. 809-818, 2021, doi: 10.1109/TNSRE.2021.3076234.

.. [2] Chambon, S., Galtier, M., Arnal, P., Wainrib, G. and Gramfort, A.
      (2018)A Deep Learning Architecture for Temporal Sleep Stage
      Classification Using Multivariate and Multimodal Time Series.
      IEEE Trans. on Neural Systems and Rehabilitation Engineering 26:
      (758-769)

.. [3] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of
       a sleep-dependent neuronal feedback loop: the slow-wave
       microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).

.. [4] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
       Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
       PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
       Research Resource for Complex Physiologic Signals.
       Circulation 101(23):e215-e220
"""
# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)


######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# Loading
# ~~~~~~~
#
# First, we load the data using the
# :class:`braindecode.datasets.sleep_physionet.SleepPhysionet` class. We load
# two recordings from two different individuals: we will use the first one to
# train our network and the second one to evaluate performance (as in the `MNE`_
# sleep staging example).
#
# .. _MNE: https://mne.tools/stable/auto_tutorials/sample-datasets/plot_sleep.html
#

from numbers import Integral
from braindecode.datasets import SleepPhysionet

subject_ids = [0, 1]
crop = (0, 30 * 400)  # we only keep 400 windows of 30s to speed example
dataset = SleepPhysionet(
    subject_ids=subject_ids, recording_ids=[2], crop_wake_mins=30,
    crop=crop)

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#
# Next, we preprocess the raw data. We convert the data to microvolts and apply
# a lowpass filter.

from braindecode.preprocessing import preprocess, Preprocessor, scale

high_cut_hz = 30

preprocessors = [
    Preprocessor(scale, factor=1e6, apply_on_array=True),
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz)
]

# Transform the data
preprocess(dataset, preprocessors)

######################################################################
# Extract windows
# ~~~~~~~~~~~~~~~
#
# We extract 30-s windows to be used in the classification task.
# The Eldele2021 model takes a single channel as input. Here, the Fpz-Cz channel is used as it
# was found to give better performance than using the Pz-Oz channel

from braindecode.preprocessing import create_windows_from_events

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    picks="Fpz-Cz",  # the other option is Pz-Oz,
    preload=True,
    mapping=mapping
)

######################################################################
# Window preprocessing
# ~~~~~~~~~~~~~~~~~~~~
#
# We also preprocess the windows by applying channel-wise z-score normalization
# in each window.

from sklearn.preprocessing import scale as standard_scale

preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])

######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We split the dataset into training and validation set taking
# every other subject as train or valid.

split_ids = dict(train=subject_ids[::2], valid=subject_ids[1::2])
splits = windows_dataset.split(split_ids)
train_set, valid_set = splits["train"], splits["valid"]

######################################################################
# Create sequence samplers
# ------------------------
#
# Following the time distributed approach of [2]_, we need to provide our
# neural network with sequences of windows, such that the embeddings of
# multiple consecutive windows can be concatenated and provided to a final
# classifier. We can achieve this by defining Sampler objects that return
# sequences of window indices.
# To simplify the example, we train the whole model end-to-end on sequences,
# rather than using the two-step approach of [2]_ (i.e. training the feature
# extractor on single windows, then freezing its weights and training the
# classifier).
#

from braindecode.samplers import SequenceSampler

n_windows = 3  # Sequences of 3 consecutive windows
n_windows_stride = 3  # Maximally overlapping sequences

train_sampler = SequenceSampler(train_set.get_metadata(), n_windows, n_windows_stride)
valid_sampler = SequenceSampler(valid_set.get_metadata(), n_windows, n_windows_stride)

# Print number of examples per class
print('Training examples: ', len(train_sampler))
print('Validation examples: ', len(valid_sampler))


######################################################################
# We also implement a transform to extract the label of the center window of a
# sequence to use it as target.

import numpy as np


# Use label of center window in the sequence
def get_center_label(x):
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x


train_set.target_transform = get_center_label
valid_set.target_transform = get_center_label

######################################################################
# Finally, since some sleep stages appear a lot more often than others (e.g.
# most of the night is spent in the N2 stage), the classes are imbalanced. To
# avoid overfitting on the more frequent classes, we compute weights that we
# will provide to the loss function when training.

from sklearn.utils import compute_class_weight

y_train = [train_set[idx][1] for idx in train_sampler]
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

######################################################################
# Create model
# ------------
#
# We can now create the deep learning model. In this tutorial, we use the sleep
# staging architecture introduced in [1]_, which is an attention-based
# neural network. We use the time distributed version of the model, where the
# feature vectors of a sequence of windows are concatenated and passed to a
# linear layer for classification.
#

import torch
from torch import nn
from braindecode.util import set_random_seeds
from braindecode.models import SleepStagerEldele2021, TimeDistributed

cuda = torch.cuda.is_available()  # check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to reproduce results
set_random_seeds(seed=31, cuda=cuda)

n_classes = 5
# Extract number of channels and time steps from dataset
n_channels, input_size_samples = train_set[0][0].shape

feat_extractor = SleepStagerEldele2021(
    sfreq,
    n_classes=n_classes,
    input_size_s=input_size_samples / sfreq,
    return_feats=True)

model = nn.Sequential(
    TimeDistributed(feat_extractor),  # apply model on each 30-s window
    nn.Sequential(  # apply linear layer on concatenated feature vectors
        nn.Flatten(start_dim=1),
        nn.Dropout(0.5),
        nn.Linear(feat_extractor.len_last_layer * n_windows, n_classes)
    )
)

# Send model to GPU
if cuda:
    model.cuda()


######################################################################
# Training
# --------
#
# We can now train our network. :class:`braindecode.EEGClassifier` is a
# braindecode object that is responsible for managing the training of neural
# networks. It inherits from :class:`skorch.NeuralNetClassifier`, so the
# training logic is the same as in
# `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#

from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier

lr = 1e-3
batch_size = 32
n_epochs = 3  # we use few epochs for speed and but more than one for plotting

train_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
    lower_is_better=False)
valid_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
    lower_is_better=False)
callbacks = [
    ('train_bal_acc', train_bal_acc),
    ('valid_bal_acc', valid_bal_acc)
]

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    criterion__weight=torch.Tensor(class_weights).to(device),
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=False,
    iterator_train__sampler=train_sampler,
    iterator_valid__sampler=valid_sampler,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device
)
# Model training for a specified number of epochs. `y` is None as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)


######################################################################
# Plot results
# ------------
#
# We use the history stored by Skorch during training to plot the performance of
# the model throughout training. Specifically, we plot the loss and the balanced
# balanced accuracy for the training and validation sets.

import matplotlib.pyplot as plt
import pandas as pd

# Extract loss and balanced accuracy values for plotting from history object
df = pd.DataFrame(clf.history.to_list())
df.index.name = "Epoch"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
df[['train_loss', 'valid_loss']].plot(color=['r', 'b'], ax=ax1)
df[['train_bal_acc', 'valid_bal_acc']].plot(color=['r', 'b'], ax=ax2)
ax1.set_ylabel('Loss')
ax2.set_ylabel('Balanced accuracy')
ax1.legend(['Train', 'Valid'])
ax2.legend(['Train', 'Valid'])
fig.tight_layout()
plt.show()

######################################################################
# Finally, we also display the confusion matrix and classification report:
#

from sklearn.metrics import confusion_matrix, classification_report

y_true = [valid_set[[i]][1][0] for i in range(len(valid_sampler))]
y_pred = clf.predict(valid_set)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))


######################################################################
# The model was able to learn despite the low amount of data that was available
# (only two recordings in this example) and reached a balanced accuracy of
# about 43% in a 5-class classification task (chance-level = 20%) on held-out
# data over 10 epochs.
#
# .. note::
#    To further improve performance, the number of epochs should be increased.
#    It has been reduced here for faster run-time in document generation. In
#    testing, 10 epochs provided reasonable performance with around 89% balanced
#    accuracy on training data and around 43% on held out validation data.
#    Increasing the number of training recordings and optimizing the hyperparameters
#    will also help increase performance

"""
Sleep staging on the Sleep Physionet dataset using U-Sleep network
==================================================================

.. note::
    Please take a look at the simpler sleep staging example
    :ref:`here <sphx_glr_auto_examples_plot_sleep_staging.py>`
    before going through this example. The current example uses a more complex
    architecture and a sequence-to-sequence (seq2seq) approach.

This tutorial shows how to train and test a sleep staging neural network with
Braindecode. We adapt the U-Sleep approach of [1]_ to learn on sequences of EEG
windows using the openly accessible Sleep Physionet dataset [2]_ [3]_.

.. warning::
    The example is written to have a very short execution time.
    This number of epochs is here too small and very few recordings are used.
    To obtain competitive results you need to use more data and more epochs.

"""
# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
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
# Next, we preprocess the raw data. We scale each channel recording-wise to
# have zero median and unit interquartile range. We don't upsample to 128 Hz as
# done in [1]_ so that we keep the example as light as possible. No filtering
# is described in [1]_.

from braindecode.preprocessing import preprocess, Preprocessor
from sklearn.preprocessing import robust_scale

preprocessors = [Preprocessor(robust_scale, channel_wise=True)]

# Transform the data
preprocess(dataset, preprocessors)

######################################################################
# Extract windows
# ~~~~~~~~~~~~~~~
#
# We extract 30-s windows to be used in the classification task.

from braindecode.preprocessing import create_windows_from_events

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
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
    preload=True,
    mapping=mapping,
)

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
# Following the sequence-to-sequence approach of [1]_, we need to provide our
# neural network with sequences of windows. We can achieve this by defining
# Sampler objects that return sequences of windows.
# Non-overlapping sequences of 35 windows are used in [1]_, however to limit
# the memory requirements for this example we use shorter sequences of 3
# windows.

from braindecode.samplers import SequenceSampler

n_windows = 3  # Sequences of 3 consecutive windows; originally 35 in paper
n_windows_stride = 3  # Non-overlapping sequences

train_sampler = SequenceSampler(
    train_set.get_metadata(), n_windows, n_windows_stride, randomize=True
)
valid_sampler = SequenceSampler(valid_set.get_metadata(), n_windows, n_windows_stride)

# Print number of examples per class
print(len(train_sampler))
print(len(valid_sampler))

######################################################################
# Finally, since some sleep stages appear a lot more often than others (e.g.
# most of the night is spent in the N2 stage), the classes are imbalanced. To
# avoid overfitting to the more frequent classes, we compute weights that we
# will provide to the loss function when training.

import numpy as np
from sklearn.utils import compute_class_weight

y_train = [train_set[idx][1][1] for idx in train_sampler]
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

######################################################################
# Create model
# ------------
#
# We can now create the deep learning model. In this tutorial, we use the
# U-Sleep architecture introduced in [1]_, which is fully convolutional
# neural network.

import torch
from braindecode.util import set_random_seeds
from braindecode.models import USleep

cuda = torch.cuda.is_available()  # check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
set_random_seeds(seed=31, cuda=cuda)

n_classes = 5
classes = list(range(n_classes))
# Extract number of channels and time steps from dataset
in_chans, input_size_samples = train_set[0][0].shape
model = USleep(
    n_chans=in_chans,
    sfreq=sfreq,
    depth=12,
    with_skip_connection=True,
    n_outputs=n_classes,
    n_times=input_size_samples
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
# .. note::
#    We use different hyperparameters from [1]_, as these hyperparameters were
#    optimized on different datasets and with a different number of recordings.
#    Generally speaking, it is recommended to perform hyperparameter
#    optimization if reusing this code on a different dataset or with more
#    recordings.

from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier

lr = 1e-3
batch_size = 32
n_epochs = 3  # we use few epochs for speed and but more than one for plotting

from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy_multi(model, X, y):
    y_pred = model.predict(X)
    return balanced_accuracy_score(y.flatten(), y_pred.flatten())


train_bal_acc = EpochScoring(
    scoring=balanced_accuracy_multi,
    on_train=True,
    name='train_bal_acc',
    lower_is_better=False,
)
valid_bal_acc = EpochScoring(
    scoring=balanced_accuracy_multi,
    on_train=False,
    name='valid_bal_acc',
    lower_is_better=False,
)
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
    device=device,
    classes=classes,
)
# Deactivate the default valid_acc callback:
clf.set_params(callbacks__valid_acc=None)

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
from braindecode.visualization import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

y_true = np.array([valid_set[i][1] for i in valid_sampler])
y_pred = clf.predict(valid_set)

confusion_mat = confusion_matrix(y_true.flatten(), y_pred.flatten())

plot_confusion_matrix(confusion_mat=confusion_mat,
                      class_names=['Wake', 'N1', 'N2', 'N3', 'REM'])

print(classification_report(y_true.flatten(), y_pred.flatten()))

######################################################################
# Finally, we can also visualize the hypnogram of the recording we used for
# validation, with the predicted sleep stages overlaid on top of the true
# sleep stages. We can see that the model cannot correctly identify the
# different sleep stages with this amount of training.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_true.flatten(), color='b', label='Expert annotations')
ax.plot(y_pred.flatten(), color='r', label='Predict annotations', alpha=0.5)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Sleep stage')

######################################################################
# Our model was able to learn, as shown by the decreasing training and
# validation loss values, despite the low amount of data that was available
# (only two recordings in this example). To further improve performance, more
# recordings should be included in the training set, the model should be
# trained for more epochs and hyperparameters should be optimized.

###########################################################################
# References
# ----------
#
# .. [1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.
#        U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021).
#        https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
#
# .. [2] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of
#        a sleep-dependent neuronal feedback loop: the slow-wave
#        microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).
#
# .. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
#        Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
#        PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
#        Research Resource for Complex Physiologic Signals.
#        Circulation 101(23):e215-e220

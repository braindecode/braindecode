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
    The example is written to have a very short excecution time.
    This number of epochs is here too small and very few recordings are used.
    To obtain competitive results you need to use more data and more epochs.

References
----------
.. [1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.
       U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021).
       https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py

.. [2] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of
       a sleep-dependent neuronal feedback loop: the slow-wave
       microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).

.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
       Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
       PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
       Research Resource for Complex Physiologic Signals.
       Circulation 101(23):e215-e220
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

from braindecode.datasets.sleep_physionet import SleepPhysionet

dataset = SleepPhysionet(
    subject_ids=[0, 1], recording_ids=[2], crop_wake_mins=30)


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
# We split the dataset into training and validation set using additional info
# stored in the `description` attribute of
# :class:`braindecode.datasets.BaseDataset`, in this case using the ``subject``
# column.

import numpy as np
from sklearn.model_selection import train_test_split
from braindecode.datasets import BaseConcatDataset

random_state = 31
subjects = np.unique(windows_dataset.description['subject'])
subj_train, subj_valid = train_test_split(
    subjects, test_size=0.5, random_state=random_state
)

split_ids = {'train': subj_train, 'valid': subj_valid}
splitted = dict()
for name, values in split_ids.items():
    splitted[name] = BaseConcatDataset(
        [ds for ds in windows_dataset.datasets
         if ds.description['subject'] in values]
    )

train_set = splitted['train']
valid_set = splitted['valid']


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

train_sampler = SequenceSampler(train_set.get_metadata(), n_windows, n_windows_stride)
valid_sampler = SequenceSampler(valid_set.get_metadata(), n_windows, n_windows_stride)

# Print number of examples per class
print(len(train_sampler))
print(len(valid_sampler))


######################################################################
# Finally, since some sleep stages appear a lot more often than others (e.g.
# most of the night is spent in the N2 stage), the classes are imbalanced. To
# avoid overfitting to the more frequent classes, we compute weights that we
# will provide to the loss function when training.

from sklearn.utils.class_weight import compute_class_weight

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
set_random_seeds(seed=random_state, cuda=cuda)

n_classes = 5
# Extract number of channels and time steps from dataset
in_chans, input_size_samples = train_set[0][0].shape

model = USleep(
    in_chans=in_chans,
    sfreq=sfreq,
    depth=12,
    with_skip_connection=True,
    n_classes=n_classes,
    input_size_s=input_size_samples / sfreq,
    apply_softmax=False
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
n_epochs = 1  # this number is kept too small to reduce running time in the doc

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
callbacks = [('train_bal_acc', train_bal_acc),
             ('valid_bal_acc', valid_bal_acc)]

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
# misclassification rate (1 - balanced accuracy) for the training and validation
# sets.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Extract loss and balanced accuracy values for plotting from history object
df = pd.DataFrame(clf.history.to_list())
df[['train_mis_clf', 'valid_mis_clf']] = (
    100 - df[['train_bal_acc', 'valid_bal_acc']] * 100
)

# get percent of misclass for better visual comparison to loss
plt.style.use('seaborn-talk')
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14
)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel('Loss', color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_mis_clf', 'valid_mis_clf']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False
)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel('Balanced misclassification rate [%]', color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel('Epoch', fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(
    Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train')
)
handles.append(
    Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid')
)
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()


######################################################################
# Finally, we also display the confusion matrix and classification report:

from sklearn.metrics import confusion_matrix, classification_report

y_true = np.array([valid_set[i][1] for i in valid_sampler])
y_pred = clf.predict(valid_set)

print(confusion_matrix(y_true.flatten(), y_pred.flatten()))
print(classification_report(y_true.flatten(), y_pred.flatten()))


######################################################################
# Our model was able to learn, as shown by the decreasing training and
# validation loss values, despite the low amount of data that was available
# (only two recordings in this example). To further improve performance, more
# recordings should be included in the training set, the model should be
# trained for more epochs and hyperparameters should be optimized.

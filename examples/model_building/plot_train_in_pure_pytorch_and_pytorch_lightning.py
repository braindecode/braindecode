"""
Training a Braindecode model in PyTorch
======================================

This tutorial shows you how to train a Braindecode model with PyTorch. The data
preparation and model instantiation steps are identical to that of the tutorial
`How to train, test and tune your model <./plot_how_train_test_and_tune.html>`__

We will use the BCIC IV 2a dataset as a showcase example.

The methods shown can be applied to any standard supervised trial-based decoding setting.
This tutorial will include additional parts of code like loading and preprocessing,
defining a model, and other details which are not exclusive to this page (compare
`Cropped Decoding Tutorial <./plot_bcic_iv_2a_moabb_trial.html>`__). Therefore we
will not further elaborate on these parts and you can feel free to skip them.

In general, we distinguish between "usual" training and evaluation and hyperparameter search.
The tutorial is therefore split into two parts, one for the three different training schemes
and one for the two different hyperparameter tuning methods.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Why should I care about model evaluation?
# -----------------------------------------
# Short answer: To produce reliable results!
#
# In machine learning, we usually follow the scheme of splitting the
# data into two parts, training and testing sets. It sounds like a
# simple division, right? But the story does not end here.
#
# While developing a ML model you usually have to adjust and tune
# hyperparameters of your model or pipeline (e.g., number of layers,
# learning rate, number of epochs). Deep learning models usually have
# many free parameters; they could be considered complex models with
# many degrees of freedom. If you kept using the test dataset to
# evaluate your adjustmentyou would run into data leakage.
#
# This means that if you use the test set to adjust the hyperparameters
# of your model, the model implicitly learns or memorizes the test set.
# Therefore, the trained model is no longer independent of the test set
# (even though it was never used for training explicitly!).
# If you perform any hyperparameter tuning, you need a third split,
# the so-called validation set.
#
# This tutorial shows the three basic schemes for training and evaluating
# the model as well as two methods to tune your hyperparameters.
#

######################################################################
# .. warning::
#    You might recognize that the accuracy gets better throughout
#    the experiments of this tutorial. The reason behind that is that
#    we always use the same model with the same paramters in every
#    segment to keep the tutorial short and readable. If you do your
#    own experiments you always have to reinitialize the model before
#    training.
#

######################################################################
# Loading, preprocessing, defining a model, etc.
# ----------------------------------------------
#


######################################################################
# Loading
# ~~~~~~~
#

from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#

import numpy as np

from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(
        lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
        factor=1e6,
    ),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)

######################################################################
# Cut Compute Windows
# ~~~~~~~~~~~~~~~~~~~
#

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

######################################################################
# Create model
# ~~~~~~~~~~~~
#

import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model.cuda()

######################################################################
# How to train and evaluate your model
# ------------------------------------
#

######################################################################
# Split dataset into train and test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case the ``session`` column. We
# select ``session_T`` for training and ``session_E`` for testing.
# For other datasets, you might have to choose another column.
#
# .. note::
#    No matter which of the three schemes you use, this initial
#    two-fold split into train_set and test_set always remains the same.
#    Remember that you are not allowed to use the test_set during any
#    stage of training or tuning.
#

splitted = windows_dataset.split("session")
train_set = splitted["session_T"]
test_set = splitted["session_E"]

######################################################################
# Option 1: Pure pytorch training loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# `model` is an instance of `torch.nn.Module`, and can as such be trained
# using PyTorch optimization capabilities.
# The following training scheme is simple as the dataset is only
# split into two distinct sets (``train_set`` and ``test_set``).
# This scheme uses no separate validation split and should only be
# used for the final evaluation of the (previously!) found
# hyperparameters configuration.
#
# .. warning::
#    If you make any use of the ``test_set`` during training
#    (e.g. by using EarlyStopping) there will be data leakage
#    which will make the reported generalization capability/decoding
#    performance of your model less credible.
#
# .. warning::
#    The parameter values showcased here for optimizing the network are
#    chosen to make this tutorial fast to run and build. Real-world values
#    would be higher, especially when it comes to batch n_epochs.

from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 2


######################################################################
# The following method runs one training epoch over the dataloader for the
# given model. It needs a loss function, optimization algorithm, and
# learning rate updating callback.

def train(
    dataloader: DataLoader, model: Module, loss_fn, optimizer, scheduler: LRScheduler
):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.train()  # declare the model as trainable
    train_loss, correct = 0, 0
    for X, y, _ in dataloader:
        X, y = X.to(device), y.to(device)  # map tensors to their device
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(
        f"Train Accuracy: {(100*correct):>0.5f}, Train Loss: {train_loss / n_batches:>8f}, "
        f"lr: {scheduler.get_last_lr()[0]:<0.4f}",
        end=", ",
    )

    # Update the learning rate
    scheduler.step()

######################################################################
# Very similarly, the evaluation function loops over the entire dataloader
# and accumulate the metrics, but doesn't update the model weights.


def test(dataloader: DataLoader, model: Module, loss_fn):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()  # switch to eval mode
    test_loss, correct = 0, 0
    with torch.no_grad():  # skip gradient computation as it's not necessary for evaluation
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= n_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Test loss: {test_loss:>8f} \n")


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
loss_fn = torch.nn.NLLLoss()

# train_set and test_set are instances of torch Datasets, and can seamlessly be
# wrapped in data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

for t in range(n_epochs):
    print(f"Epoch {t+1}, ", end="")
    train(train_loader, model, loss_fn, optimizer, scheduler)
    test(test_loader, model, loss_fn)


######################################################################
# Option 2: Train it with pytorch lightning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# Alternatively, lightning provides a nice interface around torch modules
# which integrates the previous logic.


import lightning as L
from torchmetrics.functional import accuracy


class LitModule(L.LightningModule):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.loss = torch.nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y, "multiclass", num_classes=4)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
        return [optimizer], [scheduler]

trainer = L.Trainer(max_epochs=n_epochs)
trainer.fit(LitModule(model), DataLoader(train_set, batch_size=batch_size, shuffle=True))
trainer.test(dataloaders=test_loader)

""".. _basic-training-epochs:

Simple training on MNE epochs
=============================

The braindecode library gives you access to a large number of neural network
architectures that were developed for EEG data decoding. This tutorial will
show you how you can easily use any of these models to decode your own data.
In particular, we assume that have your data in an MNE format and want to
train one of the Braindecode models on it.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

######################################################################
# Finding the model you want
# --------------------------
#
# Exploring the braindecode online documentation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's suppose you recently stumbled upon the Schirrmeister 2017 article [1]_.
# In this article, the authors mention that their novel architecture ShallowConvNet
# is performing well on the BCI Competition IV 2a dataset and you would like to use
# it on your own data. Fortunately, the authors also mentioned they published their
# architecture on Braindecode!
#
# In order to use this architecture, you first need to find what is its exact
# name in Braindecode. To do so, you can visit Braindecode's Models Page
# page for information on which are the available models.
#
# Alternatively, the :doc:`API </api>` also provide a dictionary with all available models:

from braindecode.models.util import models_dict

print(f"All the Braindecode models:\n{list(models_dict.keys())}")

######################################################################
# After your investigation, you found out that the model you are looking for is
# ``ShallowFBCSPNet``. You can now import it from Braindecode:

import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

######################################################################
# Examining the model
# ~~~~~~~~~~~~~~~~~~~
#
# Now that you found your model, you must check which parameters it expects.
# You can find this information either in the online documentation here:
# :class:`braindecode.models.ShallowFBCSPNet` or directly in the module's docstring:

print(ShallowFBCSPNet.__doc__)

# Set random seed to ensure reproducible initialization below
seed = 20240205
cuda = torch.cuda.is_available()
set_random_seeds(seed=seed, cuda=cuda)

######################################################################
# Additionally, you might be interested in visualizing the model's architecture.
# This can be done by initializing the model and calling its ``__str__()`` method.
# To initialize it, we need to specify some parameters that we set at random
# values for now:

model = ShallowFBCSPNet(
    n_chans=32,
    n_times=1000,
    n_outputs=2,
    final_conv_length="auto",
)
print(model)

######################################################################
# Loading your own data with MNE
# ------------------------------
#
# In this tutorial, we demonstrate how to train the model on MNE data.
# MNE is quite a popular library for EEG data analysis as it provides methods
# to load data from many different file formats and a large collection of algorithms
# to preprocess it.
# However, Braindecode is not limited to MNE and can be used with numpy arrays or
# PyTorch tensors/datasets.
#
# For this example, we generate some random data containing 100 examples with each
# 3 channels and 1024 time points. We also generate some random labels for our data
# that simulate a 4-class classification problem.

import mne
import numpy as np

info = mne.create_info(ch_names=["C3", "C4", "Cz"], sfreq=256.0, ch_types="eeg")
X = np.random.randn(100, 3, 1024)  # 100 epochs, 3 channels, 4 seconds (@256Hz)
epochs = mne.EpochsArray(X, info=info)
y = np.random.randint(0, 4, size=100)  # 4 classes
print(epochs)

######################################################################
# Training your model (scikit-learn compatible)
# ---------------------------------------------
#
# Now that you know which model you want to use, you know how to instantiate it,
# and that we have some fake data, it is time to train the model!
#
# .. note::
#    `<skorch_>`_  is a library that allows you to wrap
#    any PyTorch module into a scikit-learn-compatible classifier or regressor.
#    Braindecode provides wrappers that inherit form the original Skorch ones and simply
#    implement a few additional features that facilitate the use of Braindecode models.
#
# To train a Braindecode model, the easiest way is by using braindecode's
# Skorch wrappers. These wrappers are :class:`braindecode.classifier.EEGClassifier` and
# :class:`braindecode.regressor.EEGRegressor`. As our fake data is a classification task,
# we will use the former.
#
# The wrapper :class:`braindecode.classifier.EEGClassifier` expects a model class as its first argument but
# to facilitate the usage, you can also simply pass the name of any braindecode model as a string.
# The wrapper automatically finds and instantiates the model for you.
# If you want to pass parameters to your model, you can give them to the wrapper
# with the prefix ``module__``.
#
from skorch.dataset import ValidSplit

from braindecode import EEGClassifier

net = EEGClassifier(
    "ShallowFBCSPNet",
    module__final_conv_length="auto",
    train_split=ValidSplit(0.2),
    # To train a neural network you need validation split, here, we use 20%.
    batch_size=16,
    # The 80 training examples are fewer than the default batch size (128); as
    # incomplete batches are dropped, a smaller batch size is needed to train.
)

######################################################################
# In this example, we passed one additional parameter to the wrapper: ``module__final_conv_length``
# that will be forwarded to the model (without the prefix ``module__``).
#
# We also note that the parameters ``n_chans``, ``n_times`` and ``n_outputs`` were not specified
# even if :class:`braindecode.models.ShallowFBCSPNet` needs them to be initialized. This is because the
# wrapper will automatically infer them, along with some other signal-related parameters,
# from the input data at training time.
#
# Under the hood, the wrapper uses the :func:`braindecode.datautil.infer_signal_properties` function
# to extract these parameters:
from braindecode.datautil import infer_signal_properties

sig_props = infer_signal_properties(epochs, y, mode="classification")
print(f"Inferred signal properties:\n{sig_props}")

#################################################################
# Now that we have our model wrapped in a scikit-learn-compatible classifier,
# we can train it by simply calling the ``fit`` method:

net.fit(epochs, y)

######################################################################
# ``net.history`` keeps the loss of each epoch on the training data and on
# the 20 % validation split, and the validation accuracy. Our labels are
# random, so the network has nothing to learn: the validation accuracy
# stays around the 25 % chance level of four balanced classes.

import matplotlib.pyplot as plt

BLUE, GREY, INK = "#0072B2", "#7F7F7F", "#14181F"

plt.rcdefaults()  # importing MOABB applies seaborn's "whitegrid" style globally
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

history_epochs = net.history[:, "epoch"]
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
for split, color in [("Train", GREY), ("Valid", BLUE)]:
    loss = net.history[:, f"{split.lower()}_loss"]
    ax_loss.plot(history_epochs, loss, color=color, lw=1.8)
    ax_loss.annotate(
        split,
        xy=(history_epochs[-1], loss[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        va="center",
        fontweight="bold",
    )
ax_acc.plot(history_epochs, 100 * np.array(net.history[:, "valid_acc"]), color=BLUE)
ax_acc.axhline(25, color=GREY, ls="--", lw=0.9)
ax_acc.text(
    history_epochs[-1],
    23,
    "Chance (25%)",
    color=GREY,
    fontsize=8.5,
    ha="right",
    va="top",
)
ax_acc.set_ylim(0, 100)
ax_acc.set_yticks([0, 25, 50, 75, 100])
for ax, letter, title, ylabel in [
    (ax_loss, "A", "Cross-entropy loss", "Loss"),
    (ax_acc, "B", "Validation accuracy", "Accuracy (%)"),
]:
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))

######################################################################
# The pre-trained model is accessible via the ``module_`` attribute:

print(net.module_)

######################################################################
# And we can see that all the following parameters were indeed automatically inferred
# from the training data:

print(
    f"{net.module_.n_chans=}\n{net.module_.n_times=}\n{net.module_.n_outputs=}"
    f"\n{net.module_.input_window_seconds=}\n{net.module_.sfreq=}\n{net.module_.chs_info=}"
)

######################################################################
# Depending on the type of data used for training, some parameters might not be
# possible to infer. For example if you pass a numpy array or a
# :class:`braindecode.datasets.WindowsDataset` with ``targets_from="metadata"``,
# then only ``n_chans``, ``n_times`` and ``n_outputs`` will be inferred.
# And if you pass other types of datasets, only ``n_chans`` and ``n_times`` will be inferred.
# In these case, you will have to pass the missing parameters manually
# (with the prefix ``module__``).

######################################################################
# References
# ----------
#
# .. [1] Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter,
#        M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T.(2017).
#        Deep learning with convolutional neural networks for EEG decoding and visualization.
#        Human Brain Mapping, Aug. 2017.
#        Online: http://dx.doi.org/10.1002/hbm.23730
#
# .. include:: /links.inc

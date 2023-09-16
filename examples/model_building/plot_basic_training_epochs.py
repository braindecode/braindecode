"""
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's suppose you recently stumbled upon the Schirrmeister 2017 article [1]_.
# In this article, the authors mention that their novel architecture ShallowConvNet
# is performing well on the BCI Competition IV 2a dataset and you would like to use
# it on your own data. Fortunately, the authors also mentioned they published their
# architecture on Braindecode!
#
# In order to use this architecture, you first need to find what is its exact
# name in Braindecode. To do so, you can visit the Braindecode online documentation
# which lists all the available models.
#
# Models list: https://braindecode.org/stable/api.html#models
#
# Alternatively, the API also provide a dictionary with all available models:

from braindecode.models.util import models_dict

print(f'All the Braindecode models:\n{list(models_dict.keys())}')

######################################################################
# After your investigation, you found out that the model you are looking for is
# ``ShallowFBCSPNet``. You can now import it from Braindecode:

from braindecode.models import ShallowFBCSPNet

######################################################################
# Examining the model
# ~~~~~~~~~~~~~~~~~~~
#
# Now that you found your model, you must check which parameters it expects.
# You can find this information either in the online documentation here:
# :class:`braindecode.models.ShallowFBCSPNet` or directly in the module's docstring:

print(ShallowFBCSPNet.__doc__)

######################################################################
# Additionally, you might be interested in visualizing the model's architecture.
# This can be done by initializing the model and calling its ``__str__()`` method.
# To initialize it, we need to specify some parameters that we set at random
# values for now:

model = ShallowFBCSPNet(
    n_chans=32,
    n_times=1000,
    n_outputs=2,
    final_conv_length='auto',
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

info = mne.create_info(ch_names=['C3', 'C4', 'Cz'], sfreq=256., ch_types='eeg')
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
#    `Skorch <https://skorch.readthedocs.io>`_  is a library that allows you to wrapp
#    any PyTorch module into a scikit-learn-compatible classifier or regressor.
#    Braindecode provides wrappers that inherit form the original Skorch ones and simply
#    implement a few additional features that facilitate the use of Braindecode models.
#
# To train a Braindecode model, the easiest way is by using braindecode's
# Skorch wrappers. These wrappers are :class:`braindecode.EEGClassifier` and
# :class:`braindecode.EEGRegressor`. As our fake data is a classification task,
# we will use the former.
#
# The wrapper :class:`braindecode.EEGClassifier` expects a model class as its first argument but
# to facilitate the usage, you can also simply pass the name of any braindecode model as a string.
# The wrapper automatically finds and instantiates the model for you.
# If you want to pass parameters to your model, you can give them to the wrapper
# with the prefix ``module__``.
#
from skorch.dataset import ValidSplit
from braindecode import EEGClassifier

net = EEGClassifier(
    'ShallowFBCSPNet',
    module__final_conv_length='auto',
    train_split=ValidSplit(0.2),
    # To train a neural network you need validation split, here, we use 20%.
)

######################################################################
# In this example, we passed one additional parameter to the wrapper: ``module__final_conv_length``
# that will be forwarded to the model (without the prefix ``module__``).
#
# We also note that the parameters ``n_chans``, ``n_times`` and ``n_outputs`` were not specified
# even if :class:`braindecode.ShallowFBCSPNet` needs them to be initialized. This is because the
# wrapper will automatically infer them, along with some other signal-related parameters,
# from the input data at training time.
#
# Now that we have our model wrapped in a scikit-learn-compatible classifier,
# we can train it by simply calling the ``fit`` method:

net.fit(epochs, y)

######################################################################
# The pre-trained model is accessible via the ``module_`` attribute:

print(net.module_)

######################################################################
# And we can see that all the following parameters were automatically inferred
# from the training data:

print(f'{net.module_.n_chans=}\n{net.module_.n_times=}\n{net.module_.n_outputs=}'
      f'\n{net.module_.input_window_seconds=}\n{net.module_.sfreq=}\n{net.module_.chs_info=}')

######################################################################
# Depending on the type of data used for training, some parameters might not be
# possible to infer. For example if you pass a numpy array or a
# :class:`braindecode.dataset.WindowsDataset` with ``target_from="metadata"``,
#  then only ``n_chans``, ``n_times`` and ``n_outputs`` will be inferred.
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

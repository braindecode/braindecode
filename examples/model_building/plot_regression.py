"""
Convolutional neural network regression model on fake data
==========================================================

This example shows how to train a CNN regression model on fake dataset.

"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
# 
# License: BSD-3

###################################################################################################
# Fake regression data
# -------------------------------
# Function for generation of the fake regression dataset. It generates `n_fake_recs` recordings, 
# each containing Gaussian noise ~N(0, 1). Each fake recording signal has `n_fake_chs` channels, 
# it is lasts `fake_duration` [s] and it is sampled with `fake_sfreq` [Hz]. One of the 
# `n_fake_recs` recordings is used for evaluation and the others for training. 

import numpy as np
import pandas as pd
from braindecode.util import create_mne_dummy_raw
from braindecode.datasets import BaseDataset, BaseConcatDataset

# Function for generating fake regression data
def fake_regression_dataset(n_fake_recs, n_fake_chs, fake_sfreq, fake_duration, n_fake_targets):
    """ Generates fake regression dataset.

        Arguments:
            n_fake_recs    (int)  : number of the fake recordings
            n_fake_chs     (int)  : number of the fake EEG channels
            fake_sfreq     (float): fake sampling frequency in Hz
            fake_duration  (float): fake recording duration in s
            n_fake_targets (int)  : number of targets

        Returns:
            dataset: BaseConcatDataset object
    """
    datasets = []
    for i in range(n_fake_recs):
        train_or_eval = "eval" if i == 0 else "train"
        raw, _ = create_mne_dummy_raw(n_channels=n_fake_chs, 
                                      n_times=fake_duration * fake_sfreq, sfreq=fake_sfreq)
        target = np.random.randint(0, 100, n_fake_targets)
        if n_fake_targets == 1:
            target = target[0]
        fake_description = pd.Series(data=[target, train_or_eval], index=["target", "session"])
        datasets.append(BaseDataset(raw, fake_description, target_name="target"))

    return BaseConcatDataset(datasets)

# -------------------------------------------------------------------------------------------------
# Generating fake regression dataset
n_fake_rec = 5
n_fake_chans = 21
fake_sfreq = 100
fake_duration=60
n_fake_targets = 1
dataset = fake_regression_dataset(n_fake_recs=n_fake_rec, n_fake_chs=n_fake_chans,
                                  fake_sfreq=fake_sfreq, fake_duration=fake_duration,
                                  n_fake_targets=n_fake_targets)

###################################################################################################
# Defining a CNN model
# ----------------------
# Choosing and defining a CNN model, ShallowFBCSPNet or Deep4Net, introduced in [1]_. 
# and choosing the processor 
# (GPU/CPU) where the model training and evaluation will take place 
# (by default is GPU if it exists).
import torch
from braindecode.util import set_random_seeds
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet

# Choosing a CNN model
model_name = "deep"  # 'shallow' or 'deep'

# Choosing between GPU and CPU processors
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

# Setting a random seed
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

# Defining a CNN model
if model_name in ["shallow", "Shallow", "ShallowConvNet"]:
    model = ShallowFBCSPNet(in_chans=n_fake_chans, n_classes=n_fake_targets,
                            input_window_samples=fake_sfreq * fake_duration,
                            n_filters_time=40, n_filters_spat=40,
                            final_conv_length=35,)
elif model_name in ["deep", "Deep", "DeepConvNet"]:
    model = Deep4Net(in_chans=n_fake_chans, n_classes=n_fake_targets,
                     input_window_samples=fake_sfreq * fake_duration,
                     n_filters_time=25, n_filters_spat=25,
                     stride_before_pool=True,
                     n_filters_2=n_fake_chans * 2,
                     n_filters_3=n_fake_chans * 4,
                     n_filters_4=n_fake_chans * 8,
                     final_conv_length=1,)
else:
    raise ValueError(f'{model_name} unknown')


# Conversion of the CNN classifier into regressor 
# by removing `softmax` function from the last layer. 
new_model = torch.nn.Sequential()
for name, module_ in model.named_children():
    if "softmax" in name:
        continue
    new_model.add_module(name, module_)
model = new_model

if cuda:
    model.cuda()

from braindecode.models.util import to_dense_prediction_model, get_output_shape
to_dense_prediction_model(model)
###################################################################################################
# Data windowing
# ----------------
# Windowing data with a sliding window and splitting into train and valid subsets. 
n_preds_per_input = get_output_shape(model, n_fake_chans, fake_sfreq * fake_duration)[2]
from braindecode.preprocessing import create_fixed_length_windows
windows_dataset = create_fixed_length_windows(dataset,
                                              start_offset_samples=0, stop_offset_samples=0,
                                              window_size_samples=fake_sfreq * fake_duration,
                                              window_stride_samples=n_preds_per_input,
                                              drop_last_window=False, drop_bad_windows=True,)
splits = windows_dataset.split("session")
train_set = splits["train"]
valid_set = splits["eval"]

###################################################################################################
# Model training
# -----------------
# 
from braindecode import EEGRegressor
from braindecode.training.losses import CroppedLoss
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
batch_size = 64
n_epochs = 3
optimizer_lr = 0.000625
optimizer_weight_decay = 0
regressor = EEGRegressor(model, cropped=True, 
                         criterion=CroppedLoss, criterion__loss_function=torch.nn.functional.mse_loss,
                         optimizer=torch.optim.AdamW,
                         optimizer__lr=optimizer_lr, optimizer__weight_decay=optimizer_weight_decay,
                         train_split=predefined_split(valid_set),
                         iterator_train__shuffle=True,
                         batch_size=batch_size,
                         callbacks=["neg_root_mean_squared_error",
                         # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
                         ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),],
                         device=device,)
regressor.fit(train_set, y=None, epochs=n_epochs)
###################################################################################################
# References
# ----------
#
# .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., 
#        Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). 
#        Deep learning with convolutional neural networks for EEG decoding and visualization. 
#        Human brain mapping, 38(11), 5391-5420.
#
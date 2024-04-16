"""
How to process and do transfer learning with braindecode
====================================================================================

This tutorial shows how-to perform transfer learning using braindecode.
Indeed, it is known that the best augmentation to use often dependent on the task
or phenomenon studied. Here we follow the methodology proposed in [1]_ on the
openly available BCI IV 2a Dataset and another dataset.


.. topic:: Transfer Learning

    REPLACE... Data augmentation could be a step in training deep learning models.
    For decoding brain signals, recent studies have shown that artificially
    generating samples may increase the final performance of a deep learning model [1]_.
    Other studies have shown that data augmentation can be used to cast
    a self-supervised paradigm, presenting a more diverse
    view of the data, both with pretext tasks and contrastive learning [2]_.


REPLACE...

Figure about transfer learning.


.. contents:: This example covers:
   :local:
   :depth: 2

"""
# Authors: MJ Bayazi <mj.darvishi92@gmail.com>
#           Bruno Aristimunha <a.bruno@ufabc.edu.br>
# License: BSD (3-clause)

######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# Loading
# ~~~~~~~
#
# This example shows how to train a neural network with supervision on TUH [2]
# EEG data and transfer the model to NMT [3] EEG dataset. We follow the approach of [1]

random_state = 2024
n_jobs = 1

# ## Loading and preprocessing the dataset
#
# ### Load and save the raw recordings
#
# Here we assume you already load and preprocess the raw recordings for both TUAB and NMT datasetsand saved the file in `TUAB_path' and 'NMT_path' respectively. To read more see this notebook [here](https://braindecode.org/stable/auto_examples/applied_examples/plot_tuh_eeg_corpus.html)
#
# ### Load the preprocessed data
#
# Now, we load a few recordings from the TUAB dataset. Running
# this example with more recordings should yield better representations and
# downstream classification performance.


TUAB_path = "PATH_TO_TUAB_DATASET"
NMT_path = "PATH_TO_NMT_DATASET"

from braindecode.datautil.serialization import load_concat_dataset

TUAB_ds = load_concat_dataset(
    TUAB_path,
    preload=False,
    target_name=["pathological", "age", "gender"],  # )
    ids_to_load=range(20),
)

######################################################################
# Target is being set to pathological
# -------------------------------------
import pandas as pd

target = TUAB_ds.description["pathological"].astype(int)
for d, y in zip(TUAB_ds.datasets, target):
    d.description["pathological"] = y
    d.target_name = "pathological"
    d.target = d.description[d.target_name]
TUAB_ds.set_description(
    pd.DataFrame([d.description for d in TUAB_ds.datasets]), overwrite=True
)

# Splitting dataset into train, valid and test sets
#
# We split the recordings by subject into train, validation and
# testing sets.
#

# split based on train split from dataset
train_set = TUAB_ds.split("train")["True"]
test_set = TUAB_ds.split("train")["False"]


######################################################################
# Create the models
# -------------------------------------
# We can now create the deep learning model.
# In this tutorial, we use DeepNet introduced in [4].


from braindecode.models import Deep4Net

n_chans = 21
n_classes = 2
input_window_samples = 6000
drop_prob = 0.5
cuda = True  # Set to False if you don't have a GPU
n_start_chans = 25
final_conv_length = 1
n_chan_factor = 2
stride_before_pool = True
# input_window_samples =6000
model = Deep4Net(
    n_chans,
    n_classes,
    n_filters_time=n_start_chans,
    n_filters_spat=n_start_chans,
    input_window_samples=input_window_samples,
    n_filters_2=int(n_start_chans * n_chan_factor),
    n_filters_3=int(n_start_chans * (n_chan_factor**2.0)),
    n_filters_4=int(n_start_chans * (n_chan_factor**3.0)),
    final_conv_length=final_conv_length,
    stride_before_pool=stride_before_pool,
    drop_prob=drop_prob,
)
# Send model to GPU
if cuda:
    model.cuda()

from braindecode.models.util import to_dense_prediction_model, get_output_shape

to_dense_prediction_model(model)

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

######################################################################
# Extracting windows
# -------------------------------------
#
# We extract 60-s windows to be used in both datasets. We use a window size of 6000 samples.
#

from braindecode.datautil.windowers import create_fixed_length_windows

window_train_set = create_fixed_length_windows(
    train_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    preload=True,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=True,
)

window_test_set = create_fixed_length_windows(
    test_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    preload=False,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
)

######################################################################
# Defining the classifier windows
# -------------------------------------


from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

weight_decay = 0.5 * 0.001
lr = 0.0625 * 0.01
n_epochs = 10
batch_size = 64

clf = EEGClassifier(
    model,
    cropped=True,
    classes=[0, 1],
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(window_test_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        "balanced_accuracy",
        "f1",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device="cuda" if cuda else "cpu",
)

clf.initialize()  # This is important!
print("classifier initialized")
print(
    "Number of parameters = ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)


######################################################################
# Training
# -------------------------------------

# We can now train our network on the TUAB. We use similar
# hyperparameters as in [1]_, but reduce the number of epochs and
# increase the learning rate to account for the smaller setting of
# this example.
#
#


clf.fit(window_train_set, y=None, epochs=n_epochs)


######################################################################
# Saving the model
# -------------------------------------


result_path = "PATH_TO_SAVE_THE_MODEL"
path = result_path + "model_{}.pt".format(random_state)
torch.save(clf.module, path)
path = result_path + "state_dict_{}.pt".format(random_state)
torch.save(clf.module.state_dict(), path)

clf.save_params(
    f_params=result_path + "model.pkl",
    f_optimizer=result_path + "opt.pkl",
    f_history=result_path + "history.json",
)


######################################################################
# Loading the NMT dataset
# -------------------------------------


import mne

mne.set_log_level("ERROR")

NMT_ds = load_concat_dataset(
    NMT_path,
    preload=False,
    target_name=["pathological", "age", "gender"],  # )
    ids_to_load=range(200),
)

######################################################################
# split based on train split from dataset
train_set = NMT_ds.split("train")["True"]
test_set = NMT_ds.split("train")["False"]

import pandas as pd

print("target is being set to pathological clf")
target = NMT_ds.description["pathological"].astype(int)
for d, y in zip(NMT_ds.datasets, target):
    d.description["pathological"] = y
    d.target_name = "pathological"
    d.target = d.description[d.target_name]
NMT_ds.set_description(
    pd.DataFrame([d.description for d in NMT_ds.datasets]), overwrite=True
)

# ### extracting windows

from braindecode.datautil.windowers import create_fixed_length_windows

window_train_set = create_fixed_length_windows(
    train_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    preload=True,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=True,
)
window_test_set = create_fixed_length_windows(
    test_set,
    start_offset_samples=0,
    stop_offset_samples=None,
    preload=False,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
)
######################################################################
# Loading the pre-trained model
# -------------------------------------


load_path = result_path + "state_dict_2024.pt"
state_dicts = torch.load(load_path)
model.load_state_dict(state_dicts, strict=False)
print("pre-trained model loaded using pytorch")

## freeze layers ##
freeze = False
if freeze:
    for ii, (name, param) in enumerate(model.named_parameters()):
        # if 'temporal_block_0' in name or 'temporal_block_1'
        # in name or 'temporal_block_2' in name or
        # 'temporal_block_3' in name: # or 'temporal_block_5'
        # in name or 'conv_classifier' in name:
        if "conv_classifier" not in name:
            param.requires_grad = False
            print("param:", name, param.requires_grad)

# ## fine tuning the model on NMT dataset


from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

weight_decay = 0.5 * 0.001
lr = 0.0625 * 0.01
n_epochs = 10
batch_size = 64

clf = EEGClassifier(
    model,
    cropped=True,
    classes=[0, 1],
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(window_test_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        "balanced_accuracy",
        "f1",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device="cuda" if cuda else "cpu",
    max_epochs=n_epochs,
)

clf.initialize()  # This is important!
print("classifier initialized")
print(
    "Number of parameters = ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)


######################################################################
clf.fit(window_train_set, y=None)

######################################################################
# Conclusion
# -------------------------------------
#
# In this example, we used transfer learning (TL) as a way to learn
# representations from a large EEG data and transfer to a smaller dataset.
# You can put one of the results from your paper here.


######################################################################
# References
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# .. [1] Darvishi-Bayazi, M. J., Ghaemi, M. S., Lesort, T., Arefin, M. R.,
#    Faubert, J., & Rish, I. (2024). Amplifying pathological detection in
#    EEG signaling pathways through cross-dataset transfer learning.
#    Computers in Biology and Medicine, 169, 107893.
# .. [2] Shawki, N., Shadin, M. G., Elseify, T., Jakielaszek, L., Farkas, T.,
#    Persidsky, Y., ... & Picone, J. (2022). Correction to:
#    The temple university hospital digital pathology corpus.
#    In Signal Processing in Medicine and Biology:
#    Emerging Trends in Research and Applications (pp. C1-C1).
#    Cham: Springer International Publishing.
# .. [3] Khan, H. A., Ul Ain, R., Kamboh, A. M., & Butt, H. T. (2022).
#    The NMT scalp EEG dataset: an open-source annotated dataset of
#    healthy and pathological EEG recordings for predictive modeling.
#    Frontiers in neuroscience, 15, 755817.
# .. [4] Aristimunha, B. ...

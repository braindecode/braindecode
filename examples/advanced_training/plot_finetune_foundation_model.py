""".. _finetune-foundation-model:

Fine-tuning a Foundation Model (Signal-JEPA)
==============================================

Foundation models are large-scale pre-trained models that serve as a starting point
for a wide range of downstream tasks, leveraging their generalization capabilities.
Fine-tuning these models is necessary to adapt them to specific tasks or datasets,
ensuring optimal performance in specialized applications.

In this tutorial, we demonstrate how to load a pre-trained foundation model
and fine-tune it for a specific task. We use the Signal-JEPA model [1]_
and a MOABB motor-imagery dataset for this tutorial.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)
#
import mne
import numpy as np
import torch

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import SignalJEPA_PreLocal
from braindecode.preprocessing import create_windows_from_events

torch.use_deterministic_algorithms(True)
torch.manual_seed(12)
np.random.seed(12)


##################################################################
#
# Loading and preparing the data
# ------------------------------
#
# Loading a dataset
# ~~~~~~~~~~~~~~~~~
#
# We start by loading a MOABB dataset, a single subject only for speed.
# The dataset contains motor imagery EEG recordings, which we will preprocess and use for fine-tuning.
#

subject_id = 3  # Just one subject for speed
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

# Set the standard 10-20 montage for EEG channel locations
montage = mne.channels.make_standard_montage("standard_1020")
for ds in dataset.datasets:
    ds.raw.set_montage(montage)

##################################################################
# Preprocessing to match the pretrained model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The pretrained SignalJEPA checkpoint expects 19 EEG channels at 128 Hz
# with 2-second windows.  We adapt the dataset accordingly: keep only
# EEG channels, pick the first 19, and resample.
#

for ds in dataset.datasets:
    ds.raw.pick_types(eeg=True)  # drop EOG / stim channels
    ds.raw.pick(ds.raw.ch_names[:19])  # match pretrained channel count
    ds.raw.resample(128)  # match pretrained sampling frequency

##################################################################
# Define Dataset parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We extract the sampling frequency and channel information after
# preprocessing so they match the pretrained model.
#

sfreq = dataset.datasets[0].raw.info["sfreq"]
chs_info = dataset.datasets[0].raw.info["chs"]

print(f"{sfreq=}, {len(chs_info)=}")

##################################################################
# Create Windows from Events
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use the `create_windows_from_events` function from Braindecode to segment
# the dataset into windows based on events.
#

classes = ["feet", "left_hand", "right_hand"]
classes_mapping = {c: i for i, c in enumerate(classes)}

windows_dataset = create_windows_from_events(
    dataset,
    preload=True,  # Preload the data into memory for faster processing
    mapping=classes_mapping,
    window_size_samples=256,  # 2 s at 128 Hz — matches pretrained model
    window_stride_samples=256,
)
metadata = windows_dataset.get_metadata()
print(metadata.head(10))

##################################################################
#
# Loading a pre-trained foundation model
# --------------------------------------
#
# Load Pre-trained Weights from the Hub
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We load the pre-trained SignalJEPA downstream model from the Hugging Face
# Hub using ``from_pretrained``.  The ``SignalJEPA_PreLocal`` checkpoint
# already bundles the SSL backbone together with the downstream
# classification layers, so a single call is all that is needed.
#
# For other foundation models (BENDR, BIOT, Labram, etc.) the same
# one-line pattern applies — see :ref:`load-pretrained-models`.
#

model = SignalJEPA_PreLocal.from_pretrained(
    "braindecode/signal-jepa_without-chans",
    n_chans=19,
    n_times=256,
    n_outputs=len(classes),
    strict=False,
)
print(model)

##################################################################
#
# Fine-tuning the Model
# ---------------------
#
# Signal-JEPA is a model trained in a self-supervised manner on a masked
# prediction task. In this task, the model is configured in a many-to-many
# fashion, which is not suited for a classification task. Therefore, we need to
# adjust the model architecture for finetuning. This is what is done by the
# :class:`SignalJEPA_PreLocal`, :class:`SignalJEPA_Contextual`, and
# :class:`SignalJEPA_PostLocal` classes. In these classes, new layers are added
# specifically for classification, as described in the article [1]_ and in the following figure:
#
# .. image:: /_static/model/sjepa_pre-local.jpg
#    :alt: Signal-JEPA Pre-Local Downstream Architecture
#    :align: center
#
# With this downstream architecture, two options are possible for fine-tuning:
#
# 1) Fine-tune only the newly added layers
# 2) Fine-tune the entire model
#
# Freezing Pre-trained Layers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As the second option is rather straightforward to implement,
# we will focus on the first option here.
# We will freeze all layers except the newly added ones.
#

# Keep the task-specific head layers (spatial_conv and final_layer)
# trainable and freeze the pretrained backbone.
new_layers = {
    name
    for name, _ in model.named_parameters()
    if name.startswith(("spatial_conv.", "final_layer."))
}

for name, param in model.named_parameters():
    if name not in new_layers:
        param.requires_grad = False

print("Trainable parameters:")
other_modules = set()
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
    else:
        other_modules.add(name.split(".")[0])

print("\nOther modules:")
print(other_modules)

#############################################################
# Fine-tuning Procedure
# ~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we set up the fine-tuning procedure using Braindecode's
# :class:`EEGClassifier`. We define the loss function, optimizer, and training
# parameters. We then fit the model to the windows dataset.
#
# We only train for a few epochs for demonstration purposes.
#

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=0.005,
    batch_size=16,
    callbacks=["accuracy"],
    classes=range(3),
)
_ = clf.fit(windows_dataset, y=metadata["target"], epochs=10)

#############################################################
# All-in-one Implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the implementation above, we manually loaded the weights and froze the layers.
# This forces us to pass an initialized model to :class:`EEGClassifier`, which may
# create issues if we use it in a cross-validation setting.
#
# Instead, we can implement the same procedure in a more compact and reproducible way,
# by using skorch's callback system.
#
# Here, we import a callback to freeze layers and define a custom
# callback to load the pre-trained weights at the beginning of training:
#

from skorch.callbacks import Callback, Freezer


class WeightsLoader(Callback):
    def __init__(self, url, strict=False):
        self.url = url
        self.strict = strict

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        state_dict = torch.hub.load_state_dict_from_url(url=self.url)
        net.module_.load_state_dict(state_dict, strict=self.strict)


#############################################################
# We can now define a classifier with those callbacks, without having
# to pass an initialized model, and fit it as before:
#

clf = EEGClassifier(
    "SignalJEPA_PreLocal",
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=0.005,
    batch_size=16,
    callbacks=[
        "accuracy",
        WeightsLoader(
            url="https://huggingface.co/braindecode/signal-jepa_without-chans/resolve/main/pytorch_model.bin"
        ),
        Freezer(patterns="feature_encoder.*"),
    ],
    classes=range(3),
)
_ = clf.fit(windows_dataset, y=metadata["target"], epochs=10)

#############################################################
#
# Conclusion and Next Steps
# -------------------------
#
# In this tutorial, we demonstrated how to fine-tune a pre-trained foundation
# model, Signal-JEPA, for a motor imagery classification task. We now have a basic
# implementation that can automatically load pre-trained weights and freeze specific layers.
#
# This setup can easily be extended to explore different fine-tuning techniques,
# base foundation models, and downstream tasks.
#

#############################################################
#
# References
# ----------
#
# .. [1] Guetschel, P., Moreau, T., and Tangermann, M. (2024)
#        “S-JEPA: towards seamless cross-dataset transfer
#        through dynamic spatial attention”.  https://arxiv.org/abs/2403.11772

""".. _finetune-foundation-model:

Fine-tuning a Foundation Model (Signal-JEPA)
===========================================

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
# Define Dataset parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We extract the sampling frequency and ensure that it is consistent across
# all recordings. We also extract the window size from the annotations and
# information about the EEG channels (names, positions, etc.).
#

# Extract sampling frequency
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

# Extract and validate window size from annotations
window_size_seconds = dataset.datasets[0].raw.annotations.duration[0]
assert all(
    d == window_size_seconds
    for ds in dataset.datasets
    for d in ds.raw.annotations.duration
)

# Extract channel information
chs_info = dataset.datasets[0].raw.info["chs"]  # Channel information

print(f"{sfreq=}, {window_size_seconds=}, {len(chs_info)=}")

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
)
metadata = windows_dataset.get_metadata()
print(metadata.head(10))

##################################################################
#
# Loading a pre-trained foundation model
# --------------------------------------
#
# Download and Load Pre-trained Weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We download the pre-trained weights for the SignalJEPA model from the Hugging Face Hub.
# These weights will serve as the starting point for finetuning.
#

model_state_dict = torch.hub.load_state_dict_from_url(
    url="https://huggingface.co/braindecode/SignalJEPA/resolve/main/signal-jepa_16s-60_adeuwv4s.pth"
)
# print(model_state_dict.keys())

##################################################################
# Instantiate the Foundation Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We create an instance of the SignalJEPA model using the pre-local downstream
# architecture. The model is initialized with the dataset's sampling frequency,
# window size, and channel information.
#


model = SignalJEPA_PreLocal(
    sfreq=sfreq,
    input_window_seconds=window_size_seconds,
    chs_info=chs_info,
    n_outputs=len(classes),
)
print(model)

##################################################################
# Load the Pre-trained Weights into the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We load the pre-trained weights into the model. The transformer layers are excluded
# as this module is not used in the pre-local downstream architecture (see [1]_).
#

# Define layers to exclude from the pre-trained weights
new_layers = {
    "spatial_conv.1.weight",
    "spatial_conv.1.bias",
    "final_layer.1.weight",
    "final_layer.1.bias",
}

# Filter out transformer weights and load the state dictionary
model_state_dict = {
    k: v for k, v in model_state_dict.items() if not k.startswith("transformer.")
}
missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

# Ensure no unexpected keys and validate missing keys
assert unexpected_keys == [], f"{unexpected_keys=}"
assert set(missing_keys) == new_layers, f"{missing_keys=}"

##################################################################
#
# Fine-tuning the Model
# --------------------
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
# ~~~~~~~~~~~~~~~~~~~~
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
            url="https://huggingface.co/braindecode/SignalJEPA/resolve/main/signal-jepa_16s-60_adeuwv4s.pth"
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

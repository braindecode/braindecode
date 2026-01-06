""".. _upload-eegpt-hub:

Upload EEGPT Pretrained Weights to HuggingFace Hub
===================================================

This example demonstrates how to upload pretrained EEGPT model weights
to the HuggingFace Hub using braindecode's built-in Hub integration.

The EEGPT model with pretrained weights is available at
`braindecode/eegpt-pretrained <https://huggingface.co/braindecode/eegpt-pretrained>`_.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Kkuntal990 <kukokate@ucsd.edu>
#
# License: BSD (3-clause)

######################################################################
# Loading the Lightning Checkpoint
# ---------------------------------
#
# The pretrained EEGPT weights are stored in a PyTorch Lightning checkpoint.
# We first load this checkpoint and extract the relevant weights.

from pathlib import Path

import torch

checkpoint_path = (
    Path.home() / "Downloads/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Total state_dict keys: {len(checkpoint['state_dict'])}")

# Extract only the target_encoder weights (used by braindecode EEGPT)
target_encoder_keys = [
    k for k in checkpoint["state_dict"].keys() if k.startswith("target_encoder.")
]
print(f"Target encoder keys: {len(target_encoder_keys)}")

######################################################################
# Creating the EEGPT Model with Matching Configuration
# ----------------------------------------------------
#
# We create an EEGPT model with the same configuration as the pretrained
# checkpoint. These parameters were inferred from the checkpoint structure.

import mne

from braindecode.models import EEGPT

# Create valid channel info (EEGPT requires valid channel names from CHANNEL_DICT)
# Using 58 unique channels that match the checkpoint
ch_names = [
    "FP1",
    "FPZ",
    "FP2",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO8",
]
info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=["eeg"] * 58)

# Create model with configuration matching the checkpoint
model = EEGPT(
    n_outputs=4,  # Adjust based on your use case
    n_chans=58,
    n_times=1024,  # 4 seconds at 250 Hz
    chs_info=info["chs"],
    sfreq=250.0,
    patch_size=64,
    patch_stride=32,
    embed_dim=512,
    embed_num=4,
    depth=8,
    num_heads=8,
)

print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

######################################################################
# Loading Pretrained Weights
# ---------------------------
#
# We load the target_encoder weights from the Lightning checkpoint
# into our EEGPT model.

# Extract target_encoder state dict and remove prefix
target_encoder_state = {}
for k, v in checkpoint["state_dict"].items():
    if k.startswith("target_encoder."):
        new_key = k.replace("target_encoder.", "")
        target_encoder_state[new_key] = v

# Load weights into model's target_encoder
missing_keys, unexpected_keys = model.target_encoder.load_state_dict(
    target_encoder_state, strict=True
)

print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")
print("Weights loaded successfully!")

######################################################################
# Uploading to HuggingFace Hub
# -----------------------------
#
# Now we use the built-in push_to_hub() method to upload the model
# to HuggingFace Hub. This will:
# 1. Save the model configuration to config.json
# 2. Save the weights in both SafeTensors and PyTorch formats
# 3. Upload everything to the Hub
#
# Note: You need to be logged in to HuggingFace CLI and have write
# permissions to the braindecode organization.

# Uncomment the following line to upload (requires HF authentication)
model.push_to_hub("braindecode/eegpt-pretrained")

print("\nTo upload the model, run:")
print("  huggingface-cli login")
print("  # Then uncomment the push_to_hub line in this script")

######################################################################
# Loading from Hub
# ----------------
#
# Once uploaded, anyone can load the model with:
#
# .. code-block:: python
#
#     from braindecode.models import EEGPT
#     model = EEGPT.from_pretrained("braindecode/eegpt-pretrained")
#

######################################################################
# References
# ----------
#
# .. [1] Young-Truong, D., et al. (2024). EEGPT: Pretrained Transformer for
#        Universal and Generalizable EEG Representation Learning.
#        https://arxiv.org/abs/2410.16690

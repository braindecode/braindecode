""".. _hub-integration:

Uploading and downloading datasets to Hugging Face Hub
=======================================================

This example demonstrates how to upload and download EEG datasets to/from
the Hugging Face Hub using braindecode.

The Hub integration supports three dataset types:

1. **WindowsDataset** - Epoched data (mne.Epochs-based)
2. **EEGWindowsDataset** - Continuous raw data with windowing metadata
3. **RawDataset** - Continuous raw data without windowing

The Hub integration allows you to:

- **Share datasets** with the research community
- **Version control** your datasets with git-like versioning
- **Collaborate** on dataset curation and preprocessing
- **Access datasets** from anywhere with automatic caching

We'll use the :class:`braindecode.datasets.BNCI2014_001` dataset as an example.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

import os

from huggingface_hub import login

from braindecode.datasets import BNCI2014_001, BaseConcatDataset
from braindecode.preprocessing import (
    create_windows_from_events,
)

# Login to Hugging Face Hub using token from environment variable
hf_token = os.environ.get("HUGGING_FACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HUGGING_FACE_TOKEN not set. Hub uploads will fail.")

###############################################################################
# Load and prepare datasets
# --------------------------
# We'll demonstrate all three supported dataset types.

print("Loading BNCI2014_001 dataset...")
# Load only subject 1 for this example
dataset = BNCI2014_001(subject_ids=[1])

print(f"  Number of recordings: {len(dataset.datasets)}")
print(f"  Channels: {len(dataset.datasets[0].raw.ch_names)}")
print(f"  Sampling frequency: {dataset.datasets[0].raw.info['sfreq']} Hz")

###############################################################################
# Example 1: WindowsDataset (Epoched data)
# -----------------------------------------
# Create epoched data using mne.Epochs (use_mne_epochs=True)

print("\n1. Creating WindowsDataset (epoched data)...")
windows_dataset = create_windows_from_events(
    concat_ds=BaseConcatDataset([dataset.datasets[0]]),
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    use_mne_epochs=True,  # Creates WindowsDataset with mne.Epochs
)

print(f"   Total windows: {len(windows_dataset)}")
print("   Dataset type: WindowsDataset (epoched)")

###############################################################################
# Example 2: EEGWindowsDataset (Continuous with windowing)
# ---------------------------------------------------------
# Create continuous raw data with windowing metadata (use_mne_epochs=False)

print("\n2. Creating EEGWindowsDataset (continuous with windowing)...")
eegwindows_dataset = create_windows_from_events(
    concat_ds=BaseConcatDataset([dataset.datasets[0]]),
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    use_mne_epochs=False,  # Creates EEGWindowsDataset with continuous raw
)

print(f"   Total windows: {len(eegwindows_dataset)}")
print("   Dataset type: EEGWindowsDataset (continuous)")

###############################################################################
# Example 3: RawDataset (Continuous without windowing)
# -----------------------------------------------------
# Use the original raw data without any windowing

print("\n3. Using RawDataset (continuous without windowing)...")
raw_dataset = BaseConcatDataset([dataset.datasets[0]])

print(f"   Number of recordings: {len(raw_dataset.datasets)}")
print("   Dataset type: RawDataset (continuous, no windows)")

###############################################################################
# Upload datasets to Hugging Face Hub
# ------------------------------------
# To upload a dataset, you need to:
#
# 1. Create a Hugging Face account at https://huggingface.co
# 2. Login using: ``huggingface-cli login``
# 3. Choose a repository name (e.g., "username/dataset-name")
#
# **Note:** This example shows the code but doesn't actually upload.
# Uncomment the code below to perform an actual upload.

print("\n" + "=" * 70)
print("UPLOADING TO HUGGING FACE HUB")
print("=" * 70)

# Skip Hub operations if no token is available (e.g., during docs build)
if not hf_token:
    print("\n⚠️  Skipping Hub upload examples (no HUGGING_FACE_TOKEN set)")
    print("To run this example with actual uploads, set HUGGING_FACE_TOKEN")
else:
    # Example 1: Upload WindowsDataset
    # ---------------------------------
    repo_id_windows = "braindecode/example_dataset-windows"

    print(f"\nUploading WindowsDataset to {repo_id_windows}...")
    url = windows_dataset.push_to_hub(
        repo_id=repo_id_windows,
        commit_message="Upload BNCI2014_001 WindowsDataset (epoched)",
        private=False,
    )
    print(f"✅ Uploaded to {url}!")

    # Example 2: Upload EEGWindowsDataset
    # ------------------------------------
    repo_id_eegwindows = "braindecode/example_dataset-eegwindows"

    print(f"\nUploading EEGWindowsDataset to {repo_id_eegwindows}...")
    url = eegwindows_dataset.push_to_hub(
        repo_id=repo_id_eegwindows,
        commit_message="Upload BNCI2014_001 EEGWindowsDataset (continuous)",
        private=False,
    )
    print(f"✅ Uploaded to {url}!")

    # Example 3: Upload RawDataset
    # -----------------------------
    repo_id_raw = "braindecode/example_dataset-raw"

    print(f"\nUploading RawDataset to {repo_id_raw}...")
    url = raw_dataset.push_to_hub(
        repo_id=repo_id_raw,
        commit_message="Upload BNCI2014_001 RawDataset",
        private=False,
    )
    print(f"✅ Uploaded to {url}!")

print("""
The example above demonstrates uploading to the Hugging Face Hub.
All datasets are converted to Zarr format (optimized for fast loading)
and uploaded with auto-generated dataset cards.

For your own datasets, replace the repo_id with your Hugging Face username.
""")

###############################################################################
# Zarr Format
# -----------
# Datasets are uploaded in Zarr format, which provides:
#
# - Fastest random access (0.010 ms - critical for PyTorch training)
# - Excellent compression with blosc
# - Cloud-native, chunked storage
# - Ideal for datasets of all sizes
# - Based on comprehensive benchmarking with 1000 subjects
#
# The format parameters (compression, compression_level) are optimized by default
# but can be customized if needed.

###############################################################################
# Download datasets from Hugging Face Hub
# ----------------------------------------
# Loading datasets from the Hub is simple and automatic!

print("\n" + "=" * 70)
print("DOWNLOADING FROM HUGGING FACE HUB")
print("=" * 70)

# Skip Hub downloads if no token (docs build)
if not hf_token:
    print("\n⚠️  Skipping Hub download examples (no HUGGING_FACE_TOKEN set)")
    print("To run this example with actual downloads, set HUGGING_FACE_TOKEN")
else:
    # Example 1: Download WindowsDataset
    # -----------------------------------
    public_repo_windows = "braindecode/example_dataset-windows"

    print(f"\nDownloading WindowsDataset from {public_repo_windows}...")
    loaded_windows = BaseConcatDataset.pull_from_hub(
        public_repo_windows,
        preload=True,  # Load into memory (False for lazy loading)
    )
    print("✅ Loaded WindowsDataset!")
    print(f"   Number of windows: {len(loaded_windows)}")

    # Example 2: Download EEGWindowsDataset
    # --------------------------------------
    public_repo_eeg = "braindecode/example_dataset-eegwindows"

    print(f"\nDownloading EEGWindowsDataset from {public_repo_eeg}...")
    loaded_eeg = BaseConcatDataset.pull_from_hub(
        public_repo_eeg,
        preload=True,
    )
    print("✅ Loaded EEGWindowsDataset!")
    print(f"   Number of windows: {len(loaded_eeg)}")

    # Example 3: Download RawDataset
    # -------------------------------
    public_repo_raw = "braindecode/example_dataset-raw"

    print(f"\nDownloading RawDataset from {public_repo_raw}...")
    loaded_raw = BaseConcatDataset.pull_from_hub(
        public_repo_raw,
        preload=True,
    )
    print("✅ Loaded RawDataset!")
    print(f"   Number of recordings: {len(loaded_raw.datasets)}")

print("""
The example above demonstrates downloading datasets from the Hub.
Datasets are automatically downloaded and cached locally.
Subsequent loads use the cache for faster access.
""")

###############################################################################
# Using datasets with PyTorch DataLoader
# ---------------------------------------
# Datasets loaded from the Hub work seamlessly with PyTorch.

print("\n" + "=" * 70)
print("USING WITH PYTORCH DATALOADER")
print("=" * 70)

from torch.utils.data import DataLoader

# Create DataLoader
train_loader = DataLoader(
    windows_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # Set to > 0 for parallel loading
)

print("\nDataLoader created:")
print("  Batch size: 32")
print(f"  Total batches: {len(train_loader)}")

# Iterate over a few batches
print("\n  Sample batches:")
for i, batch_data in enumerate(train_loader):
    if i >= 3:  # Show only 3 batches
        break
    # Handle both 2-tuple (X, y) and 3-tuple (X, y, inds) returns
    if len(batch_data) == 3:
        X_batch, y_batch, _ = batch_data
    else:
        X_batch, y_batch = batch_data
    print(
        f"    Batch {i + 1}: X shape={tuple(X_batch.shape)}, "
        f"y shape={tuple(y_batch.shape)}"
    )

print("""
This works the same way for datasets loaded from the Hub!
The Hub integration is fully compatible with PyTorch's training pipeline.
""")

###############################################################################
# Advanced: Version control and collaboration
# --------------------------------------------
# The Hub provides powerful features for dataset management:
#
# **Versioning**
#
# Every upload creates a new commit, allowing you to track changes:
#
# .. code-block:: python
#
#     # Upload updated version
#     dataset.push_to_hub(
#         repo_id="username/dataset-name",
#         commit_message="Fixed label for subject 5"
#     )
#
# **Private datasets**
#
# For sensitive data or work-in-progress:
#
# .. code-block:: python
#
#     dataset.push_to_hub(
#         repo_id="username/private-dataset",
#         private=True
#     )
#
# **Pull requests**
#
# Propose changes without directly modifying the dataset:
#
# .. code-block:: python
#
#     dataset.push_to_hub(
#         repo_id="username/dataset-name",
#         create_pr=True,
#         commit_message="Propose additional preprocessing"
#     )

###############################################################################
# Best practices
# ---------------
# When sharing datasets on the Hub:
#
# 1. **Include good documentation** - The dataset card (README.md) is auto-
#    generated but you can edit it on the Hub to add details about:
#
#    - Data collection methodology
#    - Preprocessing steps applied
#    - Known issues or limitations
#    - Citation information
#
# 2. **Optimize compression if needed** - The default blosc compression (level 5)
#    provides an optimal balance. For very large datasets, experiment with
#    compression_level parameter (0-9) to find the best trade-off between
#    size and speed for your use case.
#
# 3. **Test before sharing** - Always test that your uploaded dataset can be
#    downloaded and used correctly:
#
#    .. code-block:: python
#
#        # Upload
#        dataset.push_to_hub("braindecode/example_dataset")
#
#        # Test download
#        test_dataset = BaseConcatDataset.pull_from_hub("braindecode/example_dataset")
#        assert len(test_dataset) == len(dataset)
#
# 4. **Consider privacy** - Ensure you have permission to share the data and
#    that personal information has been properly anonymized.

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Hugging Face Hub integration supports three dataset types:

✓ WindowsDataset - Epoched data (mne.Epochs)
✓ EEGWindowsDataset - Continuous with windowing metadata
✓ RawDataset - Continuous without windowing

Benefits:
✓ Share datasets with the research community
✓ Version control with git-like versioning
✓ Collaborate on dataset curation
✓ Access datasets from anywhere
✓ Automatic caching for faster repeated loads
✓ Optimized Zarr format for fast training

For more information:
- Hugging Face Hub: https://huggingface.co
- Braindecode docs: https://braindecode.org
- Hub docs: https://huggingface.co/docs/hub
""")

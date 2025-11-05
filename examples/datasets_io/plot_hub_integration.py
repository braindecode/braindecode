""".. _hub-integration:

Uploading and downloading datasets from Hugging Face Hub
=========================================================

This example demonstrates how to upload and download EEG datasets to/from
the Hugging Face Hub using braindecode.

The Hub integration allows you to:

1. **Share datasets** with the community
2. **Version control** your datasets
3. **Collaborate** with others on dataset curation
4. **Access datasets** from anywhere with a simple API

We'll use the :class:`braindecode.datasets.NMT` dataset as an example.
"""

# Authors: Kuntal Kokate
#
# License: BSD (3-clause)

from braindecode.datasets import BaseConcatDataset, NMT
from braindecode.preprocessing import create_fixed_length_windows

###############################################################################
# Load and prepare a dataset
# ---------------------------
# First, we'll load a small subset of the NMT dataset and create windows.

print("Loading NMT dataset...")
# Load only 2 subjects for this example
dataset = NMT(recording_ids=[0, 1], preload=True)

print(f"  Number of recordings: {len(dataset.datasets)}")
print(f"  Channels: {len(dataset.datasets[0].raw.ch_names)}")
print(f"  Sampling frequency: {dataset.datasets[0].raw.info['sfreq']} Hz")

# Create fixed-length windows
sfreq = dataset.datasets[0].raw.info["sfreq"]
windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=int(4 * sfreq),  # 4-second windows
    window_stride_samples=int(2 * sfreq),  # 2-second stride
    drop_last_window=True,
    preload=True,
)

print(f"\n  Total windows created: {len(windows_dataset)}")

###############################################################################
# Upload dataset to Hugging Face Hub
# -----------------------------------
# To upload a dataset, you need to:
#
# 1. Create a Hugging Face account at https://huggingface.co
# 2. Login using: ``huggingface-cli login``
# 3. Choose a repository name (e.g., "username/nmt-sample-dataset")
#
# **Note:** This example shows the code but doesn't actually upload.
# Uncomment the code below to perform an actual upload.

# Set your repository ID
repo_id = "your-username/nmt-sample-dataset"  # Change this!

print("\n" + "=" * 70)
print("UPLOADING TO HUGGING FACE HUB")
print("=" * 70)

# Uncomment the following lines to actually upload:
# print(f"\nUploading dataset to {repo_id}...")
# url = windows_dataset.push_to_hub(
#     repo_id=repo_id,
#     commit_message="Upload NMT sample dataset with 4s windows",
#     private=False,  # Set to True for private datasets
# )
# print(f"✅ Dataset uploaded successfully!")
# print(f"   URL: https://huggingface.co/datasets/{repo_id}")

print("""
To upload this dataset, uncomment the code above and:
1. Replace 'your-username' with your Hugging Face username
2. Login with: huggingface-cli login
3. Run this script

The dataset will be converted to Zarr format (optimized for training)
and uploaded with metadata, making it easy for others to discover and use.
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
# Download dataset from Hugging Face Hub
# ---------------------------------------
# Loading a dataset from the Hub is even simpler!

print("\n" + "=" * 70)
print("DOWNLOADING FROM HUGGING FACE HUB")
print("=" * 70)

# Example: Loading a public dataset (replace with actual repo)
# public_repo = "username/nmt-sample-dataset"

# Uncomment to download:
# print(f"\nDownloading dataset from {public_repo}...")
# loaded_dataset = BaseConcatDataset.from_pretrained(
#     public_repo,
#     preload=True,  # Load into memory (False for lazy loading)
# )
# print(f"✅ Dataset loaded!")
# print(f"   Number of windows: {len(loaded_dataset)}")

print("""
To download a dataset from the Hub:

>>> from braindecode.datasets import BaseConcatDataset
>>> dataset = BaseConcatDataset.from_pretrained("username/dataset-name")

The dataset will be automatically downloaded and cached locally.
Subsequent loads will be faster as they use the cached version.
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
print(f"  Batch size: 32")
print(f"  Total batches: {len(train_loader)}")

# Iterate over a few batches
print("\n  Sample batches:")
for i, (X_batch, y_batch) in enumerate(train_loader):
    if i >= 3:  # Show only 3 batches
        break
    print(f"    Batch {i+1}: X shape={tuple(X_batch.shape)}, "
          f"y shape={tuple(y_batch.shape)}")

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
#        dataset.push_to_hub("username/my-dataset")
#
#        # Test download
#        test_dataset = BaseConcatDataset.from_pretrained("username/my-dataset")
#        assert len(test_dataset) == len(dataset)
#
# 4. **Consider privacy** - Ensure you have permission to share the data and
#    that personal information has been properly anonymized.

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Hugging Face Hub integration makes it easy to:

✓ Share datasets with the community
✓ Version control your data
✓ Collaborate on dataset curation
✓ Access datasets from anywhere
✓ Automatically cache downloads for faster loading

For more information:
- Hugging Face Hub: https://huggingface.co
- Braindecode docs: https://braindecode.org
- Hub docs: https://huggingface.co/docs/hub
""")

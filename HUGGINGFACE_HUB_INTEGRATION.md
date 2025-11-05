# Hugging Face Hub Integration for Braindecode Datasets

This document describes the implementation of Hugging Face Hub integration for braindecode datasets, enabling easy sharing, versioning, and collaboration on EEG datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Implementation Summary](#implementation-summary)
- [Architecture](#architecture)
- [Storage Formats](#storage-formats)
- [Usage Guide](#usage-guide)
- [Benchmarking](#benchmarking)
- [Testing](#testing)
- [Next Steps](#next-steps)

## 🎯 Overview

### Problem Statement

EEG datasets in braindecode use MNE's `.fif` format, which is not directly compatible with Hugging Face Hub. We need a solution that:

1. Converts datasets to Hub-compatible formats
2. Maintains fast random access (critical for DataLoader)
3. Preserves all metadata and preprocessing history
4. Enables easy upload/download from the Hub
5. Supports lazy loading for large datasets

### Solution

We implemented a complete Hub integration system with:

- **Three storage formats**: HDF5, Zarr, and NumPy+Parquet
- **Hub mixin**: `push_to_hub()` and `from_pretrained()` methods
- **Comprehensive benchmarking**: To select the optimal format
- **Full test coverage**: Unit and integration tests
- **Documentation**: Examples and guides

## 📁 Implementation Summary

### Files Created/Modified

#### **PR #1: Format Compatibility & Benchmarking**

| File | Description | Lines |
|------|-------------|-------|
| `pyproject.toml` | Added `hub` optional dependencies | Modified |
| `braindecode/datautil/hub_formats.py` | Format converters (HDF5, Zarr, npz+parquet) | ~1100 |
| `examples/datasets_io/plot_benchmark_hub_formats.py` | Comprehensive benchmark script | ~400 |
| `test/unit_tests/datautil/test_hub_formats.py` | Format converter tests | ~450 |

#### **PR #2: Hub Integration**

| File | Description | Lines |
|------|-------------|-------|
| `braindecode/datasets/hub_mixin.py` | Hub integration mixin class | ~400 |
| `braindecode/datasets/base.py` | Modified to inherit from HubDatasetMixin | Modified |
| `examples/datasets_io/plot_hub_integration.py` | Usage example and tutorial | ~300 |
| `test/unit_tests/datasets/test_hub_integration.py` | Hub integration tests | ~350 |

**Total**: ~2,600 lines of code + documentation

## 🏗️ Architecture

### Component Diagram

```
BaseConcatDataset (base.py)
    │
    ├── Inherits from: ConcatDataset (PyTorch)
    └── Inherits from: HubDatasetMixin (hub_mixin.py)
                          │
                          ├── push_to_hub()
                          │   └── Uses hub_formats.py converters
                          │       ├── convert_to_hdf5()
                          │       ├── convert_to_zarr()
                          │       └── convert_to_npz_parquet()
                          │
                          └── from_pretrained()
                              └── Uses hub_formats.py loaders
                                  ├── load_from_hdf5()
                                  ├── load_from_zarr()
                                  └── load_from_npz_parquet()
```

### Data Flow

**Upload Flow:**
```
BaseConcatDataset (.fif format)
    │
    ├── 1. Convert to Hub format (HDF5/Zarr/npz+parquet)
    ├── 2. Generate dataset card (README.md)
    ├── 3. Save metadata (format_info.json)
    └── 4. Upload to Hub → https://huggingface.co/datasets/user/dataset
```

**Download Flow:**
```
Hub Repository
    │
    ├── 1. Download format_info.json
    ├── 2. Download dataset files (based on format)
    ├── 3. Load using format-specific loader
    └── 4. Return BaseConcatDataset → Ready for PyTorch
```

## 💾 Storage Formats

### Format Comparison

| Feature | HDF5 | Zarr | NumPy+Parquet |
|---------|------|------|---------------|
| **Random Access** | ⭐⭐⭐ Excellent | ⭐⭐⭐ Excellent | ⭐⭐ Good |
| **File Structure** | Single file | Directory | Directory |
| **Compression** | gzip, lzf | blosc, zstd, gzip | zstd, gzip, snappy |
| **Cloud Streaming** | ⭐⭐ Good | ⭐⭐⭐ Excellent | ⭐ Limited |
| **Lazy Loading** | ✅ Supported | ✅ Supported | ⚠️ Partial |
| **Metadata** | Hierarchical attrs | Hierarchical attrs | Separate parquet |
| **Best For** | Medium datasets | Large datasets | Small datasets |

### Recommendations

Based on benchmarking:

1. **HDF5 (Recommended)** - Best for most use cases
   - Fast random access (critical for training)
   - Good compression (gzip level 4)
   - Single file (easier to manage)
   - Widely supported

2. **Zarr** - For very large datasets (> 1GB)
   - Cloud-native chunked storage
   - Better for streaming from Hub
   - Excellent for distributed training

3. **NumPy+Parquet** - For small datasets (< 100MB)
   - Simplest format
   - Easy to inspect with standard tools
   - Good for quick prototyping

## 📖 Usage Guide

### Installation

```bash
# Install braindecode with Hub support
pip install braindecode[hub]

# Or install dependencies separately
pip install braindecode huggingface-hub zarr pyarrow
```

### Uploading a Dataset

```python
from braindecode.datasets import NMT
from braindecode.preprocessing import create_fixed_length_windows

# 1. Load and prepare dataset
dataset = NMT(recording_ids=[0, 1, 2], preload=True)

# 2. Create windows (optional)
windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=1000,
    window_stride_samples=500,
    drop_last_window=True,
)

# 3. Upload to Hub
windows_dataset.push_to_hub(
    repo_id="username/nmt-dataset",
    format="hdf5",                    # Choose format
    compression="gzip",                # Optional
    compression_level=4,               # Optional
    commit_message="Add NMT dataset",  # Optional
    private=False,                     # Make it public
)
```

### Downloading a Dataset

```python
from braindecode.datasets import BaseConcatDataset
from torch.utils.data import DataLoader

# Load from Hub (automatically cached locally)
dataset = BaseConcatDataset.from_pretrained(
    "username/nmt-dataset",
    preload=True  # Or False for lazy loading
)

# Use with PyTorch
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Train your model
for X_batch, y_batch in train_loader:
    # X_batch: (batch_size, n_channels, n_times)
    # y_batch: (batch_size,)
    pass
```

### Authentication

```bash
# Login to Hugging Face (one-time setup)
huggingface-cli login

# Paste your token from: https://huggingface.co/settings/tokens
```

## 📊 Benchmarking

### Running Benchmarks

```bash
# Run the benchmark script
python examples/datasets_io/plot_benchmark_hub_formats.py
```

### Metrics Measured

1. **Random Access Speed** ⭐ (Most Important)
   - Time to access random windows
   - Critical for DataLoader performance

2. **Sequential Read Speed**
   - Time to iterate through all windows
   - Important for preprocessing

3. **Save Time**
   - Time to convert and save dataset

4. **Load Time**
   - Cold start loading time

5. **File Size**
   - Storage efficiency with compression

6. **Memory Efficiency**
   - RAM usage during operations

### Expected Results (NMT dataset, 2 subjects)

| Format | Random Access (ms/sample) | File Size (MB) | Load Time (s) |
|--------|---------------------------|----------------|---------------|
| HDF5 (gzip-4) | ~0.5 | ~15 | ~2.0 |
| HDF5 (no comp) | ~0.4 | ~45 | ~1.5 |
| Zarr (blosc) | ~0.6 | ~18 | ~2.2 |
| npz+parquet | ~0.8 | ~20 | ~2.5 |

*Note: Actual results depend on hardware and dataset size*

## 🧪 Testing

### Test Coverage

#### Format Converters (`test_hub_formats.py`)

- ✅ HDF5 round-trip (raw and windowed)
- ✅ Zarr round-trip (raw and windowed)
- ✅ NumPy+Parquet round-trip (raw and windowed)
- ✅ Compression options
- ✅ Partial loading (specific recordings)
- ✅ Overwrite functionality
- ✅ Data integrity verification
- ✅ Metadata preservation
- ✅ Cross-format compatibility
- ✅ Error handling

#### Hub Integration (`test_hub_integration.py`)

- ✅ Mixin methods availability
- ✅ Import error handling
- ✅ Empty dataset validation
- ✅ Invalid format detection
- ✅ Dataset card generation
- ✅ Mock Hub upload/download
- ✅ Preprocessing kwargs preservation
- ✅ Repository creation errors
- ✅ Upload/download failures
- ✅ 404 handling

### Running Tests

```bash
# Run all Hub-related tests
pytest test/unit_tests/datautil/test_hub_formats.py -v
pytest test/unit_tests/datasets/test_hub_integration.py -v

# Run specific test
pytest test/unit_tests/datautil/test_hub_formats.py::test_hdf5_round_trip_windows -v

# Run with coverage
pytest test/unit_tests/datautil/ --cov=braindecode.datautil.hub_formats
```

## 🚀 Next Steps

### For Users

1. **Run Benchmarks**
   ```bash
   python examples/datasets_io/plot_benchmark_hub_formats.py
   ```
   This will help you choose the best format for your dataset.

2. **Upload Your Dataset**
   ```python
   dataset.push_to_hub("username/my-dataset", format="hdf5")
   ```

3. **Share with Community**
   - Add detailed documentation to the dataset card
   - Include citation information
   - Provide usage examples

### For Developers

#### PR #1: Format Compatibility (Ready for Review)

**What to Test:**
1. Install dependencies: `pip install zarr pyarrow`
2. Run format converter tests: `pytest test/unit_tests/datautil/test_hub_formats.py`
3. Run benchmark: `python examples/datasets_io/plot_benchmark_hub_formats.py`
4. Verify all three formats work correctly

**Review Checklist:**
- [ ] All tests pass
- [ ] Benchmark runs successfully
- [ ] Documentation is clear
- [ ] Code follows braindecode style
- [ ] Type hints are present
- [ ] Examples are working

#### PR #2: Hub Integration (Ready for Review)

**What to Test:**
1. Install hub dependencies: `pip install huggingface-hub`
2. Run hub tests: `pytest test/unit_tests/datasets/test_hub_integration.py`
3. Try the example: `python examples/datasets_io/plot_hub_integration.py`
4. (Optional) Test actual Hub upload if you have an account

**Review Checklist:**
- [ ] All tests pass
- [ ] Hub mixin integrates cleanly
- [ ] Examples are clear and working
- [ ] Error messages are helpful
- [ ] Dataset cards are informative
- [ ] Backward compatibility maintained

### Future Enhancements

1. **Lazy Loading Improvements**
   - Implement true lazy loading for HDF5/Zarr
   - Reduce memory footprint for large datasets

2. **Streaming Support**
   - Enable training directly from Hub without full download
   - Useful for very large datasets

3. **Dataset Search & Discovery**
   - Create a dataset catalog on Hub
   - Add tags for easy filtering

4. **Multi-Modal Support**
   - Extend to other modalities (MEG, ECoG)
   - Unified format for all braindecode datasets

5. **Compression Optimization**
   - Auto-select best compression based on dataset
   - Balance between size and access speed

6. **Version Control Features**
   - Dataset diffs between versions
   - Rollback capabilities
   - Change logs

## 📝 Notes

### Backward Compatibility

- ✅ Existing `.fif` based save/load still works
- ✅ No breaking changes to existing API
- ✅ Hub integration is opt-in (requires `[hub]` extras)

### Performance Considerations

- **HDF5 gzip level 4**: Sweet spot for compression vs speed
- **Zarr blosc**: Fastest compression for large datasets
- **Chunking**: Optimized for random access patterns
- **Caching**: Hub downloads are cached automatically

### Known Limitations

1. **Lazy Loading**: Currently simplified - full lazy loading needs work
2. **Large Datasets**: May require streaming support for > 10GB
3. **Custom Metadata**: Some MNE metadata might not be fully preserved

## 📚 References

- [Braindecode Documentation](https://braindecode.org)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [HDF5 Documentation](https://docs.h5py.org)
- [Zarr Documentation](https://zarr.readthedocs.io)
- [Benchmark Repository](https://github.com/bruAristimunha/Comparing-Different-EEG-Torch-Datasets)

---

**Implementation Authors**: Kuntal Kokate, Bruno Aristimunha
**Date**: January 2025
**Braindecode Version**: 1.0+
**License**: BSD-3-Clause

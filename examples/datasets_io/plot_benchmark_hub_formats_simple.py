
import os
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from braindecode.datasets import NMT, BaseConcatDataset
from braindecode.datautil.hub_formats import (
    convert_to_hdf5,
    convert_to_npz_parquet,
    convert_to_zarr,
    get_format_info,
    load_from_hdf5,
    load_from_npz_parquet,
    load_from_zarr,
)
from braindecode.preprocessing import create_fixed_length_windows

###############################################################################
# Configuration
n_subjects = 1000  # Using 1000 subjects for comprehensive large-scale benchmark
n_repetitions = 2
n_random_samples = 1000  # Number of random access samples to test

formats_to_test = [
    ("fif", {}),  # MNE's native format (braindecode's current default)
    ("hdf5", {"compression": "gzip", "compression_level": 4}),
    ("hdf5", {"compression": "gzip", "compression_level": 9}),
    ("hdf5", {"compression": None}),
    ("zarr", {"compression": "blosc", "compression_level": 5}),
    ("zarr", {"compression": None}),
    ("npz_parquet", {"compression": "zstd"}),
    ("npz_parquet", {"compression": None}),
]

###############################################################################
# Load dataset
print("="*70)
print(f"Loading NMT dataset ({n_subjects} subjects)...")
print(f"Random access benchmark: {n_random_samples} samples")
print("="*70)

custom_path = os.path.expanduser("~/Downloads")
dataset = NMT(path=custom_path, recording_ids=list(range(n_subjects)), preload=True)

info = get_format_info(dataset)
print(f"\nDataset info: {info['n_recordings']} recordings, {info['total_size_mb']:.2f} MB")

# Check recording durations and filter out short ones
sfreq = dataset.datasets[0].raw.info["sfreq"]
print(f"\nChecking recording durations...")
durations = [len(ds.raw) / sfreq for ds in dataset.datasets]
min_duration = min(durations)
max_duration = max(durations)
print(f"Recording durations: {min_duration:.2f}s (min) to {max_duration:.2f}s (max)")

# Use 2s windows to accommodate shorter recordings
window_size_s = 2
stride_size_s = 1
print(f"\nCreating {window_size_s}s windows with {stride_size_s}s stride...")

# Filter out recordings shorter than window size
original_count = len(dataset.datasets)
dataset.datasets = [ds for ds in dataset.datasets
                    if len(ds.raw) / sfreq >= window_size_s]
filtered_count = len(dataset.datasets)
if filtered_count < original_count:
    print(f"Filtered out {original_count - filtered_count} recordings shorter than {window_size_s}s")
    print(f"Using {filtered_count} recordings for benchmark")

windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=int(window_size_s * sfreq),
    window_stride_samples=int(stride_size_s * sfreq),
    drop_last_window=True,
    preload=True,
)

info_windowed = get_format_info(windows_dataset)
print(f"Created {info_windowed['total_samples']} windows")

###############################################################################
# Benchmark functions

def benchmark_format(format_name, kwargs, windows_dataset, tmp_dir):
    """Benchmark one format configuration."""
    compression_str = kwargs.get('compression', 'none')
    if 'compression_level' in kwargs:
        compression_str += f"_{kwargs['compression_level']}"

    print(f"\n📊 {format_name.upper()} ({compression_str if compression_str != 'none' else 'native'})")

    # Set appropriate file extension and path
    if format_name == "fif":
        output_path = Path(tmp_dir) / f"test_fif"  # FIF saves as directory
    elif format_name == "hdf5":
        output_path = Path(tmp_dir) / f"test_{format_name}_{compression_str}.h5"
    else:
        output_path = Path(tmp_dir) / f"test_{format_name}_{compression_str}.{format_name}"

    results = {
        "format": format_name,
        "compression": compression_str,
        "save_times": [],
        "load_times": [],
        "random_access_times": [],
        "sequential_times": [],
        "file_size_mb": None,
    }

    for rep in range(n_repetitions):
        print(f"  Rep {rep+1}/{n_repetitions}...", end=" ")

        # 1. Save
        start = time.time()
        if format_name == "fif":
            windows_dataset.save(output_path, overwrite=True)
        elif format_name == "hdf5":
            convert_to_hdf5(windows_dataset, output_path, overwrite=True, **kwargs)
        elif format_name == "zarr":
            convert_to_zarr(windows_dataset, output_path, overwrite=True, **kwargs)
        else:
            convert_to_npz_parquet(windows_dataset, output_path, overwrite=True, **kwargs)
        results["save_times"].append(time.time() - start)

        # File size (only once)
        if results["file_size_mb"] is None:
            if format_name == "hdf5" and output_path.is_file():
                results["file_size_mb"] = os.path.getsize(output_path) / (1024 * 1024)
            else:
                # For directory-based formats (fif, zarr, npz_parquet)
                total = sum(os.path.getsize(os.path.join(dp, f))
                           for dp, dn, fnames in os.walk(output_path) for f in fnames)
                results["file_size_mb"] = total / (1024 * 1024)

        # 2. Load
        start = time.time()
        if format_name == "fif":
            loaded = BaseConcatDataset.load(output_path, preload=True)
        elif format_name == "hdf5":
            loaded = load_from_hdf5(output_path, preload=True)
        elif format_name == "zarr":
            loaded = load_from_zarr(output_path, preload=True)
        else:
            loaded = load_from_npz_parquet(output_path, preload=True)
        results["load_times"].append(time.time() - start)

        # 3. Random access (critical for DataLoader performance)
        np.random.seed(42)
        indices = np.random.randint(0, len(loaded), size=n_random_samples)
        start = time.time()
        for idx in indices:
            X, y = loaded[idx]
        results["random_access_times"].append((time.time() - start) / n_random_samples * 1000)  # ms/sample

        # 4. Sequential read (only first repetition for speed)
        if rep == 0:
            start = time.time()
            for i in range(len(loaded)):
                X, y = loaded[i]
            results["sequential_times"].append(time.time() - start)

        # Cleanup
        if output_path.is_file():
            output_path.unlink()
        elif output_path.is_dir():
            shutil.rmtree(output_path)

        print("Done")

    # Calculate stats (ensure compression is never None for CSV)
    return {
        "format": format_name,
        "compression": compression_str if compression_str != "none" else ("native" if format_name == "fif" else "none"),
        "save_time_mean": np.mean(results["save_times"]),
        "save_time_std": np.std(results["save_times"]),
        "load_time_mean": np.mean(results["load_times"]),
        "load_time_std": np.std(results["load_times"]),
        "random_access_ms_mean": np.mean(results["random_access_times"]),
        "random_access_ms_std": np.std(results["random_access_times"]),
        "sequential_time": results["sequential_times"][0] if results["sequential_times"] else 0,
        "file_size_mb": results["file_size_mb"],
    }

###############################################################################
# Run benchmarks

print("\n" + "="*70)
print("BENCHMARKING")
print("="*70)

tmp_dir = tempfile.mkdtemp()
results = []

for format_name, kwargs in formats_to_test:
    try:
        result = benchmark_format(format_name, kwargs, windows_dataset, tmp_dir)
        results.append(result)
        print(f"  ✅ Save: {result['save_time_mean']:.2f}±{result['save_time_std']:.2f}s | "
              f"Load: {result['load_time_mean']:.2f}±{result['load_time_std']:.2f}s | "
              f"Random: {result['random_access_ms_mean']:.3f}ms/sample")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

shutil.rmtree(tmp_dir)

###############################################################################
# Results

if not results:
    print("\n❌ No benchmarks completed!")
    exit(1)

df = pd.DataFrame(results)

# Fill any NaN values in compression column with "none"
df["compression"] = df["compression"].fillna("none")

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(df.to_string(index=False))

df.to_csv("hub_formats_benchmark_results.csv", index=False)
print("\n📄 Saved to: hub_formats_benchmark_results.csv")

###############################################################################
# Visualization
print("\n📊 Generating plots...")

# Create labels (compression is now guaranteed to be a string)
df["label"] = df["format"] + "\n" + df["compression"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Storage Format Benchmark Results - Including FIF (MNE native format)\n{filtered_count} recordings ({window_size_s}s windows), 2 repetitions, {n_random_samples} random access samples",
             fontsize=16, fontweight="bold")

# 1. Save Time
ax = axes[0, 0]
ax.bar(range(len(df)), df["save_time_mean"], yerr=df["save_time_std"], capsize=5)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Time (s)")
ax.set_title("Save Time")
ax.grid(axis="y", alpha=0.3)

# 2. Load Time
ax = axes[0, 1]
ax.bar(range(len(df)), df["load_time_mean"], yerr=df["load_time_std"],
       color="orange", capsize=5)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Time (s)")
ax.set_title("Load Time (Cold Start)")
ax.grid(axis="y", alpha=0.3)

# 3. Random Access Speed (MOST IMPORTANT)
ax = axes[0, 2]
bars = ax.bar(range(len(df)), df["random_access_ms_mean"],
              yerr=df["random_access_ms_std"], color="red", alpha=0.7, capsize=5)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Time (ms/sample)")
ax.set_title("⭐ Random Access Speed\n[CRITICAL for Training]", fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Highlight best
best_idx = df["random_access_ms_mean"].idxmin()
bars[best_idx].set_color("green")

# 4. Sequential Read
ax = axes[1, 0]
ax.bar(range(len(df)), df["sequential_time"], color="purple")
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Time (s)")
ax.set_title("Sequential Read Time")
ax.grid(axis="y", alpha=0.3)

# 5. File Size
ax = axes[1, 1]
ax.bar(range(len(df)), df["file_size_mb"], color="teal")
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Size (MB)")
ax.set_title("File Size")
ax.grid(axis="y", alpha=0.3)

# 6. Overall Weighted Score
ax = axes[1, 2]
# Normalize metrics (lower is better)
norm_random = df["random_access_ms_mean"] / df["random_access_ms_mean"].max()
norm_load = df["load_time_mean"] / df["load_time_mean"].max()
norm_save = df["save_time_mean"] / df["save_time_mean"].max()
norm_size = df["file_size_mb"] / df["file_size_mb"].max()

# Weighted score - Option C: Size-optimized (40% random access, 20% load, 10% save, 30% size)
overall_score = 0.4 * norm_random + 0.2 * norm_load + 0.1 * norm_save + 0.3 * norm_size

bars = ax.bar(range(len(df)), overall_score, color="gray", alpha=0.7)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Score (lower is better)")
ax.set_title("Overall Weighted Score\n(40% random, 20% load, 10% save, 30% size)",
             fontweight="bold", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Highlight best overall
best_overall_idx = overall_score.idxmin()
bars[best_overall_idx].set_color("gold")

plt.tight_layout()
plt.savefig("hub_formats_benchmark.png", dpi=150, bbox_inches="tight")
print("📊 Plots saved to: hub_formats_benchmark.png")
plt.show()

###############################################################################
# Find best

best_random = df.loc[df["random_access_ms_mean"].idxmin()]
best_size = df.loc[df["file_size_mb"].idxmin()]

# Weighted score - Option C: Size-optimized
norm_random = df["random_access_ms_mean"] / df["random_access_ms_mean"].max()
norm_load = df["load_time_mean"] / df["load_time_mean"].max()
norm_save = df["save_time_mean"] / df["save_time_mean"].max()
norm_size = df["file_size_mb"] / df["file_size_mb"].max()

# 40% random access, 20% load, 10% save, 30% file size
overall_score = 0.4 * norm_random + 0.2 * norm_load + 0.1 * norm_save + 0.3 * norm_size

# Add scores to results BEFORE selecting best
df["overall_score"] = overall_score

# Now select best (after score column is added)
best_overall = df.loc[overall_score.idxmin()]

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"\n🏆 Best Random Access (most important for training):")
print(f"   {best_random['format']} ({best_random['compression']})")
print(f"   {best_random['random_access_ms_mean']:.3f}±{best_random['random_access_ms_std']:.3f} ms/sample")

print(f"\n💾 Best File Size:")
print(f"   {best_size['format']} ({best_size['compression']})")
print(f"   {best_size['file_size_mb']:.2f} MB")

print(f"\n⭐ Best Overall (Size-Optimized Score):")
print(f"   Weights: 40% random, 20% load, 10% save, 30% size")
print(f"   Winner: {best_overall['format']} ({best_overall['compression']})")
print(f"   Score: {best_overall['overall_score']:.3f} (lower is better)")
print(f"   Random access: {best_overall['random_access_ms_mean']:.3f} ms/sample")
print(f"   File size: {best_overall['file_size_mb']:.2f} MB")

print("\n" + "="*70)

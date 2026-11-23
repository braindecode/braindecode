""".. _benchmark-eager-lazy:

Benchmarking eager and lazy loading
======================================

In this example, we compare the execution time and memory requirements of 1)
eager loading, i.e., preloading the entire data into memory and 2) lazy loading,
i.e., only loading examples from disk when they are required. We also include
some other experiment parameters in the comparison for the sake of completeness
(e.g., ``num_workers``, ``cuda``, ``batch_size``, etc.).

While eager loading might be required for some preprocessing steps that require
continuous data (e.g., temporal filtering, resampling), it also allows
fast access to the data during training. However, this might come at the expense
of large memory usage, and can ultimately become impossible if the dataset does
not fit into memory (e.g., the TUH EEG dataset's >1,5 TB of recordings will
not fit in the memory of most machines).

Lazy loading avoids this potential memory issue by loading examples from disk
when they are required. This means large datasets can be used for training,
however this introduces some file-reading overhead every time an example must
be extracted. Some preprocessing steps that require continuous data also have to
be implemented differently to accommodate the nature of windowed data. Overall
though, we can reduce the impact of lazy loading by using the `num_workers`
parameter of pytorch's `Dataloader` class, which dispatches the data loading to
multiple processes.

To enable lazy loading in braindecode, data files must be saved in an
MNE-compatible format (e.g., 'fif', 'edf', etc.), and the `Dataset` object must
have been instantiated with parameter `preload=False`.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import time
from itertools import product

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from braindecode.datasets import TUHAbnormal
from braindecode.models import Deep4Net, ShallowFBCSPNet
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.util import set_random_seeds

mne.set_log_level("WARNING")  # avoid messages every time a window is extracted

###############################################################################
# We start by setting two pytorch internal parameters that can affect the
# comparison:

N_JOBS = 8
torch.backends.cudnn.benchmark = True  # Enables automatic algorithm optimizations
torch.set_num_threads(N_JOBS)  # Sets the available number of threads
cuda = torch.cuda.is_available()
seed = 20240205
set_random_seeds(seed=seed, cuda=cuda)


###############################################################################
# Next, we define a few functions to automate the benchmarking.
# For the purpose of this example, we load some recordings from the TUH Abnormal
# corpus, extract sliding windows, and bundle them in a braindecode Dataset.
# We then train a neural network for a few epochs.
#
# Each one of these steps will be timed, so we can report the total time taken
# to prepare the data and train the model.


def load_example_data(preload, window_len_s, n_recordings=10):
    """Create windowed dataset from subjects of the TUH Abnormal dataset.

    Parameters
    ----------
    preload: bool
        If True, use eager loading, otherwise use lazy loading.
    window_len_s: int
        Window length in seconds.
    n_recordings: list of int
        Number of recordings to load.

    Returns
    -------
    windows_ds: BaseConcatDataset
        Windowed data.

    .. warning::
        The recordings from the TUH Abnormal corpus do not all share the same
        sampling rate. The following assumes that the files have already been
        resampled to a common sampling rate.

    """
    recording_ids = list(range(n_recordings))

    ds = TUHAbnormal(
        TUH_PATH,
        recording_ids=recording_ids,
        target_name="pathological",
        preload=preload,
    )

    fs = ds.datasets[0].raw.info["sfreq"]
    window_len_samples = int(fs * window_len_s)
    window_stride_samples = int(fs * 4)
    # window_stride_samples = int(fs * window_len_s)
    windows_ds = create_fixed_length_windows(
        ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_len_samples,
        window_stride_samples=window_stride_samples,
        on_last_window="drop",
        preload=preload,
    )

    # Drop bad epochs
    # XXX: This could be parallelized.
    # XXX: Also, this could be implemented in the Dataset object itself.
    # We don't support drop_bad since the last version braindecode,
    # to optimize the dataset speed. If you know how to fix, please open a PR.
    # for ds in windows_ds.datasets:
    #    ds.raw.drop_bad()
    #   assert ds.raw.preload == preload

    return windows_ds


def create_example_model(
    n_channels, n_classes, window_len_samples, kind="shallow", cuda=False
):
    """Create model, loss and optimizer.

    Parameters
    ----------
    n_channels : int
        Number of channels in the input
    n_times : int
        Window length in the input
    n_classes : int
        Number of classes in the output
    kind : str
        'shallow' or 'deep'
    cuda : bool
        If True, move the model to a CUDA device.

    Returns
    -------
    model : torch.nn.Module
        Model to train.
    loss :
        Loss function
    optimizer :
        Optimizer

    """
    if kind == "shallow":
        model = ShallowFBCSPNet(
            n_channels,
            n_classes,
            n_times=window_len_samples,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            final_conv_length="auto",
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
        )
    elif kind == "deep":
        model = Deep4Net(
            n_channels,
            n_classes,
            n_times=window_len_samples,
            final_conv_length="auto",
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=50,
            filter_length_2=10,
            n_filters_3=100,
            filter_length_3=10,
            n_filters_4=200,
            filter_length_4=10,
            first_pool_mode="max",
            later_pool_mode="max",
            drop_prob=0.5,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False,
        )
    else:
        raise ValueError

    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    return model, loss, optimizer


def run_training(model, dataloader, loss, optimizer, n_epochs=1, cuda=False):
    """Run training loop.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    dataloader : torch.utils.data.Dataloader
        Data loader which will serve examples to the model during training.
    loss :
        Loss function.
    optimizer :
        Optimizer.
    n_epochs : int
        Number of epochs to train the model for.
    cuda : bool
        If True, move X and y to CUDA device.

    Returns
    -------
    model : torch.nn.Module
        Trained model.

    """
    for i in range(n_epochs):
        loss_vals = list()
        for X, y, _ in dataloader:
            model.train()
            model.zero_grad()

            y = y.long()
            if cuda:
                X, y = X.cuda(), y.cuda()

            loss_val = loss(model(X), y)
            loss_vals.append(loss_val.item())

            loss_val.backward()
            optimizer.step()

        print(f"Epoch {i + 1} - mean training loss: {np.mean(loss_vals)}")

    return model


###############################################################################
# Next, we define the different hyperparameters that we want to compare:

PRELOAD = [True, False]  # True -> eager loading; False -> lazy loading
N_RECORDINGS = [10]  # Number of recordings to load from the TUH Abnormal corpus
WINDOW_LEN_S = [2, 4, 15]  # Window length, in seconds
N_EPOCHS = [2]  # Number of epochs to train the model for
BATCH_SIZE = [64, 256]  # Training minibatch size
MODEL = ["shallow", "deep"]

NUM_WORKERS = [8, 0]  # number of processes used by pytorch's Dataloader
PIN_MEMORY = [False]  # whether to use pinned memory
CUDA = (
    [True, False] if torch.cuda.is_available() else [False]
)  # whether to use a CUDA device

N_REPETITIONS = (
    3  # Number of times to repeat the experiment (to get better time estimates)
)

###############################################################################
# The following path needs to be changed to your local folder containing the
# TUH Abnormal corpus:
TUH_PATH = (
    "/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/"
    "tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/"
)

###############################################################################
# We can finally cycle through all the different combinations of the parameters
# we set above to evaluate their execution time:

all_results = list()
for (
    i,
    preload,
    n_recordings,
    win_len_s,
    n_epochs,
    batch_size,
    model_kind,
    num_workers,
    pin_memory,
    cuda,
) in product(
    range(N_REPETITIONS),
    PRELOAD,
    N_RECORDINGS,
    WINDOW_LEN_S,
    N_EPOCHS,
    BATCH_SIZE,
    MODEL,
    NUM_WORKERS,
    PIN_MEMORY,
    CUDA,
):
    results = {
        "repetition": i,
        "preload": preload,
        "n_recordings": n_recordings,
        "win_len_s": win_len_s,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "model_kind": model_kind,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "cuda": cuda,
    }
    print(f"\nRepetition {i + 1}/{N_REPETITIONS}:\n{results}")

    # Load the dataset
    data_loading_start = time.time()
    dataset = load_example_data(preload, win_len_s, n_recordings=n_recordings)
    data_loading_end = time.time()

    # Create the data loader
    training_setup_start = time.time()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=None,
    )
    # Instantiate model and optimizer
    n_channels = dataset[0][0].shape[0]
    n_times = dataset[0][0].shape[1]
    n_classes = 2
    model, loss, optimizer = create_example_model(
        n_channels, n_classes, n_times, kind=model_kind, cuda=cuda
    )
    training_setup_end = time.time()

    # Start training loop
    model_training_start = time.time()
    trained_model = run_training(
        model, dataloader, loss, optimizer, n_epochs=n_epochs, cuda=cuda
    )
    model_training_end = time.time()

    del dataset, model, loss, optimizer, trained_model

    # Record timing results
    results["data_preparation"] = data_loading_end - data_loading_start
    results["training_setup"] = training_setup_end - training_setup_start
    results["model_training"] = model_training_end - model_training_start
    all_results.append(results)

###############################################################################
# The results are formatted into a pandas DataFrame and saved locally as a CSV
# file.

results_df = pd.DataFrame(all_results)
fname = "lazy_vs_eager_loading_results.csv"
results_df.to_csv(fname)
print(f"Results saved under {fname}.")

###############################################################################
# We can finally summarize this information into the following plot. Each
# panel is one loading mode (and device, when CUDA is available); small dots
# are single runs and large dots are their mean.

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)
worker_colors = {0: "#0072B2", 8: "#D55E00"}  # keys follow NUM_WORKERS
model_names = {"shallow": "ShallowFBCSPNet", "deep": "Deep4Net"}

devices = sorted(results_df["cuda"].unique(), reverse=True)
fig, axes = plt.subplots(
    len(devices),
    2,
    figsize=(9, 2.6 * len(devices)),
    sharex=True,
    squeeze=False,
)
for i_row, cuda_used in enumerate(devices):
    for i_col, preload in enumerate([True, False]):
        ax = axes[i_row, i_col]
        panel = results_df[
            (results_df["cuda"] == cuda_used) & (results_df["preload"] == preload)
        ]
        for i_model, model_kind in enumerate(MODEL):
            for i_w, num_workers in enumerate(NUM_WORKERS):
                times = panel.loc[
                    (panel["model_kind"] == model_kind)
                    & (panel["num_workers"] == num_workers),
                    "model_training",
                ]
                y = i_model + (i_w - 0.5) * 0.3  # one row per worker count
                color = worker_colors[num_workers]
                ax.scatter(times, np.full(len(times), y), color=color, s=14, alpha=0.4)
                ax.scatter(times.mean(), y, color=color, s=60, zorder=3)
                if i_row == 0 and i_col == 0 and i_model == 0:
                    ax.annotate(
                        f"{num_workers} workers",
                        (times.max(), y),
                        xytext=(8, 0),
                        textcoords="offset points",
                        color=color,
                        va="center",
                        fontsize=9,
                    )
        ax.set_yticks(range(len(MODEL)), [model_names[m] for m in MODEL])
        ax.set_ylim(len(MODEL) - 0.4, -0.6)
        ax.grid(axis="x", color="#EAEAEA", linewidth=0.7)
        ax.set_axisbelow(True)
        letter = "ABCD"[i_row * 2 + i_col]
        title = "Eager loading" if preload else "Lazy loading"
        if len(devices) > 1:
            title += ", GPU" if cuda_used else ", CPU"
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
for ax in axes[-1]:
    ax.set_xlabel("Training time (s)")
ax.set_xlim(left=0)
fig.tight_layout()
plt.show()

###############################################################################
# .. warning::
#   The results of this comparison will change depending on the hyperparameters
#   that were set above, and on the actual hardware that is being used.

###############################################################################
# Generally speaking, we expect lazy loading to be slower than eager loading
# during model training, but to potentially be pretty competitive if multiple
# workers were enabled (i.e.., `num_workers > 0`). Training on a CUDA device
# should also yield substantial speedups.

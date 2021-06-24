'''Compare speeds of two equivalent operations, for doing Temporal Convolutions.'''

import numpy as np
from scipy.stats import iqr
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt


def set_torch_seed(seed):
    """Set torch's seed."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_times(batch_size=64,
              in_channels=2, out_channels=2,
              n_times=3000,
              kernel_size=9,
              seed=0,
              verbose=False):
    # Fix seed
    set_torch_seed(seed)

    padding = (kernel_size - 1) // 2  # preserves dim
    conv_weights = torch.randn(out_channels, in_channels,
                               kernel_size)

    # Create input
    x = torch.Tensor(batch_size, in_channels, n_times)  # shape (B, C, T)
    x_augment = torch.unsqueeze(x, dim=2)               # shape (B, C, 1, T)

    # Option 1 : Temporal Convolution via Conv1d (last axis)
    layer_1d = nn.Conv1d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding)
    layer_1d.weight.data = conv_weights

    # Option 2 : Temporal Convolution via Conv2d (dummy axis trick)
    layer_2d = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=(1, kernel_size),
                         padding=(0, padding))
    layer_2d.weight.data[:, :, 0, :] = conv_weights

    # Time both operations
    start = time.time()
    y1 = layer_1d(x)
    end = time.time()
    duration_1d = end - start

    start = time.time()
    y_augment = layer_2d(x_augment)
    end = time.time()
    duration_2d = end - start
    y2 = y_augment[:, :, 0, :]

    # Verify match
    if verbose:
        print(f"y1 shape: {y1.shape}, y2 shape: {y2.shape}")
        maxerror = torch.abs(y1 - y2).max()
        print(f"Maximum error between y1 and y2: {maxerror}")

    return duration_1d, duration_2d


# Experiment 1 : vary seed
t1s, t2s = list(), list()
t1s_error, t2s_error = list(), list()

X = np.arange(10, 150 + 1, 10)
total_seeds = 400
for kernel_size in X.astype(int):
    print(f"Doing kernel_size {kernel_size}...")

    t1s_temp, t2s_temp = list(), list()
    for seed in range(total_seeds):
        t1, t2 = get_times(seed=seed, kernel_size=kernel_size)
        t1s_temp.append(t1)
        t2s_temp.append(t2)
    t1, t2 = np.median(t1s_temp), np.median(t2s_temp)
    t1_error, t2_error = iqr(t1s_temp)/2, iqr(t2s_temp)/2,

    t1s.append(t1)
    t2s.append(t2)
    t1s_error.append(t1_error)
    t2s_error.append(t2_error)

fig, ax = plt.subplots()
ax.errorbar(X, t1s, t1s_error, label='Conv1d', color='blue')
ax.errorbar(X, t2s, t2s_error, label='Conv2d', color='red')
ax.set(title='Comparing speed for Temporal Convolutions',
       xlabel='Kernel size',
       ylabel='Speed (s)')
ax.legend()
fig.tight_layout()
plt.show()

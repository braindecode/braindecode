"""
Trialwise Decoding on BCIC IV 2a Competition Set with skorch and moabb.
=======================================================================
"""

# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD-3
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import mne
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skorch.callbacks import LRScheduler
mne.set_log_level('ERROR')

from braindecode import EEGClassifier
from braindecode.datautil import create_windows_from_events
from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.transforms import transform_concat_ds

subject_id = 3  # 1-9
model_name = "shallow"  # 'shallow' or 'deep'
low_cut_hz = 4.  # 0 or 4
n_epochs = 5
seed = 20200220

high_cut_hz = 38.
trial_start_offset_seconds = -0.5
input_time_length = 1125
batch_size = 64
factor_new = 1e-3
init_block_size = 1000
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

n_classes = 4
n_chans = 22

set_random_seeds(seed=seed, cuda=cuda)

if model_name == "shallow":
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length='auto',
    )
    lr = 0.0625 * 0.01
    weight_decay = 0

elif model_name == "deep":
    model = Deep4Net(
        n_chans,
        n_classes,
        input_time_length=input_time_length,
        final_conv_length='auto',
    )
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

if cuda:
    model.cuda()

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

standardize_func = partial(
    exponential_running_standardize, factor_new=factor_new,
    init_block_size=init_block_size)
raw_transform_dict = OrderedDict([
    ("pick_types", dict(eeg=True, meg=False, stim=False)),
    ('apply_function', dict(fun=lambda x: x * 1e6, channel_wise=False)),
    ('filter', dict(l_freq=low_cut_hz, h_freq=high_cut_hz)),
    ('apply_function', dict(fun=standardize_func, channel_wise=False))
])
transform_concat_ds(dataset, raw_transform_dict)

sfreqs = [ds.raw.info['sfreq'] for ds in dataset.datasets]
assert len(np.unique(sfreqs)) == 1
trial_start_offset_samples = int(trial_start_offset_seconds * sfreqs[0])

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    supercrop_size_samples=input_time_length,
    supercrop_stride_samples=input_time_length,
    drop_samples=False,
    preload=True,
)


class TrainTestBCICIV2aSplit(object):
    def __call__(self, dataset, y, **kwargs):
        splitted = dataset.split('session')
        return splitted['session_T'], splitted['session_E']


clf = EEGClassifier(
    model,
    cropped=False,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=TrainTestBCICIV2aSplit(),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)

clf.fit(windows_dataset, y=None, epochs=n_epochs)

###############################################################################
# Plot Results

ignore_keys = [
        'batches', 'train_batch_count', 'valid_batch_count',
        'train_loss_best',
        'valid_loss_best', 'train_accuracy_best',
        'valid_accuracy_best', 'dur']
results = [dict([(key, val) for key, val in hist_dict.items() if
                key not in ignore_keys])
           for hist_dict in clf.history]

df = pd.DataFrame(results).set_index('epoch')
# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
         valid_misclass=100 - 100 * df.valid_accuracy)

plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8,3))

df.loc[:,['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-',':'], marker='o',
    color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:,['train_misclass', 'valid_misclass']].plot(
    ax=ax2, style=['-',':'], marker='o',
    color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0],85) # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles,[h.get_label() for h in handles], fontsize=14,)

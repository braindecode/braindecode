"""
Regression example on fake data
===============================
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import numpy as np
import pandas as pd
import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGRegressor
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.training.losses import CroppedLoss
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.util import set_random_seeds, create_mne_dummy_raw

model_name = "shallow"  # 'shallow' or 'deep'
n_epochs = 3
seed = 20200220

input_window_samples = 6000
batch_size = 64
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

n_chans = 21
# set to how many targets you want to regress (age -> 1, [x, y, z] -> 3)
n_classes = 1

set_random_seeds(seed=seed, cuda=cuda)

# initialize a model, transform to dense and move to gpu
if model_name == "shallow":
    model = ShallowFBCSPNet(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        n_filters_time=40,
        n_filters_spat=40,
        final_conv_length=35,
    )
    optimizer_lr = 0.000625
    optimizer_weight_decay = 0
elif model_name == "deep":
    model = Deep4Net(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        n_filters_time=25,
        n_filters_spat=25,
        stride_before_pool=True,
        n_filters_2=int(n_chans * 2),
        n_filters_3=int(n_chans * (2 ** 2.0)),
        n_filters_4=int(n_chans * (2 ** 3.0)),
        final_conv_length=1,
    )
    optimizer_lr = 0.01
    optimizer_weight_decay = 0.0005
else:
    raise ValueError(f'{model_name} unknown')

new_model = torch.nn.Sequential()
for name, module_ in model.named_children():
    if "softmax" in name:
        continue
    new_model.add_module(name, module_)
model = new_model

if cuda:
    model.cuda()

to_dense_prediction_model(model)
n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


def fake_regression_dataset(n_fake_recs, n_fake_chs, fake_sfreq, fake_duration_s):
    datasets = []
    for i in range(n_fake_recs):
        train_or_eval = "eval" if i == 0 else "train"
        raw, save_fname = create_mne_dummy_raw(
            n_channels=n_fake_chs, n_times=fake_duration_s * fake_sfreq,
            sfreq=fake_sfreq, savedir=None)
        target = np.random.randint(0, 100, n_classes)
        if n_classes == 1:
            target = target[0]
        fake_descrition = pd.Series(
            data=[target, train_or_eval],
            index=["target", "session"])
        base_ds = BaseDataset(raw, fake_descrition, target_name="target")
        datasets.append(base_ds)
    dataset = BaseConcatDataset(datasets)
    return dataset


dataset = fake_regression_dataset(
    n_fake_recs=5, n_fake_chs=21, fake_sfreq=100, fake_duration_s=60)

windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    drop_bad_windows=True,
)

splits = windows_dataset.split("session")
train_set = splits["train"]
valid_set = splits["eval"]

regressor = EEGRegressor(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=optimizer_lr,
    optimizer__weight_decay=optimizer_weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "neg_root_mean_squared_error",
        # seems n_epochs -1 leads to desired behavior of lr=0 after end of training?
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)

regressor.fit(train_set, y=None, epochs=n_epochs)

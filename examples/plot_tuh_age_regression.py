"""
Age regression on TUH Abnormal EEG Dataset
==========================================
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import numpy as np
import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGRegressor
from braindecode.datautil import create_fixed_length_windows
from braindecode.datasets import TUHAbnormal
from braindecode.losses import CroppedLoss
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.util import set_random_seeds
from braindecode.datautil.transforms import transform_concat_ds

model_name = "shallow"  # 'shallow' or 'deep'
n_epochs = 5
seed = 20200220

input_time_length = 6000
batch_size = 64
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

n_chans = 21
n_classes = 1

set_random_seeds(seed=seed, cuda=cuda)

# initialize a model, transform to dense and move to gpu
if model_name == "shallow":
    model = ShallowFBCSPNet(
        in_chans=n_chans,
        n_classes=n_classes,
        input_time_length=input_time_length,
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
        input_time_length=input_time_length,
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
n_preds_per_input = get_output_shape(model, n_chans, input_time_length)[2]

dataset = TUHAbnormal(
    path="/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/",
    subject_ids=np.arange(100),
    target_name="age")

sfreq = 100
tmin = 60 * 1
tmax = 60 * 11
factor = 1e6
clipping_value = 800
ch_names = sorted([
    'EEG FP2-REF', 'EEG FP1-REF', 'EEG F4-REF', 'EEG F3-REF', 'EEG C4-REF',
    'EEG C3-REF', 'EEG P4-REF', 'EEG P3-REF', 'EEG O2-REF', 'EEG O1-REF',
    'EEG F8-REF', 'EEG F7-REF', 'EEG T4-REF', 'EEG T3-REF', 'EEG T6-REF',
    'EEG T5-REF', 'EEG A2-REF', 'EEG A1-REF', 'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF'])


def clip(data, clipping_value):
    return np.clip(data, -clipping_value, clipping_value)


raw_transform_dict = [
    ('pick_channels', dict(ch_names=ch_names)),
    ('reorder_channels', dict(ch_names=ch_names)),
    ('apply_function', dict(fun=lambda x: x * factor, channel_wise=False)),
    ('crop', dict(tmin=tmin, tmax=tmax, include_tmax=False)),
    ('resample', dict(sfreq=sfreq)),
    ('apply_function', dict(fun=clip, clipping_value=clipping_value))
]
transform_concat_ds(dataset, raw_transform_dict)

windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    supercrop_size_samples=input_time_length,
    supercrop_stride_samples=n_preds_per_input,
    drop_samples=False,
    drop_bad_windows=True,
)

split_i = int(.8 * len(windows_dataset.datasets))
splitted = windows_dataset.split(split_ids=[
    np.arange(0, split_i),
    np.arange(split_i, len(windows_dataset.datasets))
])
train_set = splitted[0]
valid_set = splitted[1]

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

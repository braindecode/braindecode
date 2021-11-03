# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Tibor Schirrmeister <robintibor@gmail.com>
#
# License: BSD-3

import torch
import numpy as np

from skorch.helper import predefined_split

from braindecode.datasets.tuh import _TUHAbnormalMock
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datasets import BaseConcatDataset
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.training import CroppedLoss


def test_variable_length_trials_cropped_decoding():
    cuda = False
    set_random_seeds(seed=20210726, cuda=cuda)

    # create fake tuh abnormal dataset
    tuh = _TUHAbnormalMock(path='')
    # fake variable length trials by cropping first recording
    splits = tuh.split([[i] for i in range(len(tuh.datasets))])
    preprocess(
        concat_ds=splits['0'],
        preprocessors=[
            Preprocessor('crop', tmax=300),
        ],
    )
    variable_tuh = BaseConcatDataset(
        [splits[str(i)] for i in range(len(tuh.datasets))])
    # make sure we actually have different length trials
    assert any(np.diff([ds.raw.n_times for ds in variable_tuh.datasets]) != 0)

    # create windows
    variable_tuh_windows = create_fixed_length_windows(
        concat_ds=variable_tuh,
        window_size_samples=1000,
        window_stride_samples=1000,
        drop_last_window=False,
        mapping={True: 1, False: 0},
    )

    # create train and valid set
    splits = variable_tuh_windows.split(
        [[i] for i in range(len(variable_tuh_windows.datasets))])
    variable_tuh_windows_train = BaseConcatDataset(
        [splits[str(i)] for i in range(len(tuh.datasets)-1)])
    variable_tuh_windows_valid = BaseConcatDataset(
        [splits[str(len(tuh.datasets)-1)]])
    for x, y, ind in variable_tuh_windows_train:
        break
    train_split = predefined_split(variable_tuh_windows_valid)

    # initialize a model
    model = ShallowFBCSPNet(
        in_chans=x.shape[0],
        n_classes=len(tuh.description.pathological.unique()),
    )
    to_dense_prediction_model(model)
    if cuda:
        model.cuda()

    # create and train a classifier
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.Adam,
        batch_size=32,
        callbacks=['accuracy'],
        train_split=train_split,
    )
    clf.fit(variable_tuh_windows_train, y=None, epochs=3)

    # make sure it does what we expect
    np.testing.assert_allclose(
        clf.history[:, 'train_loss'],
        np.array([
            0.689495325088501,
            0.1353449523448944,
            0.006638816092163324,
        ]
        ),
        rtol=1e-1,
        atol=1e-1,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_loss'],
        np.array([
            2.925871,
            3.611423,
            4.23494,
        ]
        ),
        rtol=1e-1,
        atol=1e-1,
    )

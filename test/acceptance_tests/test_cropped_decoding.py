# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Simon Freyburger
#
# License: BSD-3

import numpy as np
import torch
from skorch.helper import predefined_split
from torch import optim

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
from ..get_dummy_sample import get_dummy_train_valid_and_model


def test_cropped_decoding():
    train_set, valid_set, model = get_dummy_train_valid_and_model()
    train_split = predefined_split(valid_set)

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=optim.Adam,
        train_split=train_split,
        batch_size=32,
        callbacks=['accuracy'],
    )

    clf.fit(train_set, y=None, epochs=4)

    np.testing.assert_allclose(
        clf.history[:, 'train_loss'],
        np.array(
            [
                1.6666231592496237,
                1.2292670885721841,
                1.1270817518234253,
                1.1752660751342774
            ]
        ),
        rtol=1e-3,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_loss'],
        np.array(
            [
                1.5687058925628663,
                0.8510023872057597,
                2.087181798617045,
                0.7100235184033712
            ]
        ),
        rtol=1e-3,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        clf.history[:, 'train_accuracy'],
        np.array(
            [
                0.48333333333333334,
                0.5,
                0.5,
                0.6333333333333333
            ]
        ),
        rtol=1e-3,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        clf.history[:, 'valid_accuracy'],
        np.array(
            [
                0.533333,
                0.5,
                0.466667,
                0.666667
            ]
        ),
        rtol=1e-3,
        atol=1e-4,
    )

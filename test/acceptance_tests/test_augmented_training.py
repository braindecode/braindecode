# Authors: Simon Freyburger
#          Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3

import mne
import numpy as np
import torch
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch import optim

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
from braindecode.augment import \
    mask_along_frequency, mask_along_time, Transform
from braindecode.datasets.base import AugmentedDataset
from ..get_dummy_sample import get_dummy_train_valid_and_model


def test_augmented_decoding():
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

    subpolicies_list = [
        Transform(lambda datum:datum),
        Transform(mask_along_frequency, magnitude=0.2),
        Transform(mask_along_time, magnitude=0.2)]

    train_set = AugmentedDataset(train_set, subpolicies_list)
    clf.fit(train_set, y=None, epochs=4)
    print("train_loss : ", clf.history[:, 'train_loss'])
    print("valid_loss : ", clf.history[:, 'valid_loss'])
    print("train_accuracy : ", clf.history[:, 'train_accuracy'])
    print("valid_accuracy : ", clf.history[:, 'valid_accuracy'])
    # np.testing.assert_allclose(
    #     clf.history[:, 'train_loss'],
    #     np.array(
    #         [
    #             1.455306,
    #             1.784507,
    #             1.421611,
    #             1.057717
    #         ]
    #     ),
    #     rtol=1e-3,
    #     atol=1e-4,
    # )
    # np.testing.assert_allclose(
    #     clf.history[:, 'valid_loss'],
    #     np.array(
    #         [
    #             2.547288,
    #             3.051576,
    #             0.711256,
    #             0.839392
    #         ]
    #     ),
    #     rtol=1e-3,
    #     atol=1e-3,
    # )
    # np.testing.assert_allclose(
    #     clf.history[:, 'train_accuracy'],
    #     np.array(
    #         [
    #             0.5,
    #             0.5,
    #             0.6,
    #             0.516667
    #         ]
    #     ),
    #     rtol=1e-3,
    #     atol=1e-4,
    # )
    # np.testing.assert_allclose(
    #     clf.history[:, 'valid_accuracy'],
    #     np.array(
    #         [
    #             0.533333,
    #             0.466667,
    #             0.466667,
    #             0.5
    #         ]
    #     ),
    #     rtol=1e-3,
    #     atol=1e-4,
    # )

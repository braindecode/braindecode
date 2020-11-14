# Authors: Simon Freyburger
#          Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3


from skorch.helper import predefined_split
from torch import optim
import torch
from functools import partial
import mne
from torch.nn.modules.loss import CrossEntropyLoss
from braindecode import EEGClassifier
from braindecode.augment import \
    mask_along_frequency, mask_along_time, Transform, \
    merge_two_signals, MERGE_TWO_SIGNALS_REQUIRED_VARIABLES, \
    mixup_beta, MIXUP_BETA_REQUIRED_VARIABLES, \
    general_mixup_criterion
from braindecode.datasets.base import AugmentedDataset
import numpy as np
from mne.io import concatenate_raws
from braindecode.datautil.xy import create_from_X_y
from braindecode.models import SleepStagerChambon2018
from braindecode.util import set_random_seeds
mne.set_log_level("warning")
# TODO : ordre des imports


def test_augmented_decoding():

    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False
    )

    # Load each of the files
    parts = [
        mne.io.read_raw_edf(
            path, preload=True, stim_channel="auto", verbose="WARNING"
        )
        for path in physionet_paths
    ]

    # Concatenate them
    raw = concatenate_raws(parts)

    # Find the events in this dataset
    events, _ = mne.events_from_annotations(raw)
    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(
        raw,
        events,
        dict(hands=2, feet=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True,
    )
    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1

    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is
    # available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    input_window_samples = 450
    n_classes = 2
    in_chans = X.shape[1]
    # final_conv_length determines the size of the receptive field of the
    # ConvNet

    model = SleepStagerChambon2018(
        n_channels=in_chans,
        sfreq=50,
        n_classes=n_classes,
        input_size_s=input_window_samples / 50
    )

    if cuda:
        model.cuda()

    # Perform forward pass to determine how many outputs per input

    train_set = create_from_X_y(X[:60], y[:60],
                                drop_last_window=False,
                                window_size_samples=input_window_samples,
                                window_stride_samples=1)

    valid_set = create_from_X_y(X[60:], y[60:],
                                drop_last_window=False,
                                window_size_samples=input_window_samples,
                                window_stride_samples=1)

    train_split = predefined_split(valid_set)

    lr = 0.0625 * 0.01
    weight_decay = 0
    clf = EEGClassifier(
        model,
        criterion=general_mixup_criterion,
        criterion__loss=CrossEntropyLoss,
        iterator_train__collate_fn=lambda x: x,
        optimizer=optim.Adam,
        train_split=train_split,
        batch_size=None,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        callbacks=['accuracy'],
    )

    FFT_ARGS = {"n_fft": 128, "hop_length": 64,
                "win_length": 128}

    DATA_SIZE = 450

    subpolicies_list = [
        Transform(lambda datum:datum),
        Transform(partial(mask_along_frequency,
                          fft_args=FFT_ARGS,
                          data_size=DATA_SIZE),
                  params={"magnitude": 0.2}),
        Transform(partial(mask_along_time,
                          fft_args=FFT_ARGS,
                          data_size=DATA_SIZE),
                  params={"magnitude": 0.2}),
        Transform(merge_two_signals,
                  params={"magnitude": 0.2},
                  required_variables=MERGE_TWO_SIGNALS_REQUIRED_VARIABLES),
        Transform(mixup_beta,
                  params={"alpha": 0.3,
                          "beta_per_sample": False},
                  required_variables=MIXUP_BETA_REQUIRED_VARIABLES)]

    sub_train_set = torch.utils.data.Subset(train_set, indices=[1, 67])
    aug_train_set = AugmentedDataset(sub_train_set, subpolicies_list)
    clf.fit(aug_train_set, y=None, epochs=4)
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


if __name__ == "__main__":
    test_augmented_decoding()

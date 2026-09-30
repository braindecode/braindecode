# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3
import mne
import numpy as np
import pytest
import torch
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch.utils.data import Dataset, Subset

from braindecode.classifier import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from test.acceptance_tests._history_assertions import assert_learning_history


class EpochsDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
        self.y = np.array(self.windows.events[:, -1])
        self.y = self.y - self.y.min()

    def __getitem__(self, index):
        X = self.windows.get_data(item=index)[0].astype("float32")[:, :, None]
        y = self.y[index]
        return X, y

    def __len__(self):
        return len(self.windows.events)


@pytest.mark.network
def test_trialwise_decoding():
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False
    )

    # Load each of the files
    parts = [
        mne.io.read_raw_edf(path, preload=True, stim_channel="auto", verbose="WARNING")
        for path in physionet_paths
    ]

    # Concatenate them
    raw = concatenate_raws(parts)
    raw.apply_function(lambda x: x * 1000000)

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

    ds = EpochsDataset(epoched)

    train_set = Subset(ds, np.arange(60))
    valid_set = Subset(ds, np.arange(60, len(ds)))

    train_valid_split = predefined_split(valid_set)

    cuda = False
    if cuda:
        device = "cuda"
    else:
        device = "cpu"
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    in_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]
    model = ShallowFBCSPNet(
        n_chans=in_chans,
        n_outputs=n_classes,
        n_times=input_window_samples,
        final_conv_length="auto",
    )
    if cuda:
        model.cuda()

    clf = EEGClassifier(
        model,
        cropped=False,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=train_valid_split,
        optimizer__lr=0.001,
        batch_size=30,
        callbacks=["accuracy"],
        device=device,
        classes=[0, 1],
    )
    clf.fit(train_set, y=None, epochs=6)

    # This guards the training pipeline (labels, loss, optimizer, predict and
    # scoring), not generalization. With 30 validation trials (one trial = 3.3%)
    # and 6 epochs, valid accuracy is typically within a few trials of chance
    # and the valid accuracy/loss first-vs-last checks fail for ~1 in 4 seeds,
    # so a torch/BLAS/platform change flips them even though training is
    # deterministic.
    assert_learning_history(
        clf.history,
        n_epochs=6,
        loss_keys=("train_loss",),
        accuracy_keys=("train_accuracy", "valid_accuracy"),
        improving_accuracy_keys=("train_accuracy",),
    )
    train_loss = np.asarray(clf.history[:, "train_loss"], dtype=float)
    valid_loss = np.asarray(clf.history[:, "valid_loss"], dtype=float)
    assert np.all(np.isfinite(valid_loss))
    # Best/first train-loss ratio: worst of 160 seeds was 0.651.
    assert train_loss.min() < 0.8 * train_loss[0]
    # train_accuracy is scored on the full train set in eval mode after each
    # epoch; >= 39/60 trials, the worst of 160 seeds was 43/60.
    assert clf.history[-1, "train_accuracy"] >= 0.65

    # The last history scores use the final weights on the train set and the
    # predefined valid split, so they must match predicting both directly.
    y_train = ds.y[train_set.indices]
    train_acc = np.mean(clf.predict(train_set) == y_train)
    np.testing.assert_allclose(train_acc, clf.history[-1, "train_accuracy"])
    y_valid = ds.y[valid_set.indices]
    valid_acc = np.mean(clf.predict(valid_set) == y_valid)
    np.testing.assert_allclose(valid_acc, clf.history[-1, "valid_accuracy"])

# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#
# License: BSD-3


import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import f1_score, accuracy_score
from skorch.callbacks import Callback
from torch import optim
from torch.utils.data import Dataset

from braindecode.experiments.classifier import BraindecodeClassifier
from braindecode.experiments.scoring import PostEpochTrainScoring
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds


def test_post_epoch_train_scoring():
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    n_classes = 2

    class EEGDataSet(Dataset):
        def __init__(self, X, y):
            self.X = X
            if self.X.ndim == 3:
                self.X = self.X[:, :, :, None]
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    X, y = sklearn.datasets.make_classification(
        40, (3 * 100), n_informative=3 * 50, n_classes=2
    )
    X = X.reshape(40, 3, 100).astype(np.float32)

    in_chans = X.shape[1]

    train_set = EEGDataSet(X, y)

    class TestCallback(Callback):
        def on_epoch_end(self, net, *args, **kwargs):
            preds = net.predict(train_set.X)
            y_true = train_set.y
            np.testing.assert_allclose(
                clf.history[-1]["train_f1"],
                f1_score(y_true, preds),
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                clf.history[-1]["train_acc"],
                accuracy_score(y_true, preds),
                rtol=1e-4,
                atol=1e-4,
            )

    set_random_seeds(20200114, cuda)

    # final_conv_length = auto ensures
    # we only get a single output in the time dimension
    model = ShallowFBCSPNet(
        in_chans=in_chans,
        n_classes=n_classes,
        input_time_length=train_set.X.shape[2],
        pool_time_stride=1,
        pool_time_length=2,
        final_conv_length="auto",
    ).create_network()
    if cuda:
        model.cuda()

    clf = BraindecodeClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=optim.AdamW,
        train_split=None,
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=[
            (
                "train_accuracy",
                PostEpochTrainScoring(
                    "accuracy", lower_is_better=False, name="train_acc"
                ),
            ),
            (
                "train_f1_score",
                PostEpochTrainScoring(
                    "f1", lower_is_better=False, name="train_f1"
                ),
            ),
            ("test_callback", TestCallback()),
        ],
    )

    clf.fit(train_set, y=None, epochs=4)

import numpy as np
import torch
import torch.nn as nn

from braindecode import EEGClassifier
from braindecode.util import set_random_seeds


def _train_and_predict(seed):
    set_random_seeds(seed=seed, cuda=torch.cuda.is_available())
    X = np.random.randn(8, 1, 1000).astype(np.float32)
    y = np.random.randint(0, 2, size=(8,))

    class SimpleNet(nn.Module):
        def __init__(self, n_chans, n_times, n_outputs):
            super().__init__()
            self.conv = nn.Conv1d(n_chans, 2, kernel_size=10)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(2, n_outputs)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    clf = EEGClassifier(
        SimpleNet,
        module__n_chans=X.shape[1],
        module__n_times=X.shape[2],
        module__n_outputs=2,
        max_epochs=1,
        train_split=None,
        iterator_train__shuffle=True,
    )
    clf.fit(X, y)
    return clf.predict_proba(X)


def test_reproducible_training_outputs():
    preds1 = _train_and_predict(seed=20240205)
    preds2 = _train_and_predict(seed=20240205)
    assert np.allclose(preds1, preds2)

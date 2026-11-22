""".. _contrastive-pretraining-eeg:

Contrastive pretraining of any model without labels
===================================================

This example pretrains a braindecode model on unlabelled sleep EEG with a
contrastive loss, then reuses its encoder twice: as a frozen feature extractor
under a linear probe, and as the starting point of supervised fine-tuning.

Every registered braindecode model offers the two pieces this needs:

- ``n_outputs=None`` builds the model without a head, so it holds no
  classifier parameters while it learns from unlabelled windows;
- ``model(x, return_features=True)`` returns a dict whose ``"embedding"`` is
  a ``(batch, dim)`` tensor, the encoder output averaged over its time,
  token and spatial axes.

The contrastive task is the one of SimCLR [1]_: two random augmentations of the
same window must map to nearby embeddings, and augmentations of different
windows to distant ones. :func:`braindecode.training.info_nce` computes this
loss. The data and windows are the ones of
:ref:`the relative positioning example <self-supervised-learning-eeg>`.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import scale as standard_scale

from braindecode.augmentation import Compose, FTSurrogate, SmoothTimeMask
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.models import SleepStagerChambon2018
from braindecode.modules import MLP
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.training import info_nce

random_state = 87
torch.manual_seed(random_state)

######################################################################
# Loading and windowing the data
# ------------------------------
#
# We load two recordings of the Sleep Physionet dataset [2]_ [3]_, convert them
# to microvolts, low-pass filter them at 30 Hz and cut them into the 30 s
# windows that sleep stages are scored on. The sleep stages serve only for the
# downstream evaluation: pretraining never reads them.

dataset = SleepPhysionet(subject_ids=[0, 1], recording_ids=[1], crop_wake_mins=30)
preprocess(
    dataset,
    [
        Preprocessor(lambda data: np.multiply(data, 1e6)),  # V to uV
        Preprocessor("filter", l_freq=None, h_freq=30),
    ],
)

sfreq = 100
n_times = 30 * sfreq
mapping = {  # stages 3 and 4 are merged, following AASM
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
windows = create_windows_from_events(
    dataset,
    window_size_samples=n_times,
    window_stride_samples=n_times,
    preload=True,
    mapping=mapping,
)
preprocess(windows, [Preprocessor(standard_scale, channel_wise=True)])


######################################################################
# We put each subject's windows in one tensor: subject 0 trains the downstream
# classifiers and subject 1 tests them. Pretraining uses the windows of both
# subjects without their labels.


def to_tensors(subject):
    ds = [d for d in windows.datasets if d.description["subject"] == subject]
    X = np.stack([d[i][0] for d in ds for i in range(len(d))])
    y = np.array([d[i][1] for d in ds for i in range(len(d))])
    return torch.as_tensor(X, dtype=torch.float32), y


X_train, y_train = to_tensors(0)
X_test, y_test = to_tensors(1)
n_chans = X_train.shape[1]
print(f"{len(X_train)} training and {len(X_test)} test windows, {n_chans} channels")

######################################################################
# Building a model without a head
# -------------------------------
#
# Any registered model works here; we use the small sleep stager of
# Chambon et al. [4]_. With ``n_outputs=None`` its ``final_layer`` is an
# ``nn.Identity`` without parameters, and calling it without
# ``return_features=True`` raises an error that says so.

model = SleepStagerChambon2018(
    n_chans=n_chans, sfreq=sfreq, n_times=n_times, n_outputs=None
)
print(model.final_layer)

######################################################################
# Two augmented views
# -------------------
#
# Each window gets two random views. :class:`~braindecode.augmentation.FTSurrogate`
# randomizes the Fourier phases and keeps the spectrum, and
# :class:`~braindecode.augmentation.SmoothTimeMask` zeroes a random half-second
# stretch. A small :class:`~braindecode.modules.MLP` projects the embeddings
# before the loss, as in SimCLR [1]_; it is dropped after pretraining.

augment = Compose(
    [
        FTSurrogate(probability=0.8, random_state=random_state),
        SmoothTimeMask(probability=0.8, mask_len_samples=sfreq // 2),
    ]
)

with torch.no_grad():
    dim = model(X_train[:2], return_features=True)["embedding"].shape[1]
projector = MLP(in_features=dim, hidden_features=[64], out_features=32)

######################################################################
# Contrastive pretraining
# -----------------------
#
# Each step draws a batch of windows from both subjects, embeds two views of
# it and lowers ``info_nce`` on their projections. A few epochs are enough to
# show the mechanics; real pretraining runs on far more data and steps.

X_all = torch.cat([X_train, X_test])
optimizer = torch.optim.AdamW(
    [*model.parameters(), *projector.parameters()], lr=1e-3, weight_decay=1e-4
)
batch_size, n_epochs = 64, 3
model.train()
for epoch in range(n_epochs):
    order = torch.randperm(len(X_all))
    losses = []
    for start in range(0, len(order) - batch_size + 1, batch_size):
        x = X_all[order[start : start + batch_size]]
        z1 = model(augment(x), return_features=True)["embedding"]
        z2 = model(augment(x), return_features=True)["embedding"]
        loss = info_nce(projector(z1), projector(z2), temperature=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1}: info_nce {np.mean(losses):.3f}")

######################################################################
# Linear probe on frozen embeddings
# ---------------------------------
#
# The encoder is frozen and a logistic regression is trained on the embeddings
# of subject 0, then scored on subject 1. Balanced accuracy is the mean recall
# over the five sleep stages, so chance is 0.2.


@torch.no_grad()
def embed(X):
    model.eval()
    return torch.cat(
        [model(x, return_features=True)["embedding"] for x in X.split(256)]
    ).numpy()


probe = LogisticRegression(max_iter=1000).fit(embed(X_train), y_train)
probe_score = balanced_accuracy_score(y_test, probe.predict(embed(X_test)))
print(f"Linear probe balanced accuracy: {probe_score:.3f}")

######################################################################
# Supervised fine-tuning
# ----------------------
#
# ``reset_head`` attaches a five-class head and keeps every pretrained weight;
# the model is then trained end to end on the labelled windows of subject 0.

model.reset_head(5)
print(model.final_layer)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
targets = torch.as_tensor(y_train)
model.train()
for epoch in range(3):
    order = torch.randperm(len(X_train))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        loss = torch.nn.functional.cross_entropy(model(X_train[idx]), targets[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = torch.cat([model(x) for x in X_test.split(256)]).argmax(dim=1).numpy()
print(f"Fine-tuned balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")

######################################################################
# The same code runs with any other registered model: change the class and its
# constructor arguments, keep ``n_outputs=None``, and the pretraining,
# probing and fine-tuning steps stay as they are.
#
# References
# ----------
#
# .. [1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple
#        framework for contrastive learning of visual representations. ICML.
#        https://arxiv.org/abs/2002.05709
#
# .. [2] Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A., & Oberye,
#        J. J. (2000). Analysis of a sleep-dependent neuronal feedback loop:
#        the slow-wave microcontinuity of the EEG. IEEE Transactions on
#        Biomedical Engineering, 47(9), 1185-1194.
#
# .. [3] Goldberger, A. L., Amaral, L. A., Glass, L., Hausdorff, J. M., Ivanov,
#        P. C., Mark, R. G., ... & Stanley, H. E. (2000). PhysioBank,
#        PhysioToolkit, and PhysioNet: components of a new research resource
#        for complex physiologic signals. Circulation, 101(23), e215-e220.
#
# .. [4] Chambon, S., Galtier, M., Arnal, P., Wainrib, G., & Gramfort, A.
#        (2018). A deep learning architecture for temporal sleep stage
#        classification using multivariate and multimodal time series. IEEE
#        Transactions on Neural Systems and Rehabilitation Engineering, 26(4),
#        758-768.

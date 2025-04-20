import sys

sys.path.append("source")

from __future__ import annotations

import warnings

import moabb
import pandas as pd
import torch
from fbcnet_without_first_layer import MIBIF, FBCNetNoFilter
from joblib import Parallel, delayed
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines.utils import FilterBank
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

from braindecode import EEGClassifier

# ——— fixed setup ———
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# dataset & paradigm
dataset = BNCI2014_001()
filters = (
    [4, 8], [8, 12], [12, 16], [16, 20],
    [20, 24], [24, 28], [28, 32], [32, 36], [36, 40],
)
paradigm = FilterBankMotorImagery(n_classes=4, filters=filters)

# define your two pipelines
pipelines = {
    "FBCSP": make_pipeline(
        FilterBank(estimator=CSP(n_components=4, reg="oas")),
        MIBIF(n_selected_features=8),
        LDA(solver="eigen", shrinkage="auto"),
    ),
    "FBCNetNotFilter": make_pipeline(
        EEGClassifier(
            FBCNetNoFilter(n_bands=len(filters), n_chans=22, n_outputs=4, n_times=1001, sfreq=250),
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            train_split=ValidSplit(0.2, stratified=True, random_state=42),
            batch_size=16,
            callbacks=[
                "balanced_accuracy",
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=1500 - 1)),
                ("early_stop",   EarlyStopping(patience=200)),
            ],
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_epochs=1500,
        )
    ),
}


def evaluate(subject: int, pipeline_name: str, pipe) -> list[dict]:
    """Fit & score one (subject, pipeline) pair, returning train/test rows."""
    # 1) load data
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    y = LabelEncoder().fit_transform(labels)

    # 2) splits
    ix_train = meta.query("session == '0train'").index.to_numpy()
    ix_test  = meta.query("session == '1test'").index.to_numpy()

    # 3) fit & predict
    pipe.fit(X[ix_train], y[ix_train])
    y_hat = pipe.predict(X)

    # 4) collect results
    rows = []
    for split, ix in (("train", ix_train), ("test", ix_test)):
        rows.append({
            "subject":  subject,
            "pipeline": pipeline_name,
            "split":    split,
            "acc":      accuracy_score(y[ix], y_hat[ix]),
        })
    return rows


# build all (subject, pipeline) tasks
tasks = [
    (subj, name, pipelines[name])
    for subj in dataset.subject_list
    for name in pipelines
]

# run them *all* in parallel
all_results = Parallel(n_jobs=-1)(
    delayed(evaluate)(subj, name, pipe)
    for subj, name, pipe in tasks
)

# flatten and aggregate
flat = [row for subject_out in all_results for row in subject_out]
res_df = pd.DataFrame(flat)

# save or inspect
res_df.to_csv("results_parallelized.csv", index=False)
print(res_df)

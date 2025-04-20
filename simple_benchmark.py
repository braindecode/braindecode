# %%
import sys

sys.path.append("source")

import matplotlib.pyplot as plt
import mne
import seaborn as sns
import torch
from joblib import Parallel, delayed
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import FilterBankMotorImagery, LeftRightImagery, MotorImagery
from moabb.utils import setup_seed
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from braindecode import EEGClassifier
from braindecode.models import FBCNet

torch.manual_seed(24091996)
mne.set_log_level(False)

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
seed = 42
setup_seed(seed)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Hyperparameter
BATCH_SIZE = 16  # parameter taken from BrainDecode
EPOCH = 1500
PATIENCE = 200
fmin = 0.5
fmax = 100
tmin = -0.5
tmax = None

# Load the dataset
dataset = BNCI2014_001()
paradigm = MotorImagery(
    fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
)

n_classes = 4
subjects = [1]
X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
n_times = X.shape[2]
n_chans = X.shape[1]
sfreq = 250


# helper to run one subject
def evaluate_subject(subject_id, model_name, class_module):
    # 1) build a single‐subject pipeline
    pipes = {
        f"{model_name}_subj_{subject_id}":
            EEGClassifier(
                module=class_module(n_times=n_times, n_chans=n_chans, n_outputs=n_classes, sfreq=250),
                module__n_times=n_times,
                module__n_chans=n_chans,
                module__n_outputs=n_classes,
                module__sfreq=250,
                optimizer=torch.optim.Adam,
                batch_size=BATCH_SIZE,
                max_epochs=EPOCH,
                train_split=ValidSplit(0.2, random_state=42, stratified=True),
                device=device,
                callbacks=[
                    EarlyStopping(monitor="valid_loss", patience=PATIENCE),
                    EpochScoring("accuracy", on_train=True,  name="train_acc", lower_is_better=False),
                    EpochScoring("accuracy", on_train=False, name="valid_acc", lower_is_better=False),
                ],
                criterion=torch.nn.CrossEntropyLoss(),
                verbose=1,
            )
    }
    # 2) restrict the dataset to this subject
    ds = BNCI2014_001()
    ds.subject_list = [subject_id]
    # 3) run cross‐session eval (set n_jobs=1 here to avoid nesting parallel pools)
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=ds,
        suffix=f"subject_{subject_id}",
        overwrite=True,
        n_jobs=1,
    )
    return evaluation.process(pipes)

model_name = "FBCNet"
class_module = FBCNet

# now launch them in parallel
results = Parallel(n_jobs=-1)(
    delayed(evaluate_subject)(subj, model_name, class_module)
    for subj in dataset.subject_list
)

# merge into one DataFrame if you like
import pandas as pd

all_results = pd.concat(results, ignore_index=True)

all_results.to_csv(f"results_{model_name}.csv", index=False)

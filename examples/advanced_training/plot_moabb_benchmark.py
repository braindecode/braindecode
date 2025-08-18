"""===============================================================
Cross-session motor imagery with deep learning EEGNet v4 model
===============================================================
This example shows how to use BrainDecode in combination with MOABB evaluation.
In this example, we use the architecture EEGNetv4.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import seaborn as sns
import torch
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.utils import setup_seed
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from braindecode import EEGClassifier
from braindecode.models import EEGNetv4

mne.set_log_level(False)

# Print Information PyTorch
print(f"Torch Version: {torch.__version__}")

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")

###############################################################################
# In this example, we will use only the dataset ``BNCI2014_001``.
#
# Running the benchmark
# ---------------------
#
# This example uses the CrossSession evaluation procedure. We focus on the dataset BNCI2014_001 and only on 1 subject
# to reduce computational time.
#
# To keep the computational time low, the epoch is reduced. In a real situation, we suggest using the following:
# EPOCH = 1000
# PATIENCE = 300
#
# This code is implemented to run on the CPU. If you're using a GPU, do not use multithreading
# (i.e. set n_jobs=1)


# Set random seed to be able to reproduce results
seed = 42
setup_seed(seed)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameter
LEARNING_RATE = 0.0625 * 0.01  # parameter taken from Braindecode
WEIGHT_DECAY = 0  # parameter taken from Braindecode
BATCH_SIZE = 64  # parameter taken from BrainDecode
EPOCH = 10
PATIENCE = 3
fmin = 4
fmax = 100
tmin = 0
tmax = None

# Load the dataset
dataset = BNCI2014_001()
events = ["right_hand", "left_hand"]
paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
)
subjects = [1]
X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)

##############################################################################
# Create Pipelines
# ----------------
# In order to create a pipeline, we need to load a model from braindecode.
# the second step is to define a skorch model using EEGClassifier from braindecode
# that allows converting the PyTorch model in a scikit-learn classifier.
# Here, we will use the EEGNet v4 model [1]_ .
# This model has mandatory hyperparameters (the number of channels, the number of classes,
# and the temporal length of the input) but we do not need to specify them because they will
# be set dynamically by EEGClassifier using the input data during the call to the ``.fit()`` method.

# Define a Skorch classifier
clf = EEGClassifier(
    module=EEGNetv4,
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=seed),
    device=device,
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
    ],
    verbose=1,  # Not printing the results for each epoch
)

# Create the pipelines
pipes = {}
pipes["EEGNetV4"] = make_pipeline(clf)

##############################################################################
# Evaluation
# ----------
dataset.subject_list = dataset.subject_list[:2]

evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    suffix="braindecode_example",
    overwrite=True,
    return_epochs=True,
    n_jobs=1,
)

results = evaluation.process(pipes)

print(results.head())

##############################################################################
# Plot Results
# ----------------
plt.figure()
sns.barplot(data=results, y="score", x="subject", palette="viridis")
plt.show()
##############################################################################
# References
# ----------
# .. [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
#    Hung, C. P., & Lance, B. J. (2018). `EEGNet: a compact convolutional neural
#    network for EEG-based brain-computer interfaces.
#    <https://doi.org/10.1088/1741-2552/aace8c>`_
#    Journal of neural engineering, 15(5), 056013.

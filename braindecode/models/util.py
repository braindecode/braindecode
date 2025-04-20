# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax
from sklearn.utils import deprecated

import braindecode.models as models

models_dict = {}

# For the models inside the init model, go through all the models
# check those have the EEGMixin class inherited. If they are, add them to the
# list.


def _init_models_dict():
    for m in inspect.getmembers(models, inspect.isclass):
        if (
            issubclass(m[1], models.base.EEGModuleMixin)
            and m[1] != models.base.EEGModuleMixin
        ):
            models_dict[m[0]] = m[1]


################################################################
# Test cases for models
#
# This list should be updated whenever a new model is added to
# braindecode (otherwise `test_completeness__models_test_cases`
# will fail).
# Each element in the list should be a tuple with structure
# (model_class, required_params, signal_params), such that:
#
# model_name: str
#   The name of the class of the model to be tested.
# required_params: list[str]
#   The signal-related parameters that are needed to initialize
#   the model.
# signal_params: dict | None
#   The characteristics of the signal that should be passed to
#   the model tested in case the default_signal_params are not
#   compatible with this model.
#   The keys of this dictionary can only be among those of
#   default_signal_params.
################################################################
models_mandatory_parameters = [
    ("ATCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("BDTCN", ["n_chans", "n_outputs"], None),
    ("Deep4Net", ["n_chans", "n_outputs", "n_times"], None),
    ("DeepSleepNet", ["n_outputs"], None),
    ("EEGConformer", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGInceptionERP", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGInceptionMI", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGITNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNetv1", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNetv4", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGResNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ShallowFBCSPNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "SleepStagerBlanco2020",
        ["n_chans", "n_outputs", "n_times"],
        # n_chans dividable by n_groups=2:
        dict(chs_info=[dict(ch_name=f"C{i}", kind="eeg") for i in range(1, 5)]),
    ),
    ("SleepStagerChambon2018", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    (
        "SleepStagerEldele2021",
        ["n_outputs", "n_times", "sfreq"],
        dict(sfreq=100, n_times=3000, chs_info=[dict(ch_name="C1", kind="eeg")]),
    ),  # 1 channel
    ("TIDNet", ["n_chans", "n_outputs", "n_times"], None),
    ("USleep", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=128)),
    ("BIOT", ["n_chans", "n_outputs", "sfreq"], None),
    ("AttentionBaseNet", ["n_chans", "n_outputs", "n_times"], None),
    ("Labram", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGSimpleConv", ["n_chans", "n_outputs", "sfreq"], None),
    ("SPARCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ContraWR", ["n_chans", "n_outputs", "sfreq"], dict(sfreq=200)),
    ("EEGNeX", ["n_chans", "n_outputs", "n_times"], None),
    ("TSceptionV1", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("EEGTCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SyncNet", ["n_chans", "n_outputs", "n_times"], None),
    ("MSVTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGMiner", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("CTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SincShallowNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=250)),
    ("SCCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("SignalJEPA", ["chs_info"], None),
    ("SignalJEPA_Contextual", ["chs_info", "n_times", "n_outputs"], None),
    ("SignalJEPA_PostLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("SignalJEPA_PreLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("FBCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("FBMSNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("FBLightConvNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
    ("IFNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200)),
]

################################################################
# List of models that are not meant for classification
#
# Their output shape may difer from the expected output shape
# for classification models.
################################################################
non_classification_models = [
    "SignalJEPA",
]


################################################################
def get_summary_table(dir_name=None):
    if dir_name is None:
        dir_path = Path(__file__).parent
    else:
        dir_path = Path(dir_name) if not isinstance(dir_name, Path) else dir_name

    path = dir_path / "summary.csv"

    df = pd.read_csv(
        path,
        header=0,
        index_col="Model",
        skipinitialspace=True,
    )
    return df


_summary_table = get_summary_table()

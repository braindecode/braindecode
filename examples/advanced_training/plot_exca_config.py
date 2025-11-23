# type: ignore

"""Experiment configuration with Pydantic and Exca
==================================================

This example shows how to use the ``pydantic`` and ``exca`` libraries
to configure and run EEG experiments with Braindecode.

**Pydantic**, in a nutshell, is a library for data validation and settings management
using Python type annotations. It allows defining structured configurations that can be
validated and serialized easily.

**Exca** builds on top of Pydantic, and allows you to seamlessly **ex**ecute experiments
and **ca**che their results.

Braindecode implements a Pydantic configurations for each of its models in
``braindecode.models.config``.
In this example, we will use these configurations to define an experiment that
trains and evaluates different models on a motor-imagery dataset using Exca.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Pierre Guetschel
#
# License: BSD (3-clause)

# %%#####################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# We start by loading and preprocessing the MOABB motor imagery dataset.
# We use a single subject for speed.
#
# Loading
# ~~~~~~~

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#


from braindecode.preprocessing import (
    Filter,
    PickTypes,
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    PickTypes(eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Filter(l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

preprocess(dataset, preprocessors, n_jobs=-1)

# %%######################################################################
# Extracting windows
# ~~~~~~~~~~~~~~~~~~
#

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to
# define how trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
valid_set = splitted["1test"]  # Session evaluation
train_y = train_set.get_metadata().target.values
test_y = valid_set.get_metadata().target.values

# %%#####################################################################
# Defining experiment configurations with Pydantic and Exca
# ------------------------------------------------------
#
# We define Pydantic configurations for training and evaluation using Exca.


import exca
import pydantic
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

from braindecode.models.config import (
    BraindecodeModelConfig,
    EEGConformerConfig,  # type: ignore
    EEGNetConfig,  # type: ignore
)


class TrainingConfig(pydantic.BaseModel):
    model: BraindecodeModelConfig
    max_epochs: int = 50
    lr: float = 0.01
    seed: int = 12
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=".cache/",
        cluster=None,  # local execution
    )

    @infra.apply
    def train(self) -> EEGClassifier:
        model = self.model.create_instance()  # type: ignore[attr-defined]
        clf = EEGClassifier(
            model,
            max_epochs=self.max_epochs,
            lr=self.lr,
            default=ValidSplit(5, random_state=self.seed),
            callbacks=["accuracy", EarlyStopping(patience=3)],
        )
        clf.fit(train_set, train_y)
        return clf.module_.state_dict()


class EvaluationConfig(pydantic.BaseModel):
    trainer: TrainingConfig
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=".cache/",
        cluster=None,  # local execution
    )

    @infra.apply
    def evaluate(self) -> float:
        state_dict = self.trainer.train()
        model = self.trainer.model.create_instance()
        model.load_state_dict(state_dict)
        clf = EEGClassifier(model)
        clf.initialize()
        return clf.score(valid_set, test_y)


# %%


def flatten_nested_dict(d, leaf_types=(int, float, str, bool), sep="."):
    def aux(d, parent_key):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(aux(v, parent_key + k + sep))
            elif isinstance(v, leaf_types):
                out[parent_key + k] = v
        return out

    return aux(d, "")


import time

import pandas as pd

flatten_nested_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3, "d": [4, 5]})
# %%
signal_kwargs = {"n_chans": 16, "n_times": 1000, "n_outputs": 4}
model_cfg = EEGNetConfig.model_validate(signal_kwargs)
train_cfg = TrainingConfig(model=model_cfg, max_epochs=5)
eval_cfg = EvaluationConfig(trainer=train_cfg)
# %%

t0 = time.time()
train_cfg.train()
print(f"Training took {time.time() - t0:0.2f} seconds")
# %%

t0 = time.time()
train_cfg.train()
print(f"Rerunning training using cached results took {time.time() - t0:0.4f} seconds")

# %%
t0 = time.time()
eval_cfg.evaluate()
print(f"Evaluation took {time.time() - t0:0.2f} seconds")

# %%
t0 = time.time()
eval_cfg.evaluate()
print(f"Rerunning evaluation using cached results took {time.time() - t0:0.4f} seconds")
# %%
print(eval_cfg)
eval_cfg.model_dump(mode="json", serialize_as_any=True)
# %%
model_cfg_list = [
    EEGNetConfig.model_validate(signal_kwargs),
    EEGConformerConfig.model_validate(signal_kwargs),
]

results = []
for model_cfg in model_cfg_list:
    for seed in [0, 1, 2]:
        row = flatten_nested_dict(
            eval_cfg.infra.config(uid=True, exclude_defaults=True)
        )
        train_cfg = TrainingConfig(model=model_cfg, max_epochs=10, lr=0.1, seed=seed)
        eval_cfg = EvaluationConfig(trainer=train_cfg)
        row["accuracy"] = eval_cfg.evaluate()
        results.append(row)
results_df = pd.DataFrame(results)
# %%
results_df

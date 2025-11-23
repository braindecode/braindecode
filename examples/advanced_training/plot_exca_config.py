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

#####################################################################
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

subject_id = 1
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])


######################################################################
# Extracting windows
# ~~~~~~~~~~~~~~~~~~
#
# We don't apply any preprocessing here for simplicity, but in a real experiment,
# you would typically want to filter the data, resample it, etc.
#
# Instead, we directly extract windows from the raw data:

from braindecode.preprocessing import create_windows_from_events

windows_dataset = create_windows_from_events(dataset, preload=True)

######################################################################
# Split dataset into train and valid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
valid_set = splitted["1test"]  # Session evaluation
train_y = train_set.get_metadata().target.values
test_y = valid_set.get_metadata().target.values

#####################################################################
# Extract signal properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We now extract the properties of the signals in the dataset (i.e., number of channels, number of time points, and number of classes).
# For this, we will make use of the function :func:`braindecode.datautil.infer_signal_properties`.
#
from braindecode.datautil import infer_signal_properties

signal_kwargs = infer_signal_properties(train_set, train_y, mode="classification")
print(signal_kwargs)

#####################################################################
# Configuring and running experiment with Pydantic and Exca
# ------------------------------------------------------
#
# Defining the configuration classes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that out training and testing data is ready, we can define our experiment using Pydantic and Exca.
# We define tho configurations: one for training a model, and one for testing it.
#
# We note in the training that the model has type :class:`braindecode.models.config.BraindecodeModelConfig`. This type can match all the braindecode model configurations defined in :mod:`braindecode.models.config`.
# We can see that both configs have an ``ingra: exca.TaskInfra``,
# and a method decorated with ``@infra.apply``.
# This means that, when called, exca will cache the results of these methods in the specified folder (``.cache/`` here).
# If the method is called again with the same configuration, the cached results will be returned instead of re-running the method.
# This allows for easy and efficient experimentation.
#
# Exca also offers the possibility to run experiments remotely on a SLURM-managed cluster. In this example, we run everything locally by setting ``cluster=None`` but you can find more information about how to set up cluster execution in the [Exca documentation](https://facebookresearch.github.io/exca/infra/introduction.html).


import exca
import pydantic
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch.optim import Adam

from braindecode.models.config import BraindecodeModelConfig


class TrainingConfig(pydantic.BaseModel):
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=".cache/",
        cluster=None,  # local execution
    )
    model: BraindecodeModelConfig
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    seed: int = 12

    @infra.apply
    def train(self) -> EEGClassifier:
        model = self.model.create_instance()
        clf = EEGClassifier(
            model,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            train_split=ValidSplit(0.2, random_state=self.seed, stratified=True),
            callbacks=["accuracy", EarlyStopping(patience=3)],
            optimizer=Adam,
        )
        clf.fit(train_set, train_y)
        return clf.module_.state_dict()


class EvaluationConfig(pydantic.BaseModel):
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=".cache/",
        cluster=None,  # local execution
    )
    trainer: TrainingConfig

    @infra.apply
    def evaluate(self) -> float:
        state_dict = self.trainer.train()
        model = self.trainer.model.create_instance()
        model.load_state_dict(state_dict)
        clf = EEGClassifier(model)
        clf.initialize()
        return clf.score(valid_set, test_y)


#####################################################################
# Instantiating the configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that our configuration classes are defined, we can instantiate them.
#
# We will start with the model configuration.
# Here, we use the :class:`braindecode.models.EEGNet` model.
# Like any other braindecode model, it has a corresponding configuration class in :mod:`braindecode.models.config`, called :class:`braindecode.models.config.EEGNetConfig`.
# We instantiate it using the signal properties we extracted earlier.

from braindecode.models.config import EEGConformerConfig, EEGNetConfig

model_cfg = EEGNetConfig(**signal_kwargs)

#################################################################
# The config object can easily be serialized to a JSON format:
print(model_cfg.model_dump(mode="json"))

#################################################################
# Or if you only want the non-default keys:
print(model_cfg.model_dump(exclude_defaults=True))

#####################################################################
# The config class is checking the arguments types and values, and
# raises an error if something is wrong. For example, if we try to instantiate it using an incorrec type for ``n_times``, we get an error:
true_n_times = signal_kwargs["n_times"]
signal_kwargs["n_times"] = 22.5  # float instead of int
try:
    EEGNetConfig(**signal_kwargs)
except pydantic.ValidationError as e:
    print(f"Validation error raised as expected:\n{e}")

##############################################################################
# Similarly, if a mandatory argument is missing, we get an error:
del signal_kwargs["n_times"]
try:
    EEGNetConfig(**signal_kwargs)
except pydantic.ValidationError as e:
    print(f"Validation error raised as expected:\n{e}")

# We restore the correct value for ``n_times`` for the rest of the example:
signal_kwargs["n_times"] = true_n_times

#####################################################
# Now that we have the model configuration, we can instantiate the training
# and evaluation configurations:
#
train_cfg = TrainingConfig(model=model_cfg)
eval_cfg = EvaluationConfig(trainer=train_cfg)


#################################################################
# And we can again print the configuration that defines our whole experiment using Pydantic's ``model_dump`` method:
print(eval_cfg.model_dump(exclude_defaults=True))

#################################################################
# However, we see above that the model configuration could correspond to any model in Braindecode.
#
# Exca takes it into account when generating unique IDs for caching.
# We can print the full configuration with unique IDs using Exca's ``config`` method:
print(eval_cfg.infra.config(uid=True, exclude_defaults=True))


#####################################################
# Running the experiment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now run the training using the configurations we defined.
# For this, we simply have to call the ``train`` method of the configuration.
# we will time the execution to see the benefits of caching.
#
import time

t0 = time.time()
train_cfg.train()
t1 = time.time()

print(f"Training took {t1 - t0:0.2f} seconds")

#############################################################
# If we call the ``train`` method again, using the same configuration parameters, even if it is a new instance, the results will be loaded from the cache
#

train_cfg = TrainingConfig(model=EEGNetConfig(**signal_kwargs))

t0 = time.time()
train_cfg.train()
t1 = time.time()

print(f"Rerunning training using cached results took {t1 - t0:0.4f} seconds")

#############################################################
# We can run the evaluation in the same way, by calling the ``evaluate`` method of the evaluation configuration.
# Internally, this method calls the ``train`` method of the training configuration, which will also use the cache if available.
#

t0 = time.time()
score = eval_cfg.evaluate()
t1 = time.time()

print(f"Evaluation score: {score}")
print(f"Evaluation took {t1 - t0:0.2f} seconds")


#####################################################################
# Scaling up: comparing multiple model configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have seen how to define and run an experiment using Pydantic and Exca,
# we can easily scale up to compare multiple model configurations.
#
# First, let's define a small utility function to flatten nested dictionaries.
# This will help us later when we want to log results from different configurations.
# See in the example below, the keys of different levels are concatenated with a dot "." separator.


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


flatten_nested_dict({"a": 1, "b": {"x": 1, "y": {"z": 2}}, "c": [4, 5]})

#####################################################################
# In a real experiment, we would launch all runs in parallel on a different nodes of a compute cluster.
# Please refer to the [Exca documentation](https://facebookresearch.github.io/exca/infra/introduction.html) for more details on how to set up cluster execution.
# Here, for simplicity, we will just run them locally and sequentially.
#
# In this mini-example, we will compare the EEGNet and EEGConformer models on the same dataset, with multiple random seeds.

model_cfg_list = [
    EEGNetConfig(**signal_kwargs),
    EEGConformerConfig(**signal_kwargs),
]

results = []
for model_cfg in model_cfg_list:
    for seed in [1, 2, 3]:
        train_cfg = TrainingConfig(model=model_cfg, max_epochs=10, lr=0.1, seed=seed)
        eval_cfg = EvaluationConfig(trainer=train_cfg)

        # log configuration
        row = flatten_nested_dict(
            eval_cfg.infra.config(uid=True, exclude_defaults=True)
        )
        # evaluate and log accuracy:
        row["accuracy"] = eval_cfg.evaluate()
        results.append(row)
#####################################################################
# Finally, display the results using pandas:
import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)
##############################################################
# Or first aggregated over seeds:
agg_results_df = results_df.groupby("trainer.model.model_name_").agg(
    {"accuracy": ["mean", "std"]}
)
print(agg_results_df)

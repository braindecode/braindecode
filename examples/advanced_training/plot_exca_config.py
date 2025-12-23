# type: ignore
"""Experiment configuration with Pydantic and Exca
==================================================

This example shows how to use the ``pydantic`` and ``exca`` libraries
to configure and run EEG experiments with Braindecode.

**Pydantic** is a library for data validation and settings management
using Python type annotations. It allows defining structured configurations that can be
validated and serialized easily.

**Exca** builds on top of Pydantic, and allows you to seamlessly EXecute experiments
and CAche their results.

Braindecode implements a Pydantic configuration for each of its models in
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
# Creating the experiment configurations
# ---------------------------------------
#
# We will start by defining the configurations needed for our experiment using Pydantic and Exca.
#
# Dataset configs
# ~~~~~~~~~~~~~~~
#
# Our first configuration class is related to the data. It will allow us to load and prepare the dataset.
import warnings
from typing import Annotated, Literal

import exca
import pydantic
from moabb.datasets.utils import dataset_list

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

warnings.simplefilter("ignore")

# The list of available MOABB datasets:
DATASET_NAMES = tuple(ds.__name__ for ds in dataset_list)


class WindowedMOABBDatasetConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    dataset_type: Literal["moabb"] = "moabb"
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=None,  # no disk caching
        cluster=None,  # local execution
        keep_in_ram=True,
    )
    dataset_name: Literal[DATASET_NAMES] = "BNCI2014_001"
    subject_id: list[int] | int | None = None
    window_size_seconds: float = 4.0
    overlap_seconds: float = 0.0

    @infra.apply
    def create_instance(self) -> MOABBDataset:
        # We don't apply any preprocessing here for simplicity, but in a real experiment,
        # you would typically want to filter the data, resample it, etc.
        # Instead, our config  directly extracts windows from the raw data.
        dataset = MOABBDataset(
            dataset_name=self.dataset_name, subject_ids=self.subject_id
        )
        windows_dataset = create_windows_from_events(dataset, preload=True)

        return windows_dataset


#################################################################
#
# We can see that the config has an ``infra: exca.TaskInfra`` attribute,
# and a method decorated with ``@infra.apply``.
# This means that, when called, exca will cache the result of this method.
# Here, the cache is kept in RAM for simplicity (``folder=None``), but in a real experiment,
# you would typically want to cache the results on disk, as shown in the training config.
# If the method is called again with the same configuration, the cached results will be returned instead of re-running the method.
# This allows for easy and efficient experimentation.
#
# Additionally, we define a small wrapper config to split the dataset into training and testing sets.
# Here, no caching is applied since the split operation is fast.


class DatasetSplitConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    dataset_type: Literal["split"] = "split"
    dataset: WindowedMOABBDatasetConfig
    key: str
    by: str = "session"

    def create_instance(self):
        dataset = self.dataset.create_instance()
        splitted = dataset.split(self.by)
        return splitted[self.key]


#################################################################
# Finally, we define a union type for dataset configurations,
# which can be either a ``WindowedMOABBDatasetConfig`` or a ``DatasetSplitConfig``.

DatasetConfig = Annotated[
    WindowedMOABBDatasetConfig | DatasetSplitConfig,
    pydantic.Field(discriminator="dataset_type"),
]


#####################################################################
# Training config
# ~~~~~~~~~~~~~~~~
#
# Now that out data configs are ready, we can define our training config. It will require both the dataset and model configurations.
# It will simply load the data, instantiate the model, and train the model on the data.
#

from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch.optim import Adam

from braindecode.models.config import BraindecodeModelConfig


class TrainingConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=".cache/",
        cluster=None,  # local execution
    )
    model: BraindecodeModelConfig
    train_dataset: DatasetConfig
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    seed: int = 12

    @infra.apply
    def train(self) -> EEGClassifier:
        # Load training data
        train_set = self.train_dataset.create_instance()
        train_y = train_set.get_metadata()["target"].to_numpy()

        # Instantiate the model
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

        # Train the model
        clf.fit(train_set, train_y)
        return clf.module_.state_dict()


######################################################################
# We note that the model has type :class:`braindecode.models.config.BraindecodeModelConfig`. This type can match all the braindecode model configurations defined in :mod:`braindecode.models.config`.
#
# We also see that there is now a cache folder specified (``.cache/`` here). This means that the results of the ``train()`` method will be cached on disk in this folder, instead of only in RAM.
#
#
# Evaluation config
# ~~~~~~~~~~~~~~~~~
#
# Finally, we define an evaluation config that will load the validation data,
# load the trained model from the training config, and evaluate it on the validation data.
#
class EvaluationConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    infra: exca.TaskInfra = exca.TaskInfra(
        folder=".cache/",
        cluster=None,  # local execution
    )
    test_dataset: DatasetConfig
    trainer: TrainingConfig

    @infra.apply
    def evaluate(self) -> float:
        # Load validation data
        valid_set = self.test_dataset.create_instance()
        test_y = valid_set.get_metadata()["target"].to_numpy()

        # Load trained model
        state_dict = self.trainer.train()
        model = self.trainer.model.create_instance()
        model.load_state_dict(state_dict)
        clf = EEGClassifier(model)
        clf.initialize()

        # Evaluate the model
        score = clf.score(valid_set, test_y)
        return score


#####################################################################
# .. note:: **SLURM execution.**
#     Exca also offers the possibility to run experiments remotely on a SLURM-managed cluster.
#     In this example, we run everything locally by setting ``cluster=None``
#     but you can find more information about how to set up cluster execution
#     in the Exca documentation: https://facebookresearch.github.io/exca/infra/introduction.html.
#
# Instantiating the configurations
# ---------------------------------
#
# Instantiation option 1: from class constructors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that our configuration classes are defined, we can instantiate them.
#
# We will start with the model configuration.
# Here, we use the :class:`braindecode.models.EEGNet` model.
# Like any other braindecode model, it has a corresponding configuration class in :mod:`braindecode.models.config`, called :class:`braindecode.models.config.EEGNetConfig`.
# We instantiate it using the signal properties we extracted earlier.

from braindecode.models.config import EEGConformerConfig, EEGNetConfig

signal_kwargs = {"n_times": 1000, "n_chans": 26, "n_outputs": 4}
model_cfg = EEGNetConfig(**signal_kwargs)

#################################################################
# The config object can easily be serialized to a JSON format:
print(model_cfg.model_dump(mode="json"))

#################################################################
# Alternatively, if you only want the non-default keys:
print(model_cfg.model_dump(exclude_defaults=True))

#####################################################################
# The config class is checking the arguments types and values, and
# raises an error if something is wrong. For example, if we try to instantiate it using an incorrect type for ``n_times``, we get an error:

# kept for restoration later:
true_n_times = signal_kwargs["n_times"]

# float instead of int:
signal_kwargs["n_times"] = 22.5

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
# We now have instantiated the model configuration.
# Creating the dataset, training and evaluation configurations is very similar and
# straightforward using the classes we defined earlier.
#
dataset_cfg = WindowedMOABBDatasetConfig(subject_id=1)

train_dataset_cfg = DatasetSplitConfig(dataset=dataset_cfg, key="0train")
test_dataset_cfg = DatasetSplitConfig(dataset=dataset_cfg, key="1test")

train_cfg = TrainingConfig(model=model_cfg, train_dataset=train_dataset_cfg)

eval_cfg = EvaluationConfig(trainer=train_cfg, test_dataset=test_dataset_cfg)


#################################################################
# Instantiation option 2: from nested dictionaries or JSON files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Alternatively, we can also instantiate the configurations from nested dictionaries or JSON files.
# This can be useful when loading configurations from external sources.
# Suppose we have the following JSON configuration for our evaluation.
# We can load it as a nested dictionary using the ``json`` module:
import json

JSON_CFG = """{
    "trainer": {
        "model": {
            "model_name_": "EEGNet",
            "n_times": 1000,
            "n_chans": 26,
            "n_outputs": 4
        },
        "train_dataset": {
            "dataset_type": "split",
            "dataset": {"subject_id": 1},
            "key": "0train"
        }
    },
    "test_dataset": {
        "dataset_type": "split",
        "dataset": {"subject_id": 1},
        "key": "1test"
    }
}"""
NESTED_DICT_CFG = json.loads(JSON_CFG)
print(NESTED_DICT_CFG)

###############################################################
# We can instantiate the evaluation configuration from the nested dictionary
# using the ``model_validate()`` method of Pydantic,
# and check that it is identical to the one we created using the class constructors:

eval_cfg_from_dict = EvaluationConfig.model_validate(NESTED_DICT_CFG)
assert eval_cfg_from_dict == eval_cfg

#################################################################
# Serializing the experiment configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To serialize the experiment's configuration, we can take advantage of Exca's ``config()`` method, which is similar to Pydantic's ``model_dump()`` method but will ensure that an experiment has a unique identifier (UID).
# In particular, it will also include the ``"model_name_"`` field, which will allow us to distinguish between different model configurations later on.
print(eval_cfg.infra.config(uid=True, exclude_defaults=True))


#####################################################
# Running the experiment
# ----------------------
#
# Intermediate results are cached thanks to Exca
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now run the training using the configurations we defined.
# For this, we simply have to call the ``train()`` method of the configuration.
# we will time the execution to see the benefits of caching.
#
import time

t0 = time.time()
train_cfg.train()
t1 = time.time()

print(f"Training took {t1 - t0:0.2f} seconds")

#############################################################
# If we call the ``train()`` method again, using the same configuration parameters, even if it is a new instance, the results will be loaded from the cache:
#

train_cfg = TrainingConfig(
    model=EEGNetConfig(**signal_kwargs), train_dataset=train_dataset_cfg
)

t0 = time.time()
train_cfg.train()
t1 = time.time()

print(f"Rerunning training using cached results took {t1 - t0:0.4f} seconds")

#############################################################
# We can run the evaluation in the same way, by calling the ``evaluate()`` method of the evaluation configuration.
# Internally, this method calls the ``train()`` method of the training configuration, which will also use the cache if available.
#

t0 = time.time()
score = eval_cfg.evaluate()
t1 = time.time()

print(f"Evaluation score: {score}")
print(f"Evaluation took {t1 - t0:0.2f} seconds")


#####################################################################
# Scaling up: comparing multiple model configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# Please refer to the Exca documentation for more details on how to set up cluster execution.
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
        train_cfg = TrainingConfig(
            model=model_cfg,
            train_dataset=train_dataset_cfg,
            max_epochs=10,
            lr=0.1,
            seed=seed,
        )
        eval_cfg = EvaluationConfig(trainer=train_cfg, test_dataset=test_dataset_cfg)

        # log configuration
        row = flatten_nested_dict(
            eval_cfg.infra.config(uid=True, exclude_defaults=True)
        )
        # evaluate and log accuracy:
        row["accuracy"] = eval_cfg.evaluate()
        results.append(row)
#####################################################################
# Gathering and displaying the results
# -------------------------------------
#
# Loading results from cache
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If experiments were done on a cluster, a likely scenario would be
# to first run all experiments, and then later load and analyze the results.
#
# Loading the results from cache is straightforward using Exca.
# We simply need to re-instantiate the configurations with the same parameters,
# and call the ``evaluate()`` method again.
# The cached results will be loaded in a few seconds instead of re-running the experiments:
del results  # oups, we forgot the results...

t0 = time.time()
results = []
for model_cfg in model_cfg_list:
    for seed in [1, 2, 3]:
        train_cfg = TrainingConfig(
            model=model_cfg,
            train_dataset=train_dataset_cfg,
            max_epochs=10,
            lr=0.1,
            seed=seed,
        )
        eval_cfg = EvaluationConfig(trainer=train_cfg, test_dataset=test_dataset_cfg)

        # log configuration
        row = flatten_nested_dict(
            eval_cfg.infra.config(uid=True, exclude_defaults=True)
        )
        # evaluate and log accuracy:
        row["accuracy"] = eval_cfg.evaluate()
        results.append(row)
t1 = time.time()

print(f"Loading all results from cache took {t1 - t0:0.2f} seconds")
##############################################################
# Displaying the results
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we can concatenate and display the results using pandas:
import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)
##############################################################
# Or first aggregated over seeds:
agg_results_df = results_df.groupby("trainer.model.model_name_").agg(
    {"accuracy": ["mean", "std"]}
)
print(agg_results_df)

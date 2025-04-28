# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Alexandre Gramfort
#          Pierre Guetschel
#
# License: BSD-3
from __future__ import annotations
import sys
import inspect
import os

from copy import deepcopy
import inspect
from types import MethodType
from typing import Any

import mne
import numpy as np
import pytest
import torch
from skorch.dataset import ValidSplit
from torch import nn
from torch.export import export, ExportedProgram

from braindecode import EEGClassifier
from braindecode.models import (
    EEGInceptionMI,
    EEGMiner,
    EEGResNet,
    EEGSimpleConv,
    FBCNet,
    FBLightConvNet,
    FBMSNet,
    SyncNet,
    USleep,
)
from braindecode.models.util import (
    _summary_table,
    models_dict,
    models_mandatory_parameters,
    non_classification_models,
)

rng = np.random.default_rng(12)
# Generating the channel info
chs_info = [
    {
        "ch_name": f"C{i}",
        "kind": "eeg",
        "loc": rng.random(12),
    }
    for i in range(1, 4)
]

default_signal_params = dict(
    n_times=1000,
    sfreq=250.0,
    n_outputs=2,
    chs_info=chs_info,
    n_chans=len(chs_info),
    input_window_seconds=4.0,
)


def convert_model_to_plain(model):
    basic_mixin = [
        "_n_times",
        "_sfreq",
        "_n_times",
        "_chs_info",
        "_n_outputs",
        "_n_chans",
    ]
    final_plain_model = nn.Module()

    if isinstance(model, nn.Sequential):
        final_plain_model = nn.Sequential(*model.children())
        final_plain_model.__dict__.update({k: v for k,v in model.__dict__.items() if k != '_modules'})
    else:
        final_plain_model = nn.Module()
        for name, module in model.named_children():
            final_plain_model.add_module(name, module)

        for attr, val in model.__dict__.items():
            # Skip the modules dict to avoid double‐registering
            if attr != "_modules":
                if attr in basic_mixin:
                    setattr(final_plain_model, attr[1:], val)
                else:
                    setattr(final_plain_model, attr, val)

        # Retrieves (name, value) for every attribute of type function
        methods = inspect.getmembers(model.__class__, predicate=inspect.isfunction)

        for name, func in methods:
            # Skip Python dunders or private methods if desired
            if name.startswith("_"):
                continue
            # Bind func to final_plain_model so its signature is (self, *args, **kwargs)
            bound_method = MethodType(func, final_plain_model)
            setattr(final_plain_model, name, bound_method)

    return final_plain_model


def get_sp(
    signal_params: dict[str, Any] | None, required_params: list[str] | None = None
):
    sp = deepcopy(default_signal_params)
    if signal_params is not None:
        sp.update(signal_params)
        if "chs_info" in signal_params and "n_chans" not in signal_params:
            sp["n_chans"] = len(signal_params["chs_info"])
        if "n_chans" in signal_params and "chs_info" not in signal_params:
            sp["chs_info"] = [
                {"ch_name": f"C{i}", "kind": "eeg", "loc": rng.random(12)}
                for i in range(signal_params["n_chans"])
            ]
        assert isinstance( sp["n_times"], int)
        assert isinstance( sp["sfreq"], float)
        assert isinstance( sp["input_window_seconds"], float)         
        if "input_window_seconds" not in signal_params:
            sp["input_window_seconds"] = sp["n_times"] / sp["sfreq"]
        if "sfreq" not in signal_params:
            sp["sfreq"] = sp["n_times"] / sp["input_window_seconds"]
        if "n_times" not in signal_params:
            sp["n_times"] = int(sp["input_window_seconds"] * sp["sfreq"])
    if required_params is not None:
        sp = {k: sp[k] for k in set((signal_params or {}).keys()).union(required_params)}
    return sp


@pytest.fixture(scope="module", params=models_mandatory_parameters, ids=lambda p: p[0])
def model(request):
    """Instantiated model."""
    name, req, sig_params = request.param
    sp = get_sp(sig_params, req)
    model = models_dict[name](**sp)

    model.eval()
    return model


def get_epochs_y(signal_params=None, n_epochs=10):
    """
    Generate a random dataset with the given signal parameters.
    """
    sp = get_sp(signal_params)
    X = np.random.randn(n_epochs, len(sp["chs_info"]), sp["n_times"])
    y = np.random.randint(sp["n_outputs"], size=n_epochs)
    info = mne.create_info(
        ch_names=[c["ch_name"] for c in sp["chs_info"]],
        sfreq=sp["sfreq"],
        ch_types=["eeg"] * len(sp["chs_info"]),
    )
    for dest, source in zip(info["chs"], sp["chs_info"]):
        if "loc" in source:
            dest["loc"][:] = source["loc"]
    epo = mne.EpochsArray(X, info)
    return epo, y


def test_completeness__models_test_cases():
    models_tested = set(x[0] for x in models_mandatory_parameters)
    all_models = set(models_dict.keys())
    assert (
        all_models == models_tested
    ), f"Models missing from models_test_cases: {all_models - models_tested}"


@pytest.mark.parametrize(
    "model_name, required_params, signal_params", models_mandatory_parameters
)
def test_model_integration(model_name, required_params, signal_params):
    """
    Verifies that all models can be initialized with all their parameters at
    default, except eventually the signal-related parameters.

    This lightly tests if the models will be compatible with the skorch wrappers.
    """
    # Verify that the parameters are correct:
    model_class = models_dict[model_name]
    assert isinstance(required_params, list)
    assert set(required_params) <= {
        "n_times",
        "sfreq",
        "chs_info",
        "n_outputs",
        "input_window_seconds",
        "n_chans",
    }
    assert signal_params is None or isinstance(signal_params, dict)
    if signal_params is not None:
        assert set(signal_params.keys()) <= set(default_signal_params.keys())

    sp = get_sp(signal_params)

    # create input data
    batch_size = 3
    epo, _ = get_epochs_y(sp, n_epochs=batch_size)
    X = torch.tensor(epo.get_data(), dtype=torch.float32)

    # List possible model kwargs:
    output_kwargs = []
    output_kwargs.append(dict(n_outputs=sp["n_outputs"]))

    if "n_outputs" not in required_params:
        output_kwargs.append(dict(n_outputs=None))

    channel_kwargs = []
    channel_kwargs.append(dict(chs_info=sp["chs_info"], n_chans=None))
    if "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=sp["n_chans"], chs_info=None))
    if "n_chans" not in required_params and "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=None, chs_info=None))
    time_kwargs = []
    time_kwargs.append(
        dict(n_times=sp["n_times"], sfreq=sp["sfreq"], input_window_seconds=None)
    )
    time_kwargs.append(
        dict(
            n_times=None,
            sfreq=sp["sfreq"],
            input_window_seconds=sp["input_window_seconds"],
        )
    )
    time_kwargs.append(
        dict(
            n_times=sp["n_times"],
            sfreq=None,
            input_window_seconds=sp["input_window_seconds"],
        )
    )
    if "n_times" not in required_params and "sfreq" not in required_params:
        time_kwargs.append(
            dict(
                n_times=None,
                sfreq=None,
                input_window_seconds=sp["input_window_seconds"],
            )
        )
    if (
        "n_times" not in required_params
        and "input_window_seconds" not in required_params
    ):
        time_kwargs.append(
            dict(n_times=None, sfreq=sp["sfreq"], input_window_seconds=None)
        )
    if "sfreq" not in required_params and "input_window_seconds" not in required_params:
        time_kwargs.append(
            dict(n_times=sp["n_times"], sfreq=None, input_window_seconds=None)
        )
    if (
        "n_times" not in required_params
        and "sfreq" not in required_params
        and "input_window_seconds" not in required_params
    ):
        time_kwargs.append(dict(n_times=None, sfreq=None, input_window_seconds=None))
    model_kwargs_list = [
        dict(**o, **c, **t)
        for o in output_kwargs
        for c in channel_kwargs
        for t in time_kwargs
    ]

    for model_kwargs in model_kwargs_list:
        # test initialisation:
        model = model_class(**model_kwargs)
        # test forward pass:
        out = model(X)

        # Skip the output shape test for non-classification models
        if model_name in non_classification_models:
            continue

        # test output shape
        assert out.shape[:2] == (batch_size, sp["n_outputs"])
        # We add a "[:2]" because some models return a 3D tensor.


@pytest.mark.parametrize(
    "model_name, required_params, signal_params", models_mandatory_parameters
)
def test_model_integration_full(model_name, required_params, signal_params):
    """
    Full test of the models compatibility with the skorch wrappers.
    In particular, it tests if the wrappers can set the signal-related parameters
    and if the model can be found by name.

    Parameters
    ----------
    model_name : str
        The name of the model to test.
    required_params : list[str]
        The signal-related parameters that are needed to initialize the model.
    signal_params : dict | None
        The characteristics of the signal that should be passed to the model tested
        in case the default_signal_params are not compatible with this model.
        The keys of this dictionary can only be among those of default_signal_params.

    """
    if model_name in non_classification_models:
        pytest.skip(f"Skipping {model_name} as not meant for classification")

    epo, y = get_epochs_y(signal_params, n_epochs=10)

    LEARNING_RATE = 0.0625 * 0.01
    BATCH_SIZE = 2
    EPOCH = 1
    seed = 2409
    valid_split = 0.2

    clf = EEGClassifier(
        module=model_name,
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCH,
        classes=[0, 1],
        train_split=ValidSplit(valid_split, random_state=seed),
        verbose=0,
    )

    clf.fit(X=epo, y=y)


@pytest.mark.parametrize(
    "model_name, required_params, signal_params", models_mandatory_parameters
)
def test_model_integration_full_last_layer(model_name, required_params, signal_params):
    """
    Test that the last layers of the model include a layer named 'final_layer'.

    This test iterates over various models defined in `models_mandatory_parameters`
    to ensure that each model has a layer named 'final_layer' among its last two layers.
    Models that only support cropped datasets are skipped.

    Parameters
    ----------
    model_name : str
        Name of the model to be tested.
    required_params : dict
        Required parameters for the model.
    signal_params : dict
        Parameters related to the input signals.

    Raises
    ------
    AssertionError
        If 'final_layer' is not found among the last two layers of the model.

    """
    if model_name in non_classification_models:
        pytest.skip(f"Skipping {model_name} as not meant for classification")

    epo, y = get_epochs_y(signal_params, n_epochs=10)

    LEARNING_RATE = 0.0625 * 0.01
    BATCH_SIZE = 2
    EPOCH = 1
    seed = 2409
    valid_split = 0.2

    clf = EEGClassifier(
        module=model_name,
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCH,
        classes=[0, 1],
        train_split=ValidSplit(valid_split, random_state=seed),
        verbose=0,
    )

    clf.fit(X=epo, y=y)
    last_layers_name = list(clf.module_.named_children())[-2:]

    assert len([name for name, _ in last_layers_name if name == "final_layer"]) > 0


@pytest.mark.parametrize("model_class", models_dict.values())
def test_model_has_activation_parameter(model_class):
    """
    Test that checks if the model class's __init__ method has a parameter
    named 'activation' or any parameter that starts with 'activation'.
    """
    if model_class in [EEGMiner]:
        pytest.skip(f"Skipping {model_class} as not activation layer")
    # Get the __init__ method of the class
    init_method = model_class.__init__

    # Get the signature of the __init__ method
    sig = inspect.signature(init_method)

    # Get the parameter names, excluding 'self'
    param_names = [param_name for param_name in sig.parameters if param_name != "self"]

    # Check if any parameter name contains 'activation'
    has_activation_param = any("activation" in name for name in param_names)

    # Assert that the activation parameter exists
    assert has_activation_param, (
        f"{model_class.__name__} does not have an activation parameter."
        f" Found parameters: {param_names}"
    )


@pytest.mark.parametrize("model_class", models_dict.values())
def test_activation_default_parameters_are_nn_module_classes(model_class):
    """
    Test that checks if all parameters with default values in the model class's
    __init__ method are nn.Module classes and not initialized instances.
    """
    if model_class in [EEGMiner]:
        pytest.skip(f"Skipping {model_class} as not activation layer")

    init_method = model_class.__init__

    sig = inspect.signature(init_method)

    # Filtering parameters with 'activation' in their names
    activation_list = [
        value for key, value in sig.parameters.items() if "activation" in key.lower()
    ]
    for activation in activation_list:

        assert issubclass(activation.default, nn.Module), (
            f"In class {model_class.__name__}, parameter has a default value "
            f"that is an initialized nn.Module instance. Default values should be nn.Module "
            f"classes (like nn.ReLU), not instances (like nn.ReLU())."
        )


@pytest.mark.parametrize("model_class", models_dict.values())
def test_model_has_drop_prob_parameter(model_class):
    """
    Test that checks if the model class's __init__ method has a parameter
    named 'drop_prob' or any parameter that starts with 'activation'.
    """

    if model_class in [
        SyncNet,
        EEGSimpleConv,
        EEGResNet,
        USleep,
        EEGMiner,
        EEGInceptionMI,
        FBCNet,
        FBMSNet,
        FBLightConvNet,
    ]:
        pytest.skip(f"Skipping {model_class} as not dropout layer")

    # Get the __init__ method of the class
    init_method = model_class.__init__

    # Get the signature of the __init__ method
    sig = inspect.signature(init_method)

    # Get the parameter names, excluding 'self'
    param_names = [param_name for param_name in sig.parameters if param_name != "self"]

    # Check if any parameter name contains 'activation'
    has_drop_prob_param = any("drop_prob" in name for name in param_names)

    # Assert that the activation parameter exists
    assert has_drop_prob_param, (
        f"{model_class.__name__} does not have an drop_prob parameter."
        f" Found parameters: {param_names}"
    )


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="torch.compile is known to have issues on Windows.",
)
def test_model_compiled(model):
    """
    Verifies that all models can be torch compiled without issue
    and if the outputs are the same.
    """
    # This assumes the model has attributes n_chans and n_times
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = default_signal_params["n_chans"]
    try:
        n_times = model.n_times
    except ValueError:
        n_times = default_signal_params["n_times"]
    input_tensor = torch.randn(1, n_chans, n_times)
    not_compiled_model = model
    compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    torch.compiler.reset()

    output = not_compiled_model(input_tensor)
    output_compiled = compiled_model(input_tensor)

    assert output.shape == output_compiled.shape
    assert output_compiled.allclose(output, atol=1e-4)


def test_model_exported(model):
    """
    Verifies that all models can be torch export without issue
    using torch.export.export()
    """
    # example input matching your model’s expected shape
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = default_signal_params["n_chans"]
    try:
        n_times = model.n_times
    except ValueError:
        n_times = default_signal_params["n_times"]
    example_input = torch.randn(1, n_chans, n_times)

    # this will raise if the model isn’t fully traceable
    exported_prog: ExportedProgram = export(model, args=(example_input,), strict=False)

    # sanity check: we got the right return type
    assert isinstance(exported_prog, ExportedProgram)

@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="torch.script is known to have issues on Windows.",
)
def test_model_torch_script(model):
    """
    Verifies that all models can be torch export without issue
    using torch.export.export()
    """

    not_working_models = [
        "BIOT",
        "Labram",
        "EEGMiner",
    ]

    if model.__class__.__name__ in not_working_models:
        pytest.skip(
            f"Skipping {model.__class__.__name__} as not working with torchscript"
        )

    final_plain_model = convert_model_to_plain(model)
    final_plain_model.eval()

    # example input matching your model’s expected shape
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = default_signal_params["n_chans"]
    try:
        n_times = model.n_times
    except ValueError:
        n_times = default_signal_params["n_times"]
    input_tensor = torch.randn(1, n_chans, n_times)

    output_model = model(input_tensor)
    output_model_recreated = final_plain_model(input_tensor)
    assert output_model.shape == output_model_recreated.shape
    
    torch.testing.assert_close(output_model, output_model_recreated)
    # convert the new model to scripted
    scripted_model = torch.jit.script(final_plain_model)

    fname = f"{model.__class__.__name__}_scripted.pt"
    scripted_model.save(fname)

    os.remove(fname)
    # now that we can save, 
    # erasing the model from the memory
    # 
     
    # print(f"Model {model_class.__name__} passed the test.")
    # Continue this tests later. Not now...
    # output_script = scripted_model(input_tensor)
    # assert output_script.shape == output_model.shape
    # torch.testing.assert_close(output_script, output_model)


@pytest.mark.parametrize("model_class", models_dict.values())
def test_completeness_summary_table(model_class):

    assert model_class.__name__ in _summary_table.index, (
        f"{model_class.__name__} is not in the summary table. "
        f"Please add it to the summary table."
    )

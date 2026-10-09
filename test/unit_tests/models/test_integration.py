# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Alexandre Gramfort
#          Pierre Guetschel
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3
from __future__ import annotations

import inspect
import sys
from io import BytesIO

import mne
import numpy as np
import pytest
import torch
from skorch.dataset import ValidSplit
from torch import nn
from torch.export import ExportedProgram, export

from braindecode import EEGClassifier
from braindecode.models import (
    EEGPT,
    MAPA,
    REVE,
    SSTDPN,
    ZUNA,
    BrainTokenizer,
    Brant,
    EEGInceptionMI,
    EEGMiner,
    EEGSimpleConv,
    FBCNet,
    FBLightConvNet,
    FBMSNet,
    ShallowFBCSPNet,
    SyncNet,
    USleep,
)
from braindecode.models.util import (
    _get_possible_signal_params,
    _summary_table,
    default_signal_params,
    models_dict,
    models_mandatory_parameters,
    non_classification_models,
)
from braindecode.models.util import _get_signal_params as get_sp
from test.unit_tests.models.test_model_contract import _record

rng = np.random.default_rng(12)

all_models_dict = dict(models_dict)

# First blocker of each model that torch.jit.script cannot compile, or whose
# scripted output differs from the eager one.
_TORCHSCRIPT_XFAIL = {
    "BIOT": "linear_attention_transformer forward takes **kwargs",
    "MAPA": "f-string error message in forward",
    "MVPFormer": "math.log2 in forward",
    "MetaNeuromotorHand": "einops.pack call",
    "NeuroRVQTokenizer": "getattr with a computed name",
    "REVE": "einops.rearrange call (**axes_lengths)",
    "SleepFM": "einops.rearrange call (**axes_lengths)",
    "SleepFMStager": "einops.rearrange call (**axes_lengths)",
    "TFMTokenizer": "linear_attention_transformer forward takes **kwargs",
}

_MODEL_CASES = {
    name: (required, signal_params)
    for name, required, signal_params in models_mandatory_parameters
}


@pytest.fixture(scope="module", params=models_mandatory_parameters, ids=lambda p: p[0])
def model(request):
    """Instantiated model."""
    name, req, sig_params = request.param
    sp = get_sp(sig_params, req)
    model = all_models_dict[name](**sp)

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
    all_models = set(all_models_dict.keys())
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
    model_class = all_models_dict[model_name]
    assert isinstance(required_params, list)
    assert set(required_params) <= {
        "n_times",
        "sfreq",
        "chs_info",
        "n_outputs",
        "input_window_seconds",
        "n_chans",
    }
    # signal_params may be a callable (used to defer imports that would cause
    # circular dependencies at module-load time); resolve it before checking.
    if callable(signal_params):
        signal_params = signal_params()
    assert signal_params is None or isinstance(signal_params, dict)
    if signal_params is not None:
        assert set(signal_params.keys()) <= set(default_signal_params.keys())

    sp = get_sp(signal_params)

    # create input data
    batch_size = 3
    epo, _ = get_epochs_y(sp, n_epochs=batch_size)
    X = torch.tensor(epo.get_data(), dtype=torch.float32)

    model_kwargs_list = _get_possible_signal_params(sp, required_params)

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
    and if the model can be found by name, and that the fitted module keeps a
    layer named 'final_layer' among its last two children.

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
    last_layers_name = list(clf.module_.named_children())[-2:]

    assert len([name for name, _ in last_layers_name if name == "final_layer"]) > 0


@pytest.mark.parametrize(
    "model_name, required_params, signal_params", models_mandatory_parameters
)
def test_model_integration_has_final_layer(model_name, required_params, signal_params):
    """Only tests that th model has a layer named 'final_layer'.

    This test should also cover models that are not meant for classification.
    """
    sp = get_sp(signal_params)

    model = all_models_dict[model_name](**sp)

    last_layers_name = [name for name, _ in model.named_children()][-2:]
    assert "final_layer" in last_layers_name


@pytest.mark.parametrize("model_class", all_models_dict.values())
def test_model_has_activation_parameter(model_class):
    """
    Test that checks if the model class's __init__ method has a parameter
    named 'activation' or any parameter that starts with 'activation'.
    """
    if model_class in [
        Brant,
        BrainTokenizer,  # the released SEANet codec has fixed ELU activations
        EEGMiner,
        REVE,
        EEGPT,
    ]:
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


@pytest.mark.parametrize("model_class", all_models_dict.values())
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


@pytest.mark.parametrize("model_class", all_models_dict.values())
def test_model_has_drop_prob_parameter(model_class):
    """
    Test that checks if the model class's __init__ method has a parameter
    named 'drop_prob' or any parameter that starts with 'activation'.
    """

    if model_class in [
        SyncNet,
        EEGSimpleConv,
        USleep,
        EEGMiner,
        EEGInceptionMI,
        FBCNet,
        FBMSNet,
        FBLightConvNet,
        SSTDPN,
        REVE,
        ZUNA,
        MAPA,
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


# skip if windows or python 3.14
@pytest.mark.skipif(
    sys.platform.startswith("win") or sys.version_info >= (3, 14),
    reason="torch.compile is known to have issues on Windows or with Python 3.14.",
)
def test_model_compiled(model):
    """
    Verifies that all models can be torch compiled without issue
    and if the outputs are the same.
    """
    not_compilable_models = [
        # torch.compile currently stalls on the STFT/eigendecomposition-based
        # MPF featurizer at the default handwriting input size.
        "MetaNeuromotorHand",
    ]
    if model.__class__.__name__ in not_compilable_models:
        pytest.skip(
            f"Skipping {model.__class__.__name__} as not working with torch.compile"
        )

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

    # assert_close also walks the tuple outputs (e.g. NeuroRVQTokenizer).
    torch.testing.assert_close(output_compiled, output, atol=1e-4, rtol=1e-5)


def test_model_exported(model):
    """
    Verifies that all models can be torch export without issue
    using torch.export.export()
    """
    # Models known to have export issues on Windows and Python 3.14+ (non-pytree-compatible attributes)
    model_name = model.__class__.__name__

    not_exportable_models = [
        "SyncNet",  # We found a fake tensor in the exported program constant's list.
        "EEGMiner",  # We found a fake tensor in the exported program constant's list.
        "SSTDPN",  # We found a fake tensor in the exported program constant's list.
        "Labram",  # Uses data-dependent channel/patch paths that are not export-stable yet.
        "CodeBrain",  # Data-dependent n_times // patch_size division in forward is not export-stable.
        "NeuroRVQTokenizer",  # EMA codebooks use data-dependent k-means initialization.
    ]
    if sys.platform.startswith("win"):
        not_exportable_models += [
            "LUNA",  # Has _channel_location_cache dict that breaks pytree on Windows
        ]
    if model_name in not_exportable_models:
        pytest.skip(f"{model_name} export is not possible on this platform.")

    if sys.version_info >= (3, 14):
        not_exportable_models_py314 = [
            "LUNA",  # Has _channel_location_cache dict that breaks pytree on Python 3.14+
        ]  # TODO: fix LUNA export issues on Python 3.14+ one day
        if model_name in not_exportable_models_py314:
            pytest.skip(f"{model_name} export is not compatible on Python 3.14+")

    # example input matching your model's expected shape
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = default_signal_params["n_chans"]
    try:
        n_times = model.n_times
    except ValueError:
        n_times = default_signal_params["n_times"]
    example_input = torch.randn(1, n_chans, n_times)

    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        with torch.no_grad():
            _ = model(example_input)

    # this will raise if the model isn't fully traceable
    exported_prog: ExportedProgram = export(model, args=(example_input,), strict=False)

    # sanity check: we got the right return type
    assert isinstance(exported_prog, ExportedProgram)


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(reason=_TORCHSCRIPT_XFAIL[name], strict=True)
            if name in _TORCHSCRIPT_XFAIL
            else (),
        )
        for name in _MODEL_CASES
    ],
)
def test_torch_script(model_name):
    """Models script directly and the scripted output equals the eager one.

    ``EEGModuleMixin`` exposes the signal-related parameters as properties that
    raise ``ValueError`` when unset, and annotates ``mapping`` with a postponed
    ``Optional[Dict[str, str]]``. ``torch.jit.script`` reads every class
    attribute while building the concrete type, so either one used to abort
    scripting before ``forward`` was compiled.
    """
    required, signal_params = _MODEL_CASES[model_name]
    sp = get_sp(signal_params, required)
    model = all_models_dict[model_name](**sp).eval()

    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = default_signal_params["n_chans"]
    try:
        n_times = model.n_times
    except ValueError:
        n_times = default_signal_params["n_times"]
    input_tensor = torch.randn(1, n_chans, n_times)

    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        with torch.no_grad():
            model(input_tensor)

    with torch.no_grad():
        expected = model(input_tensor)

    scripted_model = torch.jit.script(model)
    with torch.no_grad():
        torch.testing.assert_close(scripted_model(input_tensor), expected)

    buffer = BytesIO()
    torch.jit.save(scripted_model, buffer)
    buffer.seek(0)
    restored_script = torch.jit.load(buffer)

    with torch.no_grad():
        torch.testing.assert_close(restored_script(input_tensor), expected)


@pytest.mark.parametrize("n_chans", [None, default_signal_params["n_chans"]])
def test_torch_script_with_chs_info(n_chans):
    """MNE-style channel metadata must not be part of the scripted graph."""
    chs_info = default_signal_params["chs_info"]
    model = ShallowFBCSPNet(
        n_chans=n_chans,
        chs_info=chs_info,
        n_outputs=default_signal_params["n_outputs"],
        n_times=default_signal_params["n_times"],
        sfreq=default_signal_params["sfreq"],
    ).eval()
    input_tensor = torch.randn(
        2, len(chs_info), default_signal_params["n_times"]
    )
    assert model._n_chans == n_chans
    assert model.chs_info is chs_info
    expected = model(input_tensor)

    scripted = torch.jit.script(model)
    buffer = BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    restored_script = torch.jit.load(buffer)

    torch.testing.assert_close(scripted(input_tensor), expected)
    torch.testing.assert_close(restored_script(input_tensor), expected)

    restored_model = ShallowFBCSPNet.from_config(model.get_config()).eval()
    restored_model.load_state_dict(model.state_dict())
    assert restored_model._n_chans == n_chans
    assert restored_model.n_chans == len(chs_info)
    assert [ch["ch_name"] for ch in restored_model.chs_info] == [
        ch["ch_name"] for ch in chs_info
    ]
    for restored_ch, original_ch in zip(restored_model.chs_info, chs_info):
        np.testing.assert_allclose(restored_ch["loc"], original_ch["loc"])
    torch.testing.assert_close(restored_model(input_tensor), expected)


def test_signal_params_still_raise_value_error():
    """Hiding the properties from TorchScript must not silence their errors."""
    model = ShallowFBCSPNet(
        n_chans=default_signal_params["n_chans"],
        n_outputs=default_signal_params["n_outputs"],
        n_times=default_signal_params["n_times"],
    )
    with pytest.raises(ValueError, match="chs_info not specified"):
        model.chs_info
    with pytest.raises(ValueError, match="sfreq could not be inferred"):
        model.sfreq


def test_torch_script_int_signal_params():
    """Int ``sfreq``/``input_window_seconds``, as in Hub configs, still script."""
    model = ShallowFBCSPNet(n_chans=3, n_outputs=2, input_window_seconds=4, sfreq=250)
    model.eval()
    x = torch.randn(2, 3, 1000)
    torch.testing.assert_close(torch.jit.script(model)(x), model(x))


@pytest.mark.parametrize("method", ["mag", "corr", "plv"])
def test_eegminer_torch_script_methods(method):
    model = EEGMiner(
        method=method,
        n_chans=4,
        n_outputs=2,
        n_times=128,
        sfreq=100.0,
    ).eval()
    input_tensor = torch.randn(2, 4, 128)

    expected = model(input_tensor)
    scripted_model = torch.jit.script(model)

    torch.testing.assert_close(scripted_model(input_tensor), expected)


@pytest.mark.parametrize("model_class", all_models_dict.values())
def test_completeness_summary_table(model_class):

    assert model_class.__name__ in _summary_table.index, (
        f"{model_class.__name__} is not in the summary table. "
        f"Please add it to the summary table."
    )


def test_if_models_with_embedding_parameter(model):
    # Test if the model that have embedding parameters works changing the default
    # embedding value
    model_name = model.__class__.__name__
    # first step is to inspect the models parameters
    params = inspect.signature(model.__init__).parameters

    # check if there is any mention to emb_size or embedding_dim or emb
    # or return_features or feature

    parameters_related_with_emb = [
        "emb_size",
        "embedding_dim",
        "emb",
        "return_features",
        "feature",
        "embed_dim",
        "emb_dim",
    ]

    # check if any of the parameters related to embedding are present in the model
    if not any(param in params for param in parameters_related_with_emb):
        pytest.skip(f"{model_name} does not have an embedding parameter.")
    else:
        # getting the parameters name
        emb_param = next(
            (param for param in params if param in parameters_related_with_emb), None
        )

    # getting the emb parameter and re-initializing the model
    # changing only this parameters
    new_value = 128

    model_dict = {
        name: (args, defaults) for name, args, defaults in models_mandatory_parameters
    }

    mandatory_parameters, sig_params = model_dict[model_name]
    signal_params = get_sp(sig_params, mandatory_parameters)
    # redefining the embedding parameter
    signal_params[emb_param] = new_value

    model = all_models_dict[model_name](**signal_params)

    print(
        f"Model {model_name} has an embedding parameter {emb_param} with value {new_value}."
    )
    # assert not error when print the model
    try:
        print(model)
    except Exception as e:
        pytest.fail(f"Error printing model {model_name}: {e}")


@pytest.mark.parametrize(
    "model_name, dtype",
    [
        pytest.param(name, dtype, id=f"{name}-{str(dtype)[6:]}")
        for name in _MODEL_CASES
        for dtype in (torch.float64, torch.bfloat16, torch.float16)
    ],
)
def test_forward_in_dtype(model_name, dtype):
    """``model.to(dtype)`` forwards on CPU in ``dtype`` (and backpropagates in float64)."""
    required, signal_params = _MODEL_CASES[model_name]
    model = all_models_dict[model_name](**get_sp(signal_params, required)).eval()
    try:
        n_chans = model.n_chans
    except ValueError:
        n_chans = default_signal_params["n_chans"]
    try:
        n_times = model.n_times
    except ValueError:
        n_times = default_signal_params["n_times"]
    x = torch.randn(2, n_chans, n_times)
    with torch.no_grad():
        model(x)  # materialise lazy modules in float32
    # Backward only in float64: CPUs without bfloat16/float16 kernels take
    # minutes per model (DANCE: ~700 s).
    backward = dtype == torch.float64
    with torch.set_grad_enabled(backward), _record() as log:
        y = model.to(dtype)(x.to(dtype))
    if isinstance(y, tuple):  # NeuroRVQTokenizer: (target, reconstruction)
        y = y[1]
    y = y if torch.is_tensor(y) else next(iter(y.values()))
    assert y.dtype == dtype and torch.isfinite(y).all()
    if not backward:
        # no bfloat16/float16 cdist kernel (modules.quantization, #1246)
        cdist = [op for op in log.ops if op.name == "_cdist_forward"]
        assert all(t[0] != dtype for op in cdist for t in op.ins)
        return
    # Every floating tensor follows the model dtype: a float32 one comes from
    # torch.zeros/full/tensor without dtype= or a hard-coded .float() (0-dim
    # ones do not lower the dtype of the tensors they meet).
    float32 = sorted(
        {
            op.name
            for op in log.ops
            if any(t[0] == torch.float32 and t[2] for t in op.outs)
        }
    )
    assert not float32, f"float32 tensors in a float64 forward: {float32}"
    y.float().square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.dtype == dtype and torch.isfinite(g).all() for g in grads)

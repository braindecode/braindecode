# Authors: Kuntal Kokate
#          Bruno Aristimunha
#
# License: BSD-3

"""
Tests for Hugging Face Hub integration with braindecode models.

These tests verify that models can be pushed to and loaded from the
Hugging Face Hub when the optional huggingface_hub dependency is installed.
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from braindecode.models import EEGNet
from braindecode.models.base import HAS_HF_HUB, EEGModuleMixin

# importing some fixtures/utilities to help with testing
from braindecode.models.util import models_dict, models_mandatory_parameters
from test.unit_tests.models._fixtures import non_classification_models

from .test_integration import get_sp

all_models_dict = dict(models_dict)

# Dense pose regressors return tensors and support the same local Hub
# round-trip as classifiers. Other non-classification models remain skipped
# until their task-specific output/state contracts are covered here.
_HF_DENSE_POSE_MODELS = {"NeuroPose", "SensingDynamics", "VEMG2Pose"}
_HF_UNSUPPORTED_NON_CLASSIFICATION_MODELS = set(non_classification_models).difference(
    _HF_DENSE_POSE_MODELS
)

# Skip all tests in this file if huggingface_hub is not installed
pytestmark = pytest.mark.skipif(
    not HAS_HF_HUB,
    reason="huggingface_hub not installed. Install with: pip install braindecode[hug]"
)

@pytest.fixture(scope="module", params=models_mandatory_parameters, ids=lambda p: p[0])
def sample_model(request):
    """Instantiated model."""
    name, req, sig_params = request.param
    sp = get_sp(sig_params, req)
    model = all_models_dict[name](**sp)

    model.eval()
    return model, name, sp


@pytest.fixture
def sample_chs_info():
    """Create sample channel information."""
    return [
        {
            'ch_name': f'EEG {i:03d}',
            'coil_type': 0,
            'kind': 2,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': np.random.randn(12),
        }
        for i in range(22)
    ]


def test_models_work_without_hf_hub():
    """Ensure soft dependencies do not break model usage."""
    model = EEGNet(n_chans=22, n_outputs=4, n_times=1000)
    assert model is not None
    assert isinstance(model, EEGModuleMixin)


def test_has_hf_hub_flag_enabled():
    """Verify the HAS_HF_HUB flag matches the skip condition."""
    assert HAS_HF_HUB is True


def test_serialize_none_returns_none():
    result = EEGModuleMixin._serialize_chs_info(None)
    assert result is None


def test_deserialize_none_returns_none():
    result = EEGModuleMixin._deserialize_chs_info(None)
    assert result is None


def test_serialize_chs_info(sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)

    assert serialized is not None
    assert len(serialized) == len(sample_chs_info)
    assert isinstance(serialized, list)

    first_ch = serialized[0]
    assert 'ch_name' in first_ch
    assert 'coil_type' in first_ch
    assert 'kind' in first_ch
    assert 'unit' in first_ch
    assert 'cal' in first_ch
    assert 'range' in first_ch
    assert 'loc' in first_ch
    assert isinstance(first_ch['loc'], list)


def test_deserialize_chs_info(sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)
    deserialized = EEGModuleMixin._deserialize_chs_info(serialized)

    assert deserialized is not None
    assert len(deserialized) == len(sample_chs_info)
    assert isinstance(deserialized[0]['loc'], np.ndarray)


def test_roundtrip_serialization(sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)
    deserialized = EEGModuleMixin._deserialize_chs_info(serialized)

    for orig, deser in zip(sample_chs_info, deserialized):
        assert orig['ch_name'] == deser['ch_name']
        assert orig['coil_type'] == deser['coil_type']
        assert orig['kind'] == deser['kind']
        assert orig['unit'] == deser['unit']
        assert np.allclose(orig['loc'], deser['loc'])


def test_json_serialization(tmp_path, sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)

    temp_path = tmp_path / 'chs_info.json'
    with open(temp_path, 'w') as tmp_file:
        json.dump({'chs_info': serialized}, tmp_file)

    with open(temp_path, 'r') as file:
        loaded = json.load(file)

    assert 'chs_info' in loaded
    assert len(loaded['chs_info']) == len(sample_chs_info)


def test_save_pretrained_creates_config(tmp_path, sample_model):
    sample_model, name, _ = sample_model
    # TODO: fix for AttnSleep
    if name in _HF_UNSUPPORTED_NON_CLASSIFICATION_MODELS | {"AttnSleep"}:
        pytest.skip(f"Skipping config test for non-classification model: {name}")

    if any(isinstance(p, nn.UninitializedParameter) for p in sample_model.parameters()):
        pytest.skip(f"Skipping config test for model with uninitialized parameters: {name}")

    sample_model._save_pretrained(tmp_path)

    config_path = tmp_path / 'config.json'
    assert config_path.exists()

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Use getattr with exception handling to safely get model properties
    # Some models may raise ValueError when properties are not properly initialized
    def safe_get_property(obj, prop_name):
        try:
            return getattr(obj, prop_name)
        except (ValueError, AttributeError):
            return None

    n_times = safe_get_property(sample_model, 'n_times')
    n_chans = safe_get_property(sample_model, 'n_chans')
    n_outputs = safe_get_property(sample_model, 'n_outputs')

    if n_chans is not None and name != "SignalJEPA_Contextual":
        if name == "Labram":
            assert config['n_chans'] in (n_chans, None)
        elif config.get('n_chans') is None and config.get('chs_info') is not None:
            # Models built from chs_info may store it instead of n_chans
            # (n_chans may be absent OR present-but-null in the saved config)
            assert len(config['chs_info']) == n_chans
        else:
            assert config['n_chans'] == n_chans
    if n_times is not None:
        assert config['n_times'] == n_times
    if n_outputs is not None:
        if name == "Labram":
            assert config['n_outputs'] in (n_outputs, None)
        else:
            assert config['n_outputs'] == n_outputs




def test_config_contains_all_parameters(tmp_path, sample_model):
    sample_model, _, _ = sample_model

    if any(isinstance(p, nn.UninitializedParameter) for p in sample_model.parameters()):
        pytest.skip("Skipping config parameter test for model with uninitialized parameters.")

    sample_model._save_pretrained(tmp_path)

    config_path = tmp_path / 'config.json'
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # The config must contain braindecode_version and at least one key that
    # identifies the channel space.  Since all __init__ parameters are saved,
    # model-specific keys (including EEG-specific ones) will also be present
    # whenever the constructor accepts them.
    assert 'braindecode_version' in config, "Config must contain 'braindecode_version'"
    assert 'n_chans' in config or 'chs_info' in config, (
        "Config must contain either 'n_chans' or 'chs_info'"
    )


def test_local_push_and_pull_roundtrip(tmp_path, sample_model):
    """Roundtrip through local Hub save/load mimics push/pull."""
    model, name, sp = sample_model
    # TODO: fix for AttnSleep
    if name in _HF_UNSUPPORTED_NON_CLASSIFICATION_MODELS | {"AttnSleep"}:
        pytest.skip(f"Skipping Hugging Face Hub test for non-classification model: {name}")
    assert hasattr(model, 'from_pretrained')
    assert callable(getattr(model, 'from_pretrained'))
    model.eval()

    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        pytest.skip(f"Skipping Hugging Face Hub test for model with uninitialized parameters: {name}")

    repo_dir = tmp_path / f'hf_local_repo_{name}'
    repo_dir.mkdir()
    model._save_pretrained(repo_dir)

    # Load the model from the saved config using the class method
    model_class = all_models_dict[name]
    restored = model_class.from_pretrained(repo_dir)
    restored.eval()

    n_times = sp.get('n_times', 1000)
    n_chans = sp.get('n_chans')
    if n_chans is None and sp.get('chs_info') is not None:
        n_chans = len(sp['chs_info'])
    if n_chans is None:
        n_chans = 22
    # TODO: small adjust necessary for SignalJEPA_Contextual
    if name == "SignalJEPA_Contextual":
        n_chans = 3
    torch.manual_seed(42)

    sample_input = torch.randn(2, n_chans, n_times)

    out_original = model(sample_input)

    out_restored = restored(sample_input)
    torch.testing.assert_close(out_restored, out_original)


def test_serialize_chs_info_with_string_kind():
    """Test serialization when kind field is a string."""
    chs_info = [
        {
            'ch_name': 'EEG 001',
            'kind': 'eeg',  # String instead of int
            'coil_type': 0,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': np.array([0.0] * 12),
        }
    ]
    serialized = EEGModuleMixin._serialize_chs_info(chs_info)
    assert serialized[0]['kind'] == 'eeg'


def test_serialize_chs_info_without_optional_fields():
    """Test serialization when optional fields are missing."""
    chs_info = [
        {
            'ch_name': 'EEG 001',
            # Missing coil_type, unit, cal, range, loc, kind
        }
    ]
    serialized = EEGModuleMixin._serialize_chs_info(chs_info)
    assert serialized[0]['ch_name'] == 'EEG 001'
    # Optional fields should not be in the serialized output
    assert 'coil_type' not in serialized[0] or serialized[0].get('coil_type') is None


def test_init_with_serialized_chs_info():
    """Test model initialization with serialized channel info from Hub."""
    serialized_chs_info = [
        {
            'ch_name': 'EEG 001',
            'kind': 2,
            'coil_type': 0,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # List, not ndarray
        }
    ]
    # Model should deserialize this automatically in __init__
    model = EEGNet(n_chans=1, n_outputs=2, n_times=100, chs_info=serialized_chs_info)
    # Check that chs_info was deserialized (loc should be ndarray now)
    assert isinstance(model.chs_info[0]['loc'], np.ndarray)


def test_init_with_already_deserialized_chs_info():
    """Test model initialization with already-deserialized channel info."""
    deserialized_chs_info = [
        {
            'ch_name': 'EEG 001',
            'kind': 2,
            'coil_type': 0,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': np.array([0.0] * 12),  # Already ndarray
        }
    ]
    # Model should handle this without error
    model = EEGNet(n_chans=1, n_outputs=2, n_times=100, chs_info=deserialized_chs_info)
    assert isinstance(model.chs_info[0]['loc'], np.ndarray)


def test_save_pretrained_without_hf_hub():
    """Test that _save_pretrained returns early when HF Hub is not available."""
    from braindecode.models import base
    original_has_hf_hub = base.HAS_HF_HUB
    try:
        # Temporarily set HAS_HF_HUB to False
        base.HAS_HF_HUB = False
        model = EEGNet(n_chans=22, n_outputs=4, n_times=1000)
        # Should return early without error
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model._save_pretrained(tmpdir)
    finally:
        base.HAS_HF_HUB = original_has_hf_hub

def test_init_subclass_without_hf_hub():
    """Test that __init_subclass__ works when HF Hub is not available."""
    from braindecode.models import base
    original_has_hf_hub = base.HAS_HF_HUB
    try:
        base.HAS_HF_HUB = False
        # Creating a subclass should work without HF Hub
        class TestModel(base.EEGModuleMixin):
            pass
        # Should complete without error
        assert TestModel is not None
    finally:
        base.HAS_HF_HUB = original_has_hf_hub


@pytest.mark.parametrize("filename", ["", "../outside.safetensors", "/outside.safetensors", "C:/outside.safetensors", "C:outside.safetensors", "..\\outside.safetensors", "a/../outside.safetensors", "a//b", "a/./b", "a/", "bad\x00name", 42])
def test_custom_weights_reject_invalid_paths(tmp_path, monkeypatch, filename):
    from huggingface_hub import PyTorchModelHubMixin

    def unexpected(*args, **kwargs):
        pytest.fail("invalid filename reached the weights loader")

    monkeypatch.setattr(PyTorchModelHubMixin, "_from_pretrained", unexpected)
    with pytest.raises(ValueError, match="filename must be"):
        EEGNet.from_pretrained(tmp_path, filename=filename, local_files_only=True)


@pytest.mark.parametrize("filename", [None, "custom.safetensors", "nested/custom.safetensors"])
def test_local_weights_reject_external_symlinks(tmp_path, filename):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not weights")
    target = repo / (filename or "model.safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="inside the model directory"):
        EEGNet.from_pretrained(repo, filename=filename, local_files_only=True)


@pytest.mark.parametrize("filename", ["custom.safetensors", "nested/custom.safetensors"])
def test_custom_weights_roundtrip_and_global_isolation(tmp_path, monkeypatch, filename):
    import huggingface_hub
    from huggingface_hub import PyTorchModelHubMixin

    from .test_base import DummyModuleConfigRoundTrip

    model = DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2).eval()
    model.save_pretrained(tmp_path)
    custom = tmp_path / filename
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_bytes((tmp_path / "model.safetensors").read_bytes())
    original_name = huggingface_hub.constants.SAFETENSORS_SINGLE_FILE
    original_loader = PyTorchModelHubMixin._load_as_safetensor
    seen = []

    def load(cls, instance, model_file, map_location, strict):
        # Observe the value *during* loading, not only after it is restored.
        assert huggingface_hub.constants.SAFETENSORS_SINGLE_FILE == original_name
        seen.append(str(model_file))
        return original_loader(instance, model_file, map_location, strict)

    monkeypatch.setattr(DummyModuleConfigRoundTrip, "_load_as_safetensor", classmethod(load))
    restored = DummyModuleConfigRoundTrip.from_pretrained(tmp_path, filename=filename, strict=True).eval()
    default = DummyModuleConfigRoundTrip.from_pretrained(tmp_path, strict=True).eval()
    x = torch.randn(2, 2, 8)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    torch.testing.assert_close(default(x), model(x), rtol=0, atol=0)
    assert seen == [str(custom.resolve()), str(tmp_path / "model.safetensors")]
    assert huggingface_hub.constants.SAFETENSORS_SINGLE_FILE == original_name


@pytest.mark.parametrize("download_options", [{}, {"proxies": {"https": "http://proxy.invalid"}, "resume_download": True}])
def test_custom_weights_download_arguments_and_failure_isolation(tmp_path, monkeypatch, download_options):
    import huggingface_hub

    original_name = huggingface_hub.constants.SAFETENSORS_SINGLE_FILE
    calls = []

    def download(**kwargs):
        assert huggingface_hub.constants.SAFETENSORS_SINGLE_FILE == original_name
        calls.append(kwargs)
        if kwargs["filename"] == "config.json":
            raise FileNotFoundError
        raise RuntimeError("local download stub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="local download stub"):
        EEGNet._from_pretrained(
            model_id="test/model", filename="nested/custom.safetensors",
            revision="fixed-revision", cache_dir=tmp_path, force_download=True,
            local_files_only=True, token=False, **download_options,
        )
    assert calls[-1] == dict(repo_id="test/model", filename="nested/custom.safetensors", revision="fixed-revision", cache_dir=tmp_path, force_download=True, local_files_only=True, token=False, **download_options)
    assert huggingface_hub.constants.SAFETENSORS_SINGLE_FILE == original_name


@pytest.mark.parametrize("second_filename", ["second.safetensors", None])
def test_concurrent_custom_weights_keep_independent_filenames(tmp_path, monkeypatch, second_filename):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import huggingface_hub

    from .test_base import DummyModuleConfigRoundTrip

    original = huggingface_hub.constants.SAFETENSORS_SINGLE_FILE
    barrier = Barrier(2)

    def load(cls, model, model_file, map_location, strict):
        barrier.wait(timeout=10)
        assert huggingface_hub.constants.SAFETENSORS_SINGLE_FILE == original
        return model_file

    monkeypatch.setattr(DummyModuleConfigRoundTrip, "_load_as_safetensor", classmethod(load))

    def load_one(filename):
        return DummyModuleConfigRoundTrip._from_pretrained(
            model_id=str(tmp_path), filename=filename, revision=None,
            cache_dir=None, force_download=False, local_files_only=True,
            token=None, n_chans=2, n_times=8, n_outputs=2,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(load_one, ["first.safetensors", second_filename]))
    assert paths == [str(tmp_path / name) for name in ("first.safetensors", second_filename or "model.safetensors")]
    assert huggingface_hub.constants.SAFETENSORS_SINGLE_FILE == original


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/outside.safetensors", "C:/outside.safetensors"])
def test_custom_weights_reject_filename_from_saved_config(tmp_path, monkeypatch, filename):
    """Validate config-supplied filenames without requiring caller overrides."""
    from huggingface_hub import PyTorchModelHubMixin

    config = dict(n_chans=2, n_times=256, n_outputs=2, filename=filename)
    (tmp_path / "config.json").write_text(json.dumps(config))

    def unexpected(*args, **kwargs):
        pytest.fail("config filename reached the weights loader")

    monkeypatch.setattr(PyTorchModelHubMixin, "_from_pretrained", unexpected)
    monkeypatch.setattr(EEGNet, "_load_as_safetensor", unexpected)
    with pytest.raises(ValueError, match="filename must be"):
        EEGNet.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.parametrize("filename", [None, "nested/custom.safetensors"])
def test_local_weights_selected_symlinks_and_native_modes(tmp_path, filename):
    from huggingface_hub import PyTorchModelHubMixin

    from .test_base import DummyModuleConfigRoundTrip

    model = DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2)
    model.save_pretrained(tmp_path)
    # An unused legacy artifact, even a dangling external link, is not selected.
    legacy = tmp_path / "pytorch_model.bin"
    legacy.unlink()
    legacy.symlink_to(tmp_path.parent / "unused.bin")
    target = tmp_path / "weights.safetensors"
    (tmp_path / "model.safetensors").rename(target)
    selected = tmp_path / (filename or "model.safetensors")
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.symlink_to(target)
    restored = DummyModuleConfigRoundTrip.from_pretrained(
        tmp_path, filename=filename, local_files_only=True, strict=True
    )
    expected = PyTorchModelHubMixin._load_as_safetensor(
        DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2),
        str(target), "cpu", True,
    )
    # Do not normalize modes: older and newer Hub loaders differ intentionally.
    assert {n: m.training for n, m in restored.named_modules()} == {
        n: m.training for n, m in expected.named_modules()
    }
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("filename", [None, "missing.safetensors"])
def test_custom_weights_missing_does_not_use_default_pickle_fallback(tmp_path, monkeypatch, filename):
    import huggingface_hub
    from huggingface_hub import hub_mixin
    from huggingface_hub.errors import EntryNotFoundError

    from .test_base import DummyModuleConfigRoundTrip

    calls = []
    loaded = []

    def download(**kwargs):
        name = kwargs["filename"]
        calls.append(name)
        if name == "config.json":
            raise FileNotFoundError
        if name.endswith(".safetensors"):
            raise EntryNotFoundError("missing weights")
        assert name == "pytorch_model.bin"
        return str(tmp_path / name)

    def load(cls, model, model_file, map_location, strict):
        loaded.append((model_file, map_location, strict))
        return model

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(hub_mixin, "hf_hub_download", download)
    monkeypatch.setattr(DummyModuleConfigRoundTrip, "_load_as_pickle", classmethod(load))
    kwargs = dict(
        model_id="test/model", filename=filename, revision="fixed-revision",
        cache_dir=tmp_path, force_download=False, local_files_only=True,
        token=False, n_chans=2, n_times=8, n_outputs=2, strict=True,
    )
    if filename is not None:
        with pytest.raises(EntryNotFoundError, match="missing weights"):
            DummyModuleConfigRoundTrip._from_pretrained(**kwargs)
        assert calls == ["config.json", filename]
        assert loaded == []
    else:
        DummyModuleConfigRoundTrip._from_pretrained(**kwargs)
        assert calls == ["config.json", "model.safetensors", "pytorch_model.bin"]
        assert loaded == [(str(tmp_path / "pytorch_model.bin"), "cpu", True)]


def test_custom_weights_legacy_options_do_not_reach_constructor(tmp_path, monkeypatch):
    import huggingface_hub

    from .test_base import DummyModuleConfigRoundTrip

    model = DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2)
    model.save_pretrained(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        name = "model.safetensors" if kwargs["filename"] == "custom.safetensors" else kwargs["filename"]
        return str(tmp_path / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    restored = DummyModuleConfigRoundTrip._from_pretrained(
        model_id="test/model", filename="custom.safetensors", revision=None,
        cache_dir=None, force_download=False, local_files_only=True, token=None,
        n_chans=2, n_times=8, n_outputs=2, proxies={"https": "http://proxy.invalid"},
        resume_download=True, map_location="cpu", strict=True,
    )
    assert calls[-1]["proxies"] == {"https": "http://proxy.invalid"}
    assert calls[-1]["resume_download"] is True
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

# Authors: Kuntal Kokate
#          Bruno Aristimunha
#
# License: BSD-3

"""
Tests for Hugging Face Hub integration with braindecode models.

These tests verify that models can be pushed to and loaded from the
Hugging Face Hub when the optional huggingface_hub dependency is installed.
"""

import inspect
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
        return DummyModuleConfigRoundTrip.from_pretrained(
            tmp_path, filename=filename, local_files_only=True,
            n_chans=2, n_times=8, n_outputs=2,
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
    # Public Hub loading supplies these required arguments on older versions.
    parent_parameters = inspect.signature(hub_mixin.PyTorchModelHubMixin._from_pretrained).parameters
    kwargs.update({
        name: None for name in ("proxies", "resume_download")
        if name in parent_parameters
    })
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


def test_model_card_datasets_roundtrip(tmp_path):
    from .test_base import DummyModuleConfigRoundTrip

    model = DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2)
    card = {"datasets": ["TUEG", "Siena"], "base_model": "org/parent"}
    model.save_pretrained(tmp_path / "a", model_card_kwargs=card)
    model.save_pretrained(tmp_path / "b")  # card kwargs must not leak

    restored = DummyModuleConfigRoundTrip.from_pretrained(tmp_path / "a")
    assert restored.card_data.datasets == ["TUEG", "Siena"]
    assert restored.card_data.base_model == "org/parent"
    assert "braindecode" in restored.card_data.tags
    assert DummyModuleConfigRoundTrip.from_pretrained(tmp_path / "b").card_data.datasets is None

    (tmp_path / "a" / "README.md").unlink()  # no card: still loads
    assert DummyModuleConfigRoundTrip.from_pretrained(tmp_path / "a").card_data is None


def test_model_card_read_from_hub_cache_without_network(tmp_path, monkeypatch):
    import huggingface_hub

    from braindecode.models.base import _load_card_data

    readme = tmp_path / "README.md"
    readme.write_text("---\ndatasets:\n- PREST\n- SHHS\n---\ncard\n")
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(readme)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    data = _load_card_data("org/repo", revision="v1", local_files_only=True, n_outputs=3)
    assert data.datasets == ["PREST", "SHHS"]
    assert calls == [(("org/repo", "README.md"), {"revision": "v1", "local_files_only": True})]

    monkeypatch.undo()  # real hub, nothing cached, no network: None, no error
    assert _load_card_data("braindecode/no-such-repo", cache_dir=tmp_path / "c", local_files_only=True) is None


def test_pretrained_provenance_in_repr(tmp_path):
    import copy

    from braindecode.models import BIOT

    kwargs = dict(n_chans=4, n_times=400, sfreq=200, embed_dim=16, num_heads=2, num_layers=1)
    model = BIOT(n_outputs=2, **kwargs)
    datasets = ["BNCI2014_001", "ds002718", "braindecode/chbmit"]
    model.save_pretrained(tmp_path, model_card_kwargs={"datasets": datasets})

    built = BIOT(n_outputs=2, **kwargs)
    assert str(built) == str(built.get_torchinfo_statistics())
    assert "pretrained" not in built._repr_html_()

    same = BIOT.from_pretrained(tmp_path)
    assert same._pretrained_provenance["local"]
    params = set(dict(same.named_parameters()))
    assert params <= same._pretrained_provenance["loaded"]
    text = str(same)
    assert "Trained on (model card): BNCI2014_001, ds002718, braindecode/chbmit" in text
    assert "(100.0% of materialised parameters)" in text and "new:" not in text
    page = same._repr_html_()
    assert "https://openneuro.org/datasets/ds002718" in page
    assert "https://huggingface.co/datasets/braindecode/chbmit" in page
    assert "100.0%" in page and "fully checkpoint-loaded tensors" in page

    # A new head is initialised here, not loaded.
    head = BIOT.from_pretrained(tmp_path, n_outputs=3)
    head_params = {f"final_layer.{n}" for n, _ in head.final_layer.named_parameters()}
    loaded = head._pretrained_provenance["loaded"]
    assert head_params and not head_params & loaded
    assert set(dict(head.named_parameters())) - head_params <= loaded
    assert "new: final_layer" in str(head)
    assert "bd-adapt" in head._repr_html_() and "Train before use" in head._repr_html_()

    restored = copy.deepcopy(head)
    assert restored._pretrained_provenance["loaded"] == loaded
    assert restored._repr_html_() == head._repr_html_()


def test_pretrained_provenance_reads_hub_cache_commit(tmp_path, monkeypatch):
    import huggingface_hub

    from braindecode.models.base import _checkpoint_source

    commit = "0123456789abcdef0123456789abcdef01234567"
    cached = tmp_path / "snapshots" / "hf-cache" / "models--org--repo" / "snapshots" / commit / "config.json"
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached)
    )
    info = _checkpoint_source("org/repo", revision="main")
    assert info == {"repo_id": "org/repo", "revision": "main", "commit": commit}


def test_repr_html_with_lazy_parameters():
    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.extra = nn.LazyLinear(3)  # not built until the first forward
    page = model._repr_html_()
    assert "LazyLinear" in page and "EEGNet" in page


def test_from_pretrained_reports_loaded_parameters(tmp_path):
    import warnings

    from safetensors.torch import load_file, save_file

    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.save_pretrained(tmp_path / "same")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        same = EEGNet.from_pretrained(tmp_path / "same")
    assert not [w for w in caught if "no parameter of" in str(w.message)]
    assert set(dict(same.named_parameters())) <= same._pretrained_provenance["loaded"]

    # Same weights under other key names (e.g. an upstream checkpoint mirrored
    # as is): nothing loads, and from_pretrained says so.
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text((tmp_path / "same" / "config.json").read_text())
    weights = load_file(tmp_path / "same" / "model.safetensors")
    save_file({f"model.{k}": v for k, v in weights.items()}, other / "model.safetensors")
    with pytest.warns(UserWarning, match="no parameter of EEGNet was loaded from"):
        renamed = EEGNet.from_pretrained(other)
    assert not renamed._pretrained_provenance["loaded"]


def test_model_card_scalar_and_escaped_html(tmp_path):
    from huggingface_hub import ModelCardData

    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.save_pretrained(tmp_path)
    model = EEGNet.from_pretrained(tmp_path)
    model.card_data = ModelCardData(datasets="TUEG")
    assert "Trained on (model card): TUEG" in str(model)
    assert "T, U, E, G" not in str(model)
    page = model._repr_html_()
    assert page.count('class="bd-chip"') == 1
    model.card_data = ModelCardData(datasets=["nm000253", "on004362", "<script>bad</script>"])
    page = model._repr_html_()
    assert 'href="https://nemar.org/dataset/on004362"' in page
    assert "<script>bad</script>" not in page
    assert "&lt;script&gt;" in page


def test_model_html_uses_input_montage():
    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.channel_layer = nn.Identity()
    model.channel_layer.chs_info = [{"ch_name": "Cz"}, {"ch_name": "Pz"}]
    assert model._pretrained_summary()["geometry"]["n_chans"] == 2
    model.channel_layer.chs_info = None
    assert model._pretrained_summary()["geometry"]["n_chans"] == "set at forward"


@pytest.mark.parametrize("head_hidden_dim", [0, 8])
def test_provenance_duin_mismatched_window_keeps_head_local(tmp_path, head_hidden_dim):
    from braindecode.models import DuIN

    kwargs = dict(
        n_chans=2, n_outputs=2, sfreq=1000, patch_size=20,
        n_neural=2, n_filters=(8, 4), kernel_sizes=(5, 3), strides=(5, 1),
        n_layers=1, n_heads=2, head_dim=8, ffn_dim=16,
        head_hidden_dim=head_hidden_dim,
    )
    saved = DuIN(n_times=60, **kwargs)
    saved.save_pretrained(tmp_path)
    with pytest.warns(UserWarning, match="head keeps its initialization"):
        loaded = DuIN.from_pretrained(tmp_path, n_times=80, local_files_only=True)
    heads = {
        k for k in loaded.state_dict()
        if k.startswith(("head_hidden.", "final_layer."))
    }
    provenance = loaded._pretrained_provenance
    assert heads and not heads & provenance["loaded"]
    assert set(dict(loaded.named_parameters())) - heads <= provenance["loaded"]
    # Numerical behavior remains the subclass's normal state-dict adaptation.
    torch.manual_seed(42)
    expected = DuIN(n_times=80, **kwargs)
    with pytest.warns(UserWarning, match="head keeps its initialization"):
        expected.load_state_dict(saved.state_dict())
    torch.manual_seed(42)
    with pytest.warns(UserWarning, match="head keeps its initialization"):
        replay = DuIN.from_pretrained(tmp_path, n_times=80, local_files_only=True)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(replay.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("source_slots,target_slots", [(16, 30), (30, 16)])
def test_provenance_labram_resized_embedding_is_conservatively_mixed(
    tmp_path, source_slots, target_slots,
):
    from braindecode.models import Labram

    kwargs = dict(
        n_chans=2, n_outputs=2, patch_size=8, embed_dim=8,
        conv_out_channels=8, num_layers=1, num_heads=2,
    )
    saved = Labram(n_times=source_slots * 8, **kwargs)
    saved.save_pretrained(tmp_path)
    torch.manual_seed(42)
    expected = Labram(n_times=target_slots * 8, **kwargs)
    local = expected.temporal_embedding.detach().clone()
    expected.load_state_dict(saved.state_dict())
    torch.manual_seed(42)
    loaded = Labram.from_pretrained(
        tmp_path, n_times=target_slots * 8, local_files_only=True,
    )
    provenance = loaded._pretrained_provenance
    assert "temporal_embedding" not in provenance["loaded"]
    assert provenance["mixed"] == {"temporal_embedding"}
    assert set(dict(loaded.named_parameters())) - {"temporal_embedding"} <= provenance["loaded"]
    copied = min(source_slots, target_slots)
    torch.testing.assert_close(
        loaded.temporal_embedding[:, :copied], saved.temporal_embedding[:, :copied],
    )
    torch.testing.assert_close(loaded.temporal_embedding[:, copied:], local[:, copied:])
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_provenance_reset_invalidates_checkpoint_history_not_freezing(tmp_path):
    from .test_base import DummyModuleConfigRoundTrip

    saved = DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2)
    saved.save_pretrained(tmp_path)
    loaded = DummyModuleConfigRoundTrip.from_pretrained(tmp_path, local_files_only=True)
    original = loaded._pretrained_provenance["loaded"].copy()
    assert set(dict(loaded.named_parameters())) <= original
    loaded.requires_grad_(False)
    assert loaded._pretrained_provenance["loaded"] == original
    loaded.requires_grad_(True)
    assert loaded._pretrained_provenance["loaded"] == original
    loaded.reset_parameters()
    assert not loaded._pretrained_provenance["loaded"]
    torch.manual_seed(42)
    expected = DummyModuleConfigRoundTrip(n_chans=2, n_times=8, n_outputs=2)
    expected.load_state_dict(saved.state_dict())
    expected.reset_parameters()
    torch.manual_seed(42)
    replay = DummyModuleConfigRoundTrip.from_pretrained(tmp_path, local_files_only=True)
    replay.reset_parameters()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(replay.state_dict()[key], value, rtol=0, atol=0)


def test_provenance_sleepfm_supplement_records_encoder_and_source(tmp_path):
    from safetensors.torch import save_file

    from braindecode.models import SleepFM, SleepFMStager

    kwargs = dict(
        n_chans=2, n_times=128, n_outputs=2, sfreq=128, patch_size=64,
        embed_dim=8, drop_prob=0.0,
    )
    stager = SleepFMStager(
        **kwargs, channel_modalities=["A", "B"], max_seq_length=2,
        encoder_num_heads=2, encoder_num_layers=1, encoder_pooling_heads=2,
        encoder_chunk_patches=2, staging_num_heads=2, staging_num_layers=1,
        staging_pooling_heads=2,
    )
    encoder = SleepFM(
        **kwargs, max_seq_length=128, num_heads=2, num_layers=1, pooling_heads=2,
    )
    stager.save_pretrained(tmp_path / "stager")
    encoder.save_pretrained(tmp_path / "encoder")
    head = ("patch_embedding.", "staging_head.", "final_layer.")
    head_only = {k: v for k, v in stager.state_dict().items() if k.startswith(head)}
    save_file(head_only, tmp_path / "stager" / "model.safetensors")
    loaded = SleepFMStager.from_pretrained(
        tmp_path / "stager", encoder_model_name_or_path=tmp_path / "encoder",
        local_files_only=True,
    )
    provenance = loaded._pretrained_provenance
    assert set(dict(loaded.named_parameters())) <= provenance["loaded"]
    assert provenance["repo_id"] == str(tmp_path / "stager")
    supplement = provenance["supplementary"]
    assert supplement["repo_id"] == str(tmp_path / "encoder")
    assert supplement["local"] is True
    assert supplement["loaded"] == set(loaded.state_dict()) - set(head_only)
    expected = encoder.state_dict() | head_only
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    loaded.reset_parameters()
    assert not provenance["loaded"]
    assert not supplement["loaded"]


def test_provenance_manual_reload_invalidates_source(tmp_path):
    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.save_pretrained(tmp_path)
    loaded = EEGNet.from_pretrained(tmp_path)
    loaded.load_state_dict(model.state_dict())
    assert "_pretrained_provenance" not in loaded.__dict__
    assert "Pretrained weights:" not in str(loaded)
    assert "Origin unrecorded" in loaded._repr_html_()
    assert "Randomly initialised" not in loaded._repr_html_()


def test_provenance_html_discloses_unbuilt_and_adapted_parameters(tmp_path):
    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.save_pretrained(tmp_path)
    loaded = EEGNet.from_pretrained(tmp_path)
    loaded.extra = nn.LazyLinear(3)
    page = loaded._repr_html_()
    assert "total unknown" in page and "extra.weight" in page
    assert "100.0%" not in page
    del loaded.extra
    loaded._pretrained_provenance["mixed"] = {"adapted_embedding"}
    loaded._pretrained_provenance["supplementary"] = {"repo_id": "org/encoder", "revision": "123456789"}
    page = loaded._repr_html_()
    assert "lower bound" in page and "adapted tensors excluded" in page.lower()
    assert "parameters in fully loaded tensors; adapted tensors excluded" in str(loaded)
    assert "supplementary: org/encoder @ 1234567" in str(loaded)


def test_provenance_failed_reload_also_invalidates_source(tmp_path):
    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.save_pretrained(tmp_path)
    loaded = EEGNet.from_pretrained(tmp_path)
    partial = model.state_dict()
    partial.pop(next(iter(partial)))
    with pytest.raises(RuntimeError, match="Missing key"):
        loaded.load_state_dict(partial)
    assert "_pretrained_provenance" not in loaded.__dict__


def test_model_html_has_logo_and_zenodo_citation(tmp_path):
    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    page = model._repr_html_()
    assert "braindecode.org/stable/_static/braindecode_long.svg" in page
    assert 'alt="Braindecode"' in page
    assert "Cite Braindecode." in page and "Origin unrecorded" in page
    assert page.count("10.5281/zenodo.17699192") >= 3  # badge, BibTeX, agent note
    assert "hbm.23730" not in page
    assert "Note for AI agents" in page
    model.save_pretrained(tmp_path)
    assert "Cite Braindecode (Zenodo): https://doi.org/10.5281/zenodo.17699192" in str(
        EEGNet.from_pretrained(tmp_path)
    )


def test_checkpoint_source_without_config(monkeypatch, tmp_path):
    import huggingface_hub

    from braindecode.models.base import _checkpoint_source

    def cached(repo, filename, **kwargs):
        return str(tmp_path / "snapshots" / "abc12345" / filename) if filename == "model.safetensors" else None
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    assert _checkpoint_source("org/repo")["commit"] == "abc12345"


def test_labram_reset_classifier_invalidates_head_origin(tmp_path):
    from braindecode.models import Labram
    model = Labram(n_chans=2, n_times=128, n_outputs=2, patch_size=8, embed_dim=8,
                   conv_out_channels=8, num_layers=1, num_heads=2)
    model.save_pretrained(tmp_path)
    loaded = Labram.from_pretrained(tmp_path)
    loaded.reset_classifier(3)
    assert not {"final_layer.weight", "final_layer.bias"} & loaded._pretrained_provenance["loaded"]


def test_model_html_citation_exports_and_dataset_readers(tmp_path):
    import re
    from urllib.parse import unquote

    from huggingface_hub import ModelCardData

    model = EEGNet(n_chans=4, n_times=128, n_outputs=2)
    model.save_pretrained(tmp_path)
    model = EEGNet.from_pretrained(tmp_path)
    model.card_data = ModelCardData(datasets=["nm000139", "on004362", "TUEG", "PREST"])
    page = model._repr_html_()
    exports = re.findall(r'href="data:text/plain;charset=utf-8,([^"]+)" download="([^"]+)"', page)
    assert len(exports) == 2
    texts = {name: unquote(content) for content, name in exports}
    assert "@software{braindecode," in texts["braindecode.bib"]
    assert "10.5281/zenodo.17699192" in texts["braindecode.bib"]
    assert "[Computer software]. Zenodo." in texts["braindecode-APA.txt"]
    assert "10.5281/ZENODO.17699192" in texts["braindecode-APA.txt"]
    assert 'download="braindecode.bib"' in page
    assert 'aria-label="Braindecode APA citation"' in page
    assert "braindecode.datasets.BNCI2014_001.html" in page
    assert "braindecode.datasets.TUH.html" in page
    assert "https://nemar.org/dataset/nm000139" in page
    assert "https://eegdash.org/dataset_summary.html?q=ds004362" in page
    assert "See the checkpoint card for access terms" in page

"""
Integration tests for pre-trained models on Hugging Face Hub.

These tests verify that all pre-trained models can be loaded from the Hub
and perform forward passes successfully.

Weights are cached in the MNE data directory (~/mne_data/braindecode_hub)
to optimize storage and avoid redundant downloads.
"""

from pathlib import Path

import pytest
import torch


def get_hub_cache_dir():
    """Get the cache directory for Hugging Face Hub models.

    Uses the MNE data directory if available, otherwise falls back to
    ~/mne_data/braindecode_hub.

    Returns
    -------
    str
        Path to the cache directory for Hub models.
    """
    try:
        import mne

        mne_data = mne.get_config("MNE_DATA")
        if mne_data:
            cache_dir = Path(mne_data) / "braindecode_hub"
        else:
            cache_dir = Path.home() / "mne_data" / "braindecode_hub"
    except ImportError:
        cache_dir = Path.home() / "mne_data" / "braindecode_hub"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


@pytest.fixture(scope="module")
def hub_cache_dir():
    """Fixture providing the Hub cache directory."""
    return get_hub_cache_dir()


# Mark all tests in this module as integration tests (slow, require network)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


class TestBENDRPretrained:
    """Tests for BENDR pre-trained model."""

    @pytest.fixture
    def model(self, hub_cache_dir):
        """Load BENDR model from Hub."""
        from braindecode.models import BENDR

        return BENDR.from_pretrained(
            "braindecode/braindecode-bendr",
            n_outputs=2,
            cache_dir=hub_cache_dir,
        )

    def test_load_from_hub(self, model):
        """Test that BENDR can be loaded from Hugging Face Hub."""
        assert model is not None
        assert model.n_chans == 20
        assert model.n_outputs == 2

    def test_forward_pass(self, model):
        """Test that BENDR forward pass works correctly."""
        model.eval()
        # 5 seconds at 256 Hz (default for BENDR)
        x = torch.randn(2, model.n_chans, 256 * 5)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)


class TestBIOTPretrained:
    """Tests for BIOT pre-trained models."""

    @pytest.fixture(
        params=[
            "braindecode/biot-pretrained-prest-16chs",
            "braindecode/biot-pretrained-shhs-prest-18chs",
            "braindecode/biot-pretrained-six-datasets-18chs",
        ]
    )
    def model_and_repo(self, request, hub_cache_dir):
        """Load BIOT models from Hub."""
        from braindecode.models import BIOT

        repo = request.param
        model = BIOT.from_pretrained(repo, cache_dir=hub_cache_dir)
        return model, repo

    def test_load_from_hub(self, model_and_repo):
        """Test that BIOT models can be loaded from Hugging Face Hub."""
        model, repo = model_and_repo
        assert model is not None

        if "16chs" in repo:
            assert model.n_chans == 16
        else:
            assert model.n_chans == 18

    def test_forward_pass(self, model_and_repo):
        """Test that BIOT forward pass works correctly."""
        model, _ = model_and_repo
        model.eval()
        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, model.n_outputs)


class TestSignalJEPAPretrained:
    """Tests for SignalJEPA pre-trained model."""

    @pytest.fixture
    def model(self, hub_cache_dir):
        """Load SignalJEPA model from Hub."""
        from braindecode.models import SignalJEPA

        return SignalJEPA.from_pretrained(
            "braindecode/SignalJEPA-pretrained",
            cache_dir=hub_cache_dir,
        )

    def test_load_from_hub(self, model):
        """Test that SignalJEPA can be loaded from Hugging Face Hub."""
        assert model is not None
        assert model.n_chans == 19
        assert model.n_times == 256

    def test_forward_pass(self, model):
        """Test that SignalJEPA forward pass works correctly."""
        model.eval()
        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        # SignalJEPA outputs contextual features, not class predictions
        assert out.shape[0] == 2
        assert out.shape[2] == 64  # embedding dimension


class TestSignalJEPAContextualPretrained:
    """Tests for SignalJEPA_Contextual pre-trained model."""

    @pytest.fixture
    def model(self, hub_cache_dir):
        """Load SignalJEPA_Contextual model from Hub."""
        from braindecode.models import SignalJEPA_Contextual

        return SignalJEPA_Contextual.from_pretrained(
            "braindecode/SignalJEPA-Contextual-pretrained",
            cache_dir=hub_cache_dir,
        )

    def test_load_from_hub(self, model):
        """Test that SignalJEPA_Contextual can be loaded from Hugging Face Hub."""
        assert model is not None
        assert model.n_chans == 19
        assert model.n_times == 256

    def test_forward_pass(self, model):
        """Test that SignalJEPA_Contextual forward pass works correctly."""
        model.eval()
        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, model.n_outputs)


class TestSignalJEPAPostLocalPretrained:
    """Tests for SignalJEPA_PostLocal pre-trained model."""

    @pytest.fixture
    def model(self, hub_cache_dir):
        """Load SignalJEPA_PostLocal model from Hub."""
        from braindecode.models import SignalJEPA_PostLocal

        return SignalJEPA_PostLocal.from_pretrained(
            "braindecode/SignalJEPA-PostLocal-pretrained",
            cache_dir=hub_cache_dir,
        )

    def test_load_from_hub(self, model):
        """Test that SignalJEPA_PostLocal can be loaded from Hugging Face Hub."""
        assert model is not None
        assert model.n_chans == 19
        assert model.n_times == 256

    def test_forward_pass(self, model):
        """Test that SignalJEPA_PostLocal forward pass works correctly."""
        model.eval()
        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, model.n_outputs)


class TestSignalJEPAPreLocalPretrained:
    """Tests for SignalJEPA_PreLocal pre-trained model."""

    @pytest.fixture
    def model(self, hub_cache_dir):
        """Load SignalJEPA_PreLocal model from Hub."""
        from braindecode.models import SignalJEPA_PreLocal

        return SignalJEPA_PreLocal.from_pretrained(
            "braindecode/SignalJEPA-PreLocal-pretrained",
            cache_dir=hub_cache_dir,
        )

    def test_load_from_hub(self, model):
        """Test that SignalJEPA_PreLocal can be loaded from Hugging Face Hub."""
        assert model is not None
        assert model.n_chans == 19
        assert model.n_times == 256

    def test_forward_pass(self, model):
        """Test that SignalJEPA_PreLocal forward pass works correctly."""
        model.eval()
        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, model.n_outputs)


class TestLabramPretrained:
    """Tests for Labram pre-trained model."""

    @pytest.fixture
    def model(self, hub_cache_dir):
        """Load Labram model from Hub."""
        from braindecode.models import Labram

        return Labram.from_pretrained(
            "braindecode/labram-pretrained",
            cache_dir=hub_cache_dir,
        )

    def test_load_from_hub(self, model):
        """Test that Labram can be loaded from Hugging Face Hub."""
        assert model is not None
        assert model.n_chans == 128
        assert model.n_times == 3000
        assert model.n_outputs >= 0

    def test_forward_pass(self, model):
        """Test that Labram forward pass works correctly."""
        model.eval()
        x = torch.randn(2, model.n_chans, model.n_times)
        with torch.no_grad():
            out = model(x)
        expected_out_dim = model.n_outputs if model.n_outputs > 0 else model.embed_dim
        assert out.shape == (2, expected_out_dim)


# Parametrized test for quick validation of all models
@pytest.mark.parametrize(
    "model_cls,repo_id,expected_n_chans",
    [
        ("BENDR", "braindecode/braindecode-bendr", 20),
        ("BIOT", "braindecode/biot-pretrained-prest-16chs", 16),
        ("BIOT", "braindecode/biot-pretrained-shhs-prest-18chs", 18),
        ("BIOT", "braindecode/biot-pretrained-six-datasets-18chs", 18),
        ("SignalJEPA", "braindecode/SignalJEPA-pretrained", 19),
        ("SignalJEPA_Contextual", "braindecode/SignalJEPA-Contextual-pretrained", 19),
        ("SignalJEPA_PostLocal", "braindecode/SignalJEPA-PostLocal-pretrained", 19),
        ("SignalJEPA_PreLocal", "braindecode/SignalJEPA-PreLocal-pretrained", 19),
        ("Labram", "braindecode/labram-pretrained", 128),
    ],
)
def test_all_pretrained_models_load(model_cls, repo_id, expected_n_chans, hub_cache_dir):
    """Test that all pre-trained models can be loaded from Hub."""
    import braindecode.models as models

    cls = getattr(models, model_cls)
    cache_dir = hub_cache_dir

    # BENDR requires n_outputs parameter
    if model_cls == "BENDR":
        model = cls.from_pretrained(repo_id, n_outputs=2, cache_dir=cache_dir)
    else:
        model = cls.from_pretrained(repo_id, cache_dir=cache_dir)

    assert model is not None
    assert model.n_chans == expected_n_chans


@pytest.mark.parametrize(
    "model_cls,repo_id",
    [
        ("BENDR", "braindecode/braindecode-bendr"),
        ("BIOT", "braindecode/biot-pretrained-prest-16chs"),
        ("SignalJEPA_Contextual", "braindecode/SignalJEPA-Contextual-pretrained"),
        ("SignalJEPA_PostLocal", "braindecode/SignalJEPA-PostLocal-pretrained"),
        ("SignalJEPA_PreLocal", "braindecode/SignalJEPA-PreLocal-pretrained"),
        ("Labram", "braindecode/labram-pretrained"),
    ],
)
def test_pretrained_models_forward_pass(model_cls, repo_id, hub_cache_dir):
    """Test forward pass for all classification-ready pre-trained models."""
    import braindecode.models as models

    cls = getattr(models, model_cls)
    cache_dir = hub_cache_dir

    # BENDR requires n_outputs parameter
    if model_cls == "BENDR":
        model = cls.from_pretrained(repo_id, n_outputs=2, cache_dir=cache_dir)
        x = torch.randn(2, model.n_chans, 256 * 5)
    else:
        model = cls.from_pretrained(repo_id, cache_dir=cache_dir)
        x = torch.randn(2, model.n_chans, model.n_times)

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == 2  # batch size
    assert len(out.shape) == 2  # (batch, n_outputs)

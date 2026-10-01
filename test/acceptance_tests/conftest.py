import pytest
import torch


@pytest.fixture
def deterministic_algorithms():
    """Use deterministic torch algorithms for one test, then restore the flags."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

"""Regression checks for test-only state isolation and partitioning."""

import random

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("fail", [False, True])
def test_acceleration_fixture_restores_caller_state(acceleration_caller_state, fail):
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    grad = torch.is_grad_enabled()

    def destructive_recipe():
        with acceleration_caller_state():
            random.seed(72)
            np.random.seed(72)
            torch.manual_seed(72)
            torch.set_num_threads(2 if threads == 1 else 1)
            torch.set_grad_enabled(not grad)
            if fail:
                raise RuntimeError("recipe failure")

    if fail:
        with pytest.raises(RuntimeError, match="recipe failure"):
            destructive_recipe()
    else:
        destructive_recipe()
    assert random.getstate() == python_rng
    restored = np.random.get_state()
    assert restored[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored[1], numpy_rng[1])
    assert restored[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.get_num_threads() == threads
    assert torch.is_grad_enabled() == grad

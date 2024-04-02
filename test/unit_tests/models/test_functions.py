import pytest
import torch
from torch import tensor

from braindecode.models.functions import _modify_eig_backward, \
    _modify_eig_forward, logm


def test_modify_eig_forward():
    # Define the function to be applied to the eigenvalues

    # Create a tensor X with known eigenvalues and eigenvectors
    X = torch.tensor([[4.0, 1.0], [1.0, 2.0]])

    # Call the function with X and the function_applied
    output, s, U, s_modified = _modify_eig_forward(X, function_applied)

    # Check the eigenvalues
    assert torch.allclose(s, tensor([1.5858, 4.4142]), atol=1e-4)

    # Check the eigenvectors
    assert torch.allclose(U, tensor([[0.3827, -0.9239], [-0.9239, -0.3827]]),
                          atol=1e-4)

    # Check the modified eigenvalues
    assert torch.allclose(s_modified, tensor([0.4611, 1.4848]), atol=1e-4)

    # Check the output
    assert torch.allclose(
        output, tensor([[1.3349, 0.3619], [0.3619, 0.6110]]), atol=1e-4
    )


def function_applied(x):
    return x.log()


@pytest.fixture
def setup_test_case():
    X = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float32)
    grad_output = torch.tensor([[0.5, 0.1], [0.1, 0.2]], dtype=torch.float32)
    s, U = torch.linalg.eigh(X)
    s_modified = s + 1  # Simple modification for testing purposes
    return grad_output, s, U, s_modified


def derivative(s):
    """Example derivative function which applies on eigenvalues."""
    return s.pow(2)  # Example: square of each eigenvalue


def test_modify_eig_backward(setup_test_case):
    """
    Test the _modify_eig_backward function.

    Parameters
    ----------
    setup_test_case: tuple


    """
    # Sample data creation
    grad_output = torch.rand((3, 3), dtype=torch.float64)
    s = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    U = torch.eye(3, dtype=torch.float64)
    s_modified = torch.tensor([1.1, 2.1, 3.1], dtype=torch.float64)

    # Call the function under test
    grad_input = _modify_eig_backward(grad_output, s, U, s_modified,
                                      derivative)

    # Verify the output is a tensor
    assert isinstance(grad_input, torch.Tensor)

    # Verify the shape of the output tensor
    assert grad_input.shape == grad_output.shape

    # Thiking about another test case to check the correctness of the function
    # I am not sure how to do it


def test_logm_forward():
    # Create a symmetric matrix
    X = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64,
                     requires_grad=True)
    # Expected output is log of diagonal elements since input is diagonal
    expected = torch.tensor([[torch.log(torch.tensor(2.0)), 0.0],
                             [0.0, torch.log(torch.tensor(3.0))]],
                            dtype=torch.float64)

    output = logm.apply(X)
    # Check if the output matches the expected matrix logarithm
    assert torch.allclose(output,
                          expected), "Forward computation of matrix logarithm failed."


def test_logm_edge_cases():
    # Identity matrix test
    X = torch.eye(3, 3, dtype=torch.float64, requires_grad=True)
    output = logm.apply(X)
    expected = torch.zeros_like(X)
    assert torch.allclose(output,
                          expected), "Edge case of identity matrix failed."

def test_logm_numerical_stability():
    X = torch.tensor([[1e-10, 0], [0, 1e+10]], dtype=torch.float64,
                     requires_grad=True)
    output = logm.apply(X)
    expected = torch.tensor([[torch.log(torch.tensor(1e-10)), 0],
                             [0, torch.log(torch.tensor(1e+10))]],
                            dtype=torch.float64)
    assert torch.allclose(output,
                          expected), "Numerical stability test failed."


def finite_difference_grad(X, eps=1e-6):
    X.requires_grad_(True)
    finite_diff_grad = torch.zeros_like(X)
    Y = logm.apply(X)  # Original output

    for i in range(X.size(0)):
        for j in range(i, X.size(1)):  # Ensure symmetry
            # Perturbation for finite difference approximation
            X_perturb = X.clone()
            X_perturb[i, j] += eps
            X_perturb[j, i] += eps  # Maintain symmetry

            Y_perturb = logm.apply(X_perturb)

            # Ensure there are no NaN values before proceeding
            if torch.isnan(Y_perturb).any():
                print(f"NaN detected in perturbed output at ({i},{j})")
                continue

            diff = Y_perturb - Y
            if torch.isnan(diff).any():
                print(f"NaN detected in difference at ({i},{j})")
                continue

            finite_diff_grad[i, j] = (diff.sum() / (2 * eps)).item()
            if i != j:
                finite_diff_grad[j, i] = finite_diff_grad[i, j]  # Symmetry

    return finite_diff_grad


# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD-3

import os
import tempfile
from unittest import mock

import h5py
import mne
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.utils import check_random_state

from braindecode.util import (
    _cov_and_var_to_corr,
    _cov_to_corr,
    corr,
    cov,
    create_mne_dummy_raw,
    get_balanced_batches,
    np_to_th,
    read_all_file_names,
    set_random_seeds,
    th_to_np,
)


def test_create_mne_dummy_raw(tmp_path):
    n_channels, n_times, sfreq = 2, 10000, 100
    raw, fnames = create_mne_dummy_raw(
        n_channels, n_times, sfreq, savedir=tmp_path, save_format=["fif", "hdf5"]
    )

    assert isinstance(raw, mne.io.RawArray)
    assert len(raw.ch_names) == n_channels
    assert raw.n_times == n_times
    assert raw.info["sfreq"] == sfreq
    assert isinstance(fnames, dict)
    assert os.path.isfile(fnames["fif"])
    assert os.path.isfile(fnames["hdf5"])

    _ = mne.io.read_raw_fif(fnames["fif"], preload=False, verbose=None)
    with h5py.File(fnames["hdf5"], "r") as hf:
        _ = np.array(hf["fake_raw"])


def test_set_random_seeds_raise_value_error():
    with pytest.raises(
        ValueError, match="cudnn_benchmark expected to be bool or None, got 'abc'"
    ):
        set_random_seeds(100, True, "abc")


def test_set_random_seeds_warning():
    torch.backends.cudnn.benchmark = True
    with pytest.warns(
        UserWarning,
        match="torch.backends.cudnn.benchmark was set to True which may results in "
        "lack of reproducibility. In some cases to ensure reproducibility you "
        "may need to set torch.backends.cudnn.benchmark to False.",
    ):
        set_random_seeds(100, True)


def test_set_random_seeds_with_valid_cudnn_benchmark():
    with mock.patch("torch.backends.cudnn") as mock_cudnn:
        # Test with cudnn_benchmark = True
        set_random_seeds(42, cuda=True, cudnn_benchmark=True)
        assert mock_cudnn.benchmark is True

        # Test with cudnn_benchmark = False
        set_random_seeds(42, cuda=True,
                         cudnn_benchmark=False)
        assert mock_cudnn.benchmark is False


def test_set_random_seeds_with_invalid_cudnn_benchmark():
    with pytest.raises(ValueError):
        set_random_seeds(42, cuda=True,
                         cudnn_benchmark='invalid_type')


def test_th_to_np_data_preservation():
    # Test with different data types
    for dtype in [torch.float32, torch.int32]:
        tensor = torch.tensor([1, 2, 3], dtype=dtype)
        np_array = th_to_np(tensor)
        assert np_array.dtype == tensor.numpy().dtype
        # Corrected attribute access
        assert_array_equal(np_array, tensor.numpy())


def test_th_to_np_on_cpu():
    # Create a tensor on CPU
    cpu_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    np_array = th_to_np(cpu_tensor)
    # Check the type and data of the numpy array
    assert isinstance(np_array, np.ndarray)
    assert np_array.dtype == cpu_tensor.numpy().dtype
    # Correct way to check dtype
    assert_array_equal(np_array, cpu_tensor.numpy())


def test_cov_basic():
    # Create two simple identical arrays
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    expected_cov = np.array([[1, 1], [1, 1]])  # Calculated expected covariance
    computed_cov = cov(a, b)
    assert_allclose(computed_cov, expected_cov, rtol=1e-5)


def test_cov_dimension_mismatch():
    # Arrays with mismatched sample size should raise an error
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        cov(a, b)


def test_np_to_th_basic_conversion():
    # Convert a simple list to tensor
    data = [1, 2, 3]
    tensor = np_to_th(data)
    assert torch.is_tensor(tensor)
    assert_array_equal(tensor.numpy(), np.array(data))


def test_np_to_th_dtype_conversion():
    # Convert and specify dtype
    data = [1.0, 2.0, 3.0]
    tensor = np_to_th(data, dtype=np.float32)
    assert tensor.dtype == torch.float32


def test_np_to_th_requires_grad():
    # Check requires_grad attribute
    data = np.array([1.0, 2.0, 3.0])
    tensor = np_to_th(data, requires_grad=True)
    assert tensor.requires_grad is True


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Requires CUDA support")
def test_np_to_th_pin_memory():
    # Create a numpy array
    data = np.array([1, 2, 3])

    # Convert the numpy array to a tensor with pin_memory=True
    tensor = np_to_th(data, pin_memory=True)
    # Check if the tensor is pinned in memory
    assert tensor.is_pinned() is True
    # Convert the numpy array to a tensor with pin_memory=False
    tensor = np_to_th(data, pin_memory=False)
    # Check if the tensor is not pinned in memory
    assert tensor.is_pinned() is False

def test_np_to_th_requires_grad_unsupported_dtype():
    # Attempt to set requires_grad on an unsupported dtype (integers)
    data = np.array([1, 2, 3])
    with pytest.raises(RuntimeError,
                       match="Only Tensors of floating point and complex dtype can require gradients"):
        np_to_th(data, requires_grad=True)


def test_np_to_th_tensor_options():
    # Additional tensor options like device
    data = [1, 2, 3]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = np_to_th(data, device=device)
    assert tensor.device.type == device


def test_np_to_th_single_number_conversion():
    # Single number conversion to tensor
    data = 42
    tensor = np_to_th(data)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 42  # Check if the value is correct


def test_cov_and_var_to_corr_zero_variance():
    # Scenario with zero variance, expecting result to handle divide by zero
    this_cov = np.array([[1, 0],
                         [0, 1]])
    var_a = np.array([0, 1])  # Zero variance in the first variable
    var_b = np.array([1, 0])  # Zero variance in the second variable
    calculated_corr = _cov_and_var_to_corr(this_cov, var_a, var_b)

    # Expected correlation matrix
    expected_corr = np.array([[np.inf, np.nan],
                               [0, np.inf]])
    assert_array_equal(calculated_corr, expected_corr)


def test_cov_and_var_to_corr_single_element():
    # Testing with single-element arrays
    this_cov = np.array([[1]])
    var_a = np.array([1])
    var_b = np.array([1])
    expected_corr = np.array([[1]])
    calculated_corr = _cov_and_var_to_corr(this_cov, var_a, var_b)
    assert_array_equal(calculated_corr, expected_corr)



def test_cov_to_corr_unbiased():
    # Create datasets a and b with known covariance and variance characteristics
    a = np.array([[1, 2, 3, 4],
                  [2, 3, 4, 5]])
    b = np.array([[1, 3, 5, 7],
                  [5, 6, 7, 8]])
    # Covariance between the features of a and b
    # Calculating covariance manually for known values
    demeaned_a = a - np.mean(a, axis=1, keepdims=True)
    demeaned_b = b - np.mean(b, axis=1, keepdims=True)
    this_cov = np.dot(demeaned_a, demeaned_b.T) / (b.shape[1] - 1)

    # Compute expected correlation using standard formulas for correlation
    var_a = np.var(a, axis=1, ddof=1)
    var_b = np.var(b, axis=1, ddof=1)
    expected_divisor = np.outer(np.sqrt(var_a), np.sqrt(var_b))
    expected_corr = this_cov / expected_divisor

    # Compute correlation using the function
    calculated_corr = _cov_to_corr(this_cov, a, b)

    # Assert that the calculated correlation matches the expected correlation
    assert_allclose(calculated_corr, expected_corr, rtol=1e-5)


def test_balanced_batches_basic():
    n_trials = 100
    seed = 42
    rng = check_random_state(seed)
    n_batches = 10
    batches = get_balanced_batches(n_trials, rng, shuffle=False,
                                   n_batches=n_batches)

    # Check correct number of batches
    assert len(batches) == n_batches

    # Check balanced batch sizes
    all_batch_sizes = [len(batch) for batch in batches]
    max_size = max(all_batch_sizes)
    min_size = min(all_batch_sizes)
    assert max_size - min_size <= 1

    # Check if all indices are unique and accounted for
    all_indices = np.concatenate(batches)
    assert np.array_equal(np.sort(all_indices), np.arange(n_trials))


def test_balanced_batches_with_batch_size():
    n_trials = 105
    seed = 42
    rng = check_random_state(seed)
    batch_size = 20
    batches = get_balanced_batches(n_trials, rng, shuffle=False,
                                   batch_size=batch_size)

    # Check the modified batch size condition
    expected_n_batches = int(np.round(n_trials / float(batch_size)))
    assert len(batches) == expected_n_batches

    # Checking the total number of indices
    all_indices = np.concatenate(batches)
    assert len(all_indices) == n_trials


def test_balanced_batches_shuffle():
    n_trials = 50
    seed = 42
    rng = check_random_state(seed)
    batches_no_shuffle = get_balanced_batches(n_trials, rng, shuffle=False,
                                              batch_size=10)
    rng = check_random_state(seed)
    batches_with_shuffle = get_balanced_batches(n_trials, rng, shuffle=True,
                                                batch_size=10)

    # Check that shuffling changes the order of indices
    assert not np.array_equal(np.concatenate(batches_no_shuffle),
                              np.concatenate(batches_with_shuffle))


def test_corr_correlation_computation():
    # Create two 2D arrays with known correlation
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])

    # Call the corr function
    corr_result = corr(a, b)

    # Compute the known correlation
    known_corr = np.array([np.corrcoef(a[i], b[i]) for i in range(a.shape[0])])

    # Extract the correlation computation from the corr function
    computed_corr = _cov_to_corr(cov(a, b), a, b)

    # Assert that the computed correlation matches the known correlation
    assert np.allclose(computed_corr, known_corr)
    assert np.allclose(corr_result, computed_corr)


def test_get_balanced_batches_zero_batches():
    # Create a scenario where n_batches is 0
    n_trials = 10
    rng = check_random_state(0)
    shuffle = False
    n_batches = 0
    batch_size = None

    # Call the get_balanced_batches function
    batches = get_balanced_batches(n_trials, rng, shuffle, n_batches, batch_size)

    # Check if the function returns a single batch with all trials
    assert len(batches) == 1
    assert len(batches[0]) == n_trials


def test_get_balanced_batches_i_batch_less_than_n_batches_with_extra_trial():
    # Create a scenario where i_batch < n_batches_with_extra_trial
    n_trials = 10
    rng = check_random_state(0)
    shuffle = False
    n_batches = 6
    batch_size = None

    # Call the get_balanced_batches function
    batches = get_balanced_batches(n_trials, rng, shuffle, n_batches, batch_size)

    # Check if the first batch has one more trial than the last batch
    assert len(batches[0]) > len(batches[-1])


def test_read_all_file_names():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_files = []
        try:
            # Create some temporary files with .txt extension
            for i in range(5):
                temp_file = os.path.join(tmpdir, f'temp{i}.txt')
                with open(temp_file, 'w') as f:
                    f.write('This is a temporary file.')
                temp_files.append(temp_file)

            # Call the read_all_file_names function
            file_paths = read_all_file_names(tmpdir, '.txt')

            # Check if the function found all the temporary files
            assert len(file_paths) == 5

            # Check if the paths returned by the function are correct
            for i in range(5):
                assert os.path.join(tmpdir, f'temp{i}.txt') in file_paths
        finally:
            # Delete the temporary files
            for temp_file in temp_files:
                os.remove(temp_file)


def test_read_all_file_names_error():
    with pytest.raises(AssertionError):
        # Call the read_all_file_names function with a non-existent directory
        read_all_file_names('non_existent_dir', '.txt')


def test_read_all_files_not_extension():
    with pytest.raises(AssertionError):
        # Call the read_all_file_names function with a non-existent directory
        read_all_file_names('non_existent_dir', 'txt')

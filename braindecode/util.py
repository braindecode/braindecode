# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import glob
import os
import random
from warnings import warn

import h5py
import mne
import numpy as np
import torch
from sklearn.utils import check_random_state


def set_random_seeds(seed, cuda, cudnn_benchmark=None):
    """Set seeds for python random module numpy.random and torch.

    For more details about reproducibility in pytorch see
    https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    cudnn_benchmark: bool (default=None)
        Whether pytorch will use cudnn benchmark. When set to `None` it will not modify
        torch.backends.cudnn.benchmark (displays warning in the case of possible lack of
        reproducibility). When set to True, results may not be reproducible (no warning displayed).
        When set to False it may slow down computations.

    Notes
    -----
    In some cases setting environment variable `PYTHONHASHSEED` may be needed before running a
    script to ensure full reproducibility. See
    https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/14

    Using this function may not ensure full reproducibility of the results as we do not set
    `torch.use_deterministic_algorithms(True)`.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        if isinstance(cudnn_benchmark, bool):
            torch.backends.cudnn.benchmark = cudnn_benchmark
        elif cudnn_benchmark is None:
            if torch.backends.cudnn.benchmark:
                warn(
                    "torch.backends.cudnn.benchmark was set to True which may results in lack of "
                    "reproducibility. In some cases to ensure reproducibility you may need to "
                    "set torch.backends.cudnn.benchmark to False.", UserWarning)
        else:
            raise ValueError(
                f"cudnn_benchmark expected to be bool or None, got '{cudnn_benchmark}'"
            )
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def np_to_var(
    X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    warn("np_to_var has been renamed np_to_th, please use np_to_th instead")
    return np_to_th(
        X, requires_grad=requires_grad, dtype=dtype, pin_memory=pin_memory,
        **tensor_kwargs
    )


def np_to_th(
    X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


def var_to_np(var):
    warn("var_to_np has been renamed th_to_np, please use th_to_np instead")
    return th_to_np(var)


def th_to_np(var):
    """Convenience function to transform `torch.Tensor` to numpy
    array.

    Should work both for CPU and GPU."""
    return var.cpu().data.numpy()


def corr(a, b):
    """
    Computes correlation only between terms of a and terms of b, not within
    a and b.

    Parameters
    ----------
    a, b: 2darray, features x samples

    Returns
    -------
    Correlation between features in x and features in y
    """
    # Difference to numpy:
    # Correlation only between terms of x and y
    # not between x and x or y and y
    this_cov = cov(a, b)
    return _cov_to_corr(this_cov, a, b)


def cov(a, b):
    """
    Computes covariance only between terms of a and terms of b, not within
    a and b.

    Parameters
    ----------
    a, b: 2darray, features x samples

    Returns
    -------
    Covariance between features in x and features in y
    """
    demeaned_a = a - np.mean(a, axis=1, keepdims=True)
    demeaned_b = b - np.mean(b, axis=1, keepdims=True)
    this_cov = np.dot(demeaned_a, demeaned_b.T) / (b.shape[1] - 1)
    return this_cov


def _cov_to_corr(this_cov, a, b):
    # computing "unbiased" corr
    # ddof=1 for unbiased..
    var_a = np.var(a, axis=1, ddof=1)
    var_b = np.var(b, axis=1, ddof=1)
    return _cov_and_var_to_corr(this_cov, var_a, var_b)


def _cov_and_var_to_corr(this_cov, var_a, var_b):
    divisor = np.outer(np.sqrt(var_a), np.sqrt(var_b))
    return this_cov / divisor


def wrap_reshape_apply_fn(stat_fn, a, b, axis_a, axis_b):
    """
    Reshape two nd-arrays into 2d-arrays, apply function and reshape
    result back.

    Parameters
    ----------
    stat_fn: function
        Function to apply to 2d-arrays
    a: nd-array: nd-array
    b: nd-array
    axis_a: int or list of int
        sample axis
    axis_b: int or list of int
        sample axis

    Returns
    -------
    result: nd-array
        The result reshaped to remaining_dims_a + remaining_dims_b
    """
    if not hasattr(axis_a, "__len__"):
        axis_a = [axis_a]
    if not hasattr(axis_b, "__len__"):
        axis_b = [axis_b]
    other_axis_a = [i for i in range(a.ndim) if i not in axis_a]
    other_axis_b = [i for i in range(b.ndim) if i not in axis_b]
    transposed_topo_a = a.transpose(tuple(other_axis_a) + tuple(axis_a))
    n_stat_axis_a = [a.shape[i] for i in axis_a]
    n_other_axis_a = [a.shape[i] for i in other_axis_a]
    flat_topo_a = transposed_topo_a.reshape(
        np.prod(n_other_axis_a), np.prod(n_stat_axis_a)
    )
    transposed_topo_b = b.transpose(tuple(other_axis_b) + tuple(axis_b))
    n_stat_axis_b = [b.shape[i] for i in axis_b]
    n_other_axis_b = [b.shape[i] for i in other_axis_b]
    flat_topo_b = transposed_topo_b.reshape(
        np.prod(n_other_axis_b), np.prod(n_stat_axis_b)
    )
    assert np.array_equal(n_stat_axis_a, n_stat_axis_b)
    stat_result = stat_fn(flat_topo_a, flat_topo_b)
    topo_result = stat_result.reshape(
        tuple(n_other_axis_a) + tuple(n_other_axis_b)
    )
    return topo_result


def get_balanced_batches(
    n_trials, rng, shuffle, n_batches=None, batch_size=None
):
    """Create indices for batches balanced in size
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional

    Returns
    -------
    batches: list of list of int
        Indices for each batch.
    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches


def create_mne_dummy_raw(n_channels, n_times, sfreq, include_anns=True,
                         description=None, savedir=None, save_format='fif',
                         overwrite=True, random_state=None):
    """Create an mne.io.RawArray with fake data, and optionally save it.

    This will overwrite already existing files.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_times : int
        Number of samples.
    sfreq : float
        Sampling frequency.
    include_anns : bool
        If True, also create annotations.
    description : list | None
        List of descriptions used for creating annotations. It should contain
        10 elements.
    savedir : str | None
        If provided as a string, the file will be saved under that directory.
    save_format : str | list
        If `savedir` is provided, this specifies the file format the data should
        be saved to. Can be 'raw' or 'hdf5', or a list containing both.
    random_state : int | RandomState
        Random state for the generation of random data.

    Returns
    -------
    raw : mne.io.Raw
        The created Raw object.
    save_fname : dict | None
        Dictionary containing the name the raw data was saved to.
    """
    random_state = check_random_state(random_state)
    data = random_state.rand(n_channels, n_times)
    ch_names = [f'ch{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)

    if include_anns:
        n_anns = 10
        inds = np.linspace(
            int(sfreq * 2), int(n_times - sfreq * 2), num=n_anns).astype(int)
        onset = raw.times[inds]
        duration = [1] * n_anns
        if description is None:
            description = ['test'] * n_anns
        anns = mne.Annotations(onset, duration, description)
        raw = raw.set_annotations(anns)

    save_fname = dict()
    if savedir is not None:
        if not isinstance(save_format, list):
            save_format = [save_format]
        fname = os.path.join(savedir, 'fake_eeg_raw')

        if 'fif' in save_format:
            fif_fname = fname + '.fif'
            raw.save(fif_fname, overwrite=overwrite)
            save_fname['fif'] = fif_fname
        if 'hdf5' in save_format:
            h5_fname = fname + '.h5'
            with h5py.File(h5_fname, 'w') as f:
                f.create_dataset(
                    'fake_raw', dtype='f8', data=raw.get_data())
            save_fname['hdf5'] = h5_fname

    return raw, save_fname


class ThrowAwayIndexLoader(object):
    def __init__(self, net, loader, is_regression):
        self.net = net
        self.loader = loader
        self.last_i = None
        self.is_regression = is_regression

    def __iter__(self, ):
        normal_iter = self.loader.__iter__()
        for batch in normal_iter:
            if len(batch) == 3:
                x, y, i = batch
                # Store for scoring callbacks
                self.net._last_window_inds_ = i
            else:
                x, y = batch

            # TODO: should be on dataset side
            if hasattr(x, 'type'):
                x = x.type(torch.float32)
                if self.is_regression:
                    y = y.type(torch.float32)
                else:
                    y = y.type(torch.int64)
            yield x, y


def update_estimator_docstring(base_class, docstring):
    base_doc = base_class.__doc__.replace(' : ', ': ')
    idx = base_doc.find('callbacks:')
    idx_end = idx + base_doc[idx:].find('\n\n')
    # remove callback descripiton already included in braindecode docstring
    filtered_doc = base_doc[:idx] + base_doc[idx_end + 6:]
    splitted = docstring.split('Parameters\n    ----------\n    ')
    out_docstring = (
        splitted[0] +
        filtered_doc[filtered_doc.find('Parameters'):filtered_doc.find('Attributes')] +
        splitted[1] +
        filtered_doc[filtered_doc.find('Attributes'):])
    return out_docstring


def _update_moabb_docstring(base_class, docstring):
    base_doc = base_class.__doc__
    out_docstring = base_doc + f'\n\n{docstring}'
    return out_docstring


def read_all_file_names(directory, extension):
    """Read all files with specified extension from given path and sorts them
    based on a given sorting key.

    Parameters
    ----------
    directory: str
        Parent directory to be searched for files of the specified type.
    extension: str
        File extension, i.e. ".edf" or ".txt".

    Returns
    -------
    file_paths: list(str)
        List of all files found in (sub)directories of path.
    """
    assert extension.startswith('.')
    file_paths = glob.glob(directory + '**/*' + extension, recursive=True)
    assert len(file_paths) > 0, (
        f'something went wrong. Found no {extension} files in {directory}')
    return file_paths

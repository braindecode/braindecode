import os
import errno
import numpy as np


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
    topo_result = stat_result.reshape(tuple(n_other_axis_a) + tuple(n_other_axis_b))
    return topo_result


class FuncAndArgs(object):
    """Container for a function and its arguments. 
    Useful in case you want to pass a function and its arguments 
    to another function without creating a new class.
    You can call the new instance either with the apply method or 
    the ()-call operator:
    
    >>> FuncAndArgs(max, 2,3).apply(4)
    4
    >>> FuncAndArgs(max, 2,3)(4)
    4
    >>> FuncAndArgs(sum, [3,4])(8)
    15
    
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def apply(self, *other_args, **other_kwargs):
        all_args = self.args + other_args
        all_kwargs = self.kwargs.copy()
        all_kwargs.update(other_kwargs)
        return self.func(*all_args, **all_kwargs)

    def __call__(self, *other_args, **other_kwargs):
        return self.apply(*other_args, **other_kwargs)


def add_message_to_exception(exc, additional_message):
    #  give some more info...
    # see http://www.ianbicking.org/blog/2007/09/re-raising-exceptions.html
    args = exc.args
    if not args:
        arg0 = ""
    else:

        arg0 = args[0]
    arg0 += additional_message
    exc.args = (arg0,) + args[1:]


def dict_compare(d1, d2):
    """From http://stackoverflow.com/a/18860653/1469195"""
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def dict_equal(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    return (
        intersect_keys == d2_keys and intersect_keys == d1_keys and len(modified) == 0
    )


def dict_is_subset(d1, d2):
    added, removed, modified, same = dict_compare(d1, d2)
    return len(added) == 0 and len(modified) == 0


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    http://stackoverflow.com/a/26853961
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def touch_file(path):
    # from http://stackoverflow.com/a/12654798/1469195
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, "a"):
        os.utime(path, None)


def to_tuple(sequence_or_element, length=None):
    if hasattr(sequence_or_element, "__len__"):
        assert length is None
        return tuple(sequence_or_element)
    else:
        if length is None:
            return (sequence_or_element,)
        else:
            return (sequence_or_element,) * length


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def select_inverse_inds(arr, inds):
    mask = np.ones(len(arr), dtype=bool)
    mask[inds] = False
    return arr[mask]

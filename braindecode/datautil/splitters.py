import numpy as np

from braindecode.datautil.signal_target import apply_to_X_y, SignalAndTarget


def concatenate_sets(sets):
    """
    Concatenate all sets together.
    
    Parameters
    ----------
    sets: list of :class:`.SignalAndTarget`

    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    concatenated_set = sets[0]
    for s in sets[1:]:
        concatenated_set = concatenate_two_sets(concatenated_set, s)
    return concatenated_set


def concatenate_two_sets(set_a, set_b):
    """
    Concatenate two sets together.
    
    Parameters
    ----------
    set_a, set_b: :class:`.SignalAndTarget`

    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    new_X = concatenate_np_array_or_add_lists(set_a.X, set_b.X)
    new_y = concatenate_np_array_or_add_lists(set_a.y, set_b.y)
    return SignalAndTarget(new_X, new_y)

def concatenate_np_array_or_add_lists(a, b):
    if hasattr(a, 'ndim') and hasattr(b, 'ndim'):
        new = np.concatenate((a, b), axis=0)
    else:
        if hasattr(a, 'ndim'):
            a = a.tolist()
        if hasattr(b, 'ndim'):
            b = b.tolist()
        new = a + b
    return new



def split_into_two_sets(dataset, first_set_fraction=None, n_first_set=None):
    """
    Split set into two sets either by fraction of first set or by number
    of trials in first set.

    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    first_set_fraction: float, optional
        Fraction of trials in first set.
    n_first_set: int, optional
        Number of trials in first set

    Returns
    -------
    first_set, second_set: :class:`.SignalAndTarget`
        The two splitted sets.
    """
    assert (first_set_fraction is None) != (n_first_set is None), (
        "Pass either first_set_fraction or n_first_set")
    if n_first_set is None:
        n_first_set = int(round(len(dataset.X) * first_set_fraction))
    assert n_first_set < len(dataset.X)
    first_set = apply_to_X_y(lambda a: a[:n_first_set], dataset)
    second_set = apply_to_X_y(lambda a: a[n_first_set:], dataset)
    return first_set, second_set


def select_examples(dataset, indices):
    """
    Select examples from dataset.
    
    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    indices: list of int, 1d-array of int
        Indices to select

    Returns
    -------
    reduced_set: :class:`.SignalAndTarget`
        Dataset with only examples selected.
    """
    # probably not necessary
    indices = np.array(indices)
    if hasattr(dataset.X, 'ndim'):
        # numpy array
        new_X = np.array(dataset.X)[indices]
    else:
        # list
        new_X = [dataset.X[i] for i in indices]
    new_y = np.asarray(dataset.y)[indices]
    return SignalAndTarget(new_X, new_y)


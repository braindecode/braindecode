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
    if hasattr(set_a.X, 'ndim') and hasattr(set_b.X, 'ndim'):
        new_X = np.concatenate((set_a.X, set_b.X), axis=0)
    else:
        if hasattr(set_a.X, 'ndim'):
            set_a.X = set_a.X.tolist()
        if hasattr(set_b.X, 'ndim'):
            set_b.X = set_b.X.tolist()
        new_X = set_a.X + set_b.X
    new_y = np.concatenate((set_a.y, set_b.y), axis=0)
    return SignalAndTarget(new_X, new_y)


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


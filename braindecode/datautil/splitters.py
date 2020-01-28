import numpy as np

from braindecode.datasets.croppedxy import CroppedXyDataset
from braindecode.util import get_balanced_batches
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
    if hasattr(a, "ndim") and hasattr(b, "ndim"):
        new = np.concatenate((a, b), axis=0)
    else:
        if hasattr(a, "ndim"):
            a = a.tolist()
        if hasattr(b, "ndim"):
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
    assert (first_set_fraction is None) != (
        n_first_set is None
    ), "Pass either first_set_fraction or n_first_set"
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
    if hasattr(dataset.X, "ndim"):
        # numpy array
        new_X = np.array(dataset.X)[indices]
    else:
        # list
        new_X = [dataset.X[i] for i in indices]
    new_y = np.asarray(dataset.y)[indices]
    return SignalAndTarget(new_X, new_y)


def split_into_train_valid_test(dataset, n_folds, i_test_fold, rng=None):
    """
    Split datasets into folds, select one valid fold, one test fold and merge rest as train fold.

    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    n_folds: int
        Number of folds to split dataset into.
    i_test_fold: int
        Index of the test fold (0-based). Validation fold will be immediately preceding fold.
    rng: `numpy.random.RandomState`, optional
        Random Generator for shuffling, None means no shuffling

    Returns
    -------
    reduced_set: :class:`.SignalAndTarget`
        Dataset with only examples selected.
    """
    n_trials = len(dataset.X)
    if n_trials < n_folds:
        raise ValueError(
            "Less Trials: {:d} than folds: {:d}".format(n_trials, n_folds)
        )
    shuffle = rng is not None
    folds = get_balanced_batches(n_trials, rng, shuffle, n_batches=n_folds)
    test_inds = folds[i_test_fold]
    valid_inds = folds[i_test_fold - 1]
    all_inds = list(range(n_trials))
    train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
    assert np.intersect1d(train_inds, valid_inds).size == 0
    assert np.intersect1d(train_inds, test_inds).size == 0
    assert np.intersect1d(valid_inds, test_inds).size == 0
    assert np.array_equal(
        np.sort(np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
        all_inds,
    )

    train_set = select_examples(dataset, train_inds)
    valid_set = select_examples(dataset, valid_inds)
    test_set = select_examples(dataset, test_inds)

    return train_set, valid_set, test_set


def split_into_train_test(dataset, n_folds, i_test_fold, rng=None):
    """
     Split datasets into folds, select one test fold and merge rest as train fold.

    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    n_folds: int
        Number of folds to split dataset into.
    i_test_fold: int
        Index of the test fold (0-based)
    rng: `numpy.random.RandomState`, optional
        Random Generator for shuffling, None means no shuffling

    Returns
    -------
    reduced_set: :class:`.SignalAndTarget`
        Dataset with only examples selected.
    """
    n_trials = len(dataset.X)
    if n_trials < n_folds:
        raise ValueError(
            "Less Trials: {:d} than folds: {:d}".format(n_trials, n_folds)
        )
    shuffle = rng is not None
    folds = get_balanced_batches(n_trials, rng, shuffle, n_batches=n_folds)
    test_inds = folds[i_test_fold]
    all_inds = list(range(n_trials))
    train_inds = np.setdiff1d(all_inds, test_inds)
    assert np.intersect1d(train_inds, test_inds).size == 0
    assert np.array_equal(np.sort(np.union1d(train_inds, test_inds)), all_inds)

    train_set = select_examples(dataset, train_inds)
    test_set = select_examples(dataset, test_inds)
    return train_set, test_set


class TrainTestSplit(object):
    """
    Class to perform splitting on a dataset with just X,y attributes.

    TODO: try make this work without supplying input time length
    and n_preds_per_trial, they are only passed to CroppedDatasetXy

    Parameters
    ----------
    train_size: int or float
        Train size in number of trials or fraction of trials
    input_time_length: int
        Input time length aka supercrop size in number of samples.
    n_preds_per_input:
        Number of predictions per supercrop (=> will be supercrop stride)
        in number of samples.
    """
    def __init__(
            self, train_size, input_time_length, n_preds_per_input):
        assert isinstance(train_size, (int, float))
        self.train_size = train_size
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def __call__(self, dataset, y, **kwargs):
        # can we directly use this https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        # or stick to same API
        if isinstance(self.train_size, int):
            n_train_samples = self.train_size
        else:
            n_train_samples = int(self.train_size * len(dataset))
        X, y = dataset.X, dataset.y
        return (
            CroppedXyDataset(
                X[:n_train_samples],
                y[:n_train_samples],
                input_time_length=self.input_time_length,
                n_preds_per_input=self.n_preds_per_input,
            ),
            CroppedXyDataset(
                X[n_train_samples:],
                y[n_train_samples:],
                input_time_length=self.input_time_length,
                n_preds_per_input=self.n_preds_per_input,
            ),
        )

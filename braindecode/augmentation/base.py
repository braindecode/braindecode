# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

from typing import List, Tuple, Any
from numbers import Real
from collections.abc import Iterable

import pandas as pd
from sklearn.utils import check_random_state
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from braindecode.augmentation.functionals import identity

Batch = List[Tuple[torch.Tensor, int, Any]]
Output = Tuple[torch.Tensor, torch.Tensor]


class Transform(torch.nn.Module):
    """ Basic transform class used for implementing data augmentation
    operations

    Parameters
    ----------
    operation : callable
        A function taking arrays X, y (sample features and
        target resp.) and other required arguments, and returning the
        transformed X and y.
    probability : float, optional
        Float between 0 and 1 defining the uniform probability of
        applying the operation. Set to 1.0 by default (e.g always apply the
        operation).
    magnitude : float | None, optional
        Defines the strength of the transformation applied between 0 and 1 and
        depends on the nature of the transformation and on its range. Some
        transformations don't have any magnitude (=None). It can be equivalent
        to another argument of object with more meaning. In case both are
        passed, magnitude will override the latter. Defaults to None.
    mag_range : tuple of two floats | None, optional
        Valid range of the argument mapped by `magnitude` (e.g. standard
        deviation, number of sample, etc.):
        ```
        argument = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0].
        ```
        If `magnitude` is None it is ignored. Defaults to None.
    random_state: int, optional
        Seed to be used to instatiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    *args:
        Arguments to be passed to operation.
    **kwargs:
        Keyword arguments to be passed to operation.
    """

    def __init__(self, operation, probability=1.0, magnitude=None,
                 mag_range=None, random_state=None, *args, **kwargs):
        super().__init__()
        assert callable(operation), "operation should be a `callable`."
        self.operation = operation
        assert isinstance(probability, Real), (
            f"probability should be a `real`. Got {type(probability)}.")
        assert probability <= 1. and probability >= 0., \
            "probability should be between 0 and 1."
        self._probability = probability
        self.rng = check_random_state(random_state)
        assert (
            (
                isinstance(magnitude, Real) and
                0 <= magnitude <= 1
            ) or magnitude is None
        ), "magnitude can be either a float between 0 and 1 or None"
        self._magnitude = magnitude
        assert (
            (
                isinstance(mag_range, Iterable) and
                len(mag_range) == 2 and
                all([isinstance(v, Real) for v in mag_range])
            ) or mag_range is None
        ), "mag_range should be None or a tuple of two floats."
        self.mag_range = mag_range
        self.args = args
        self.kwargs = kwargs

    def forward(self, X: Tensor, y: Tensor) -> Output:
        mask = self._get_mask(X.shape[0])
        tr_X, tr_y = self.operation(
            X.clone(), y.clone(), *self.args,
            random_state=self.rng, magnitude=self.magnitude, **self.kwargs)

        mask.squeeze_()
        out_X, out_y = X.clone(), y.clone()
        num_valid = mask.sum().long()
        if num_valid > 0:
            out_X[mask == 1, ...] = tr_X[mask == 1, ...]
            if type(tr_y) is tuple:
                out_y = tuple(tmp_y[mask == 1] for tmp_y in tr_y)
            else:
                out_y[mask == 1] = tr_y[mask == 1]
        return out_X, out_y

    def _get_mask(self, batch_size=None) -> torch.Tensor:
        """Samples whether to apply operation or not over the whole batch
        """
        size = (batch_size, 1, 1)
        return torch.as_tensor(self.probability > self.rng.uniform(size=size))

    # Might seem like an overkill, but making probability and magnitude into
    # properties is useful in context where Transform is overclassed to make
    # them learnable parameters.
    @property
    def probability(self):
        return self._probability

    @property
    def magnitude(self):
        return self._magnitude

    def get_structure(self):
        """ Returns a dictionary describing the transform """
        return {
            "operation": type(self).__name__,
            "probability": self.probability,
            "magnitude": self.probability,
        }


class IdentityTransform(Transform):
    """ Identity transform

    Transform that does not change the input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(operation=identity)


class Compose(Transform):
    """ Transform composition

    Callable class allowing to cast a sequence of Transform objects into a
    single one.

    Parameters
    ----------
    transforms: list
        Sequence of Transforms to be composed.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        super().__init__(operation=identity)

    def forward(self, X, y):
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y

    def get_structure(self):
        """ Returns a DataFrame describing the transforms making the object"""
        structure = list()
        for i, transform in enumerate(self.transforms):
            transform_struct = transform.get_structure()
            transform_struct.update({"transform_idx": i})
            structure.append(transform_struct)
        res = pd.DataFrame(structure)
        return res


def make_collateable(transform):
    def _collate_fn(batch):
        X, y, _ = default_collate(batch)
        return transform(X, y)
    return _collate_fn


class BaseDataLoader(DataLoader):
    """ A base dataloader class customized to applying augmentation Transforms.

    Parameters
    ----------
    dataset: BaseDataset
    transforms: list | Transform, optional
        Transform or sequence of Transforms to be applied to each batch.
    *args: arguments to pass to standard DataLoader class. Defaults to None.
    **kwargs: keyword arguments to pass to standard DataLoader class.
    """

    def __init__(self, dataset, transforms=None, *args, **kwargs):
        # TODO: Add message when collate_fn is called or allow composing
        if transforms is None or (
            isinstance(transforms, list) and len(transforms) == 0
        ):
            self.collated_tr = make_collateable(IdentityTransform())
        elif isinstance(transforms, (Transform, nn.Module)):
            self.collated_tr = make_collateable(transforms)
        elif isinstance(transforms, list):
            self.collated_tr = make_collateable(Compose(transforms))
        else:
            raise TypeError("transforms can be either a Transform object" +
                            " or a list of Transform objects.")

        super().__init__(
            dataset,
            collate_fn=self.collated_tr,
            *args,
            **kwargs
        )

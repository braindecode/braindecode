# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

from typing import List, Tuple, Any
from numbers import Real

import pandas as pd
from sklearn.utils import check_random_state
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from braindecode.augmentation._functionals import identity

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
    random_state: int, optional
        Seed to be used to instatiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    *args: tuple
        Arguments to be passed to operation.
    **kwargs: dict, optional
        Keyword arguments to be passed to operation.
    """

    def __init__(self, operation, probability=1.0, magnitude=None,
                 random_state=None, *args, **kwargs):
        super().__init__()
        if self.forward.__func__ is Transform.forward:
            assert callable(operation), "operation should be a ``callable``."

        self.operation = operation
        assert isinstance(probability, Real), (
            f"probability should be a ``real``. Got {type(probability)}.")
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
        self.args = args
        self.kwargs = kwargs

    def forward(self, X: Tensor, y: Tensor) -> Output:
        """ General forward pass for an augmentation transform

        Parameters
        ----------
        X : Tensor
            EEG input batch.
        y : Tensor
            EEG labels for the batch.

        Returns
        -------
        Tensor
            Transformed inputs.
        Tensor
            Transformed labels (usually unchanged).
        """
        # Apply the operation defining the Transform to the whole batch
        tr_X, tr_y = self.operation(
            X.clone(), y.clone(), *self.args,
            random_state=self.rng, magnitude=self.magnitude, **self.kwargs)

        # Samples a mask setting for each example whether they should stay
        # inchanged or not
        mask = self._get_mask(X.shape[0])

        # Uses the mask to define the output
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
        mask = torch.as_tensor(self.probability > self.rng.uniform(size=size))
        return mask.squeeze_()

    @property
    def probability(self):
        return self._probability

    @property
    def magnitude(self):
        return self._magnitude

    def to_dict(self):
        """ Returns a dictionary describing the transform """
        return {
            "operation": type(self).__name__,
            "probability": self.probability,
            "magnitude": self.probabilitsy,
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
        super().__init__(operation=None)

    def forward(self, X, y):
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y

    def to_dict(self):
        """ Returns a DataFrame describing the transforms making the object"""
        structure = list()
        for i, transform in enumerate(self.transforms):
            transform_struct = transform.to_dict()
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
    dataset : BaseDataset
    transforms : list | Transform, optional
        Transform or sequence of Transforms to be applied to each batch.
    *args : tuple
        arguments to pass to standard DataLoader class. Defaults to None.
    **kwargs : dict, optional
        keyword arguments to pass to standard DataLoader class.

    """

    def __init__(self, dataset, transforms=None, *args, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "collate_fn cannot be used in this context because it is used "
                "to pass transform"
            )
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

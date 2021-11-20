# Authors: Cédric Rommel <cedric.rommel@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from typing import List, Tuple, Any
from numbers import Real

from sklearn.utils import check_random_state
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .functional import identity

Batch = List[Tuple[torch.Tensor, int, Any]]
Output = Tuple[torch.Tensor, torch.Tensor]


class Transform(torch.nn.Module):
    """Basic transform class used for implementing data augmentation
    operations.

    Parameters
    ----------
    operation : callable
        A function taking arrays X, y (inputs and targets resp.) and
        other required arguments, and returning the transformed X and y.
    probability : float, optional
        Float between 0 and 1 defining the uniform probability of applying the
        operation. Set to 1.0 by default (e.g always apply the operation).
    random_state: int, optional
        Seed to be used to instatiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = None

    def __init__(self, probability=1.0, random_state=None):
        super().__init__()
        if self.forward.__func__ is Transform.forward:
            assert callable(self.operation),\
                "operation should be a ``callable``."

        assert isinstance(probability, Real), (
            f"probability should be a ``real``. Got {type(probability)}.")
        assert probability <= 1. and probability >= 0., \
            "probability should be between 0 and 1."
        self._probability = probability
        self.rng = check_random_state(random_state)

    def get_params(self, *batch):
        return dict()

    def forward(self, X: Tensor, y: Tensor = None) -> Output:
        """General forward pass for an augmentation transform.

        Parameters
        ----------
        X : torch.Tensor
            EEG input example or batch.
        y : torch.Tensor | None
            EEG labels for the example or batch. Defaults to None.

        Returns
        -------
        torch.Tensor
            Transformed inputs.
        torch.Tensor, optional
            Transformed labels. Only returned when y is not None.
        """
        X = torch.as_tensor(X).float()
        out_X = X.clone()
        if y is not None:
            y = torch.as_tensor(y)
            out_y = y.clone()
        else:
            out_y = torch.zeros(X.shape[0])

        # Samples a mask setting for each example whether they should stay
        # inchanged or not
        mask = self._get_mask(X.shape[0])
        num_valid = mask.sum().long()

        if num_valid > 0:
            # Uses the mask to define the output
            out_X[mask, ...], tr_y = self.operation(
                out_X[mask, ...], out_y[mask],
                **self.get_params(out_X[mask, ...], out_y[mask])
            )
            # Apply the operation defining the Transform to the whole batch
            if type(tr_y) is tuple:
                out_y = tuple(tmp_y[mask] for tmp_y in tr_y)
            else:
                out_y[mask] = tr_y

        if y is not None:
            return out_X, out_y
        else:
            return out_X

    def _get_mask(self, batch_size=None) -> torch.Tensor:
        """Samples whether to apply operation or not over the whole batch
        """
        return torch.as_tensor(
            self.probability > self.rng.uniform(size=batch_size)
        )

    @property
    def probability(self):
        return self._probability


class IdentityTransform(Transform):
    """Identity transform.

    Transform that does not change the input.
    """
    operation = staticmethod(identity)


class Compose(Transform):
    """Transform composition.

    Callable class allowing to cast a sequence of Transform objects into a
    single one.

    Parameters
    ----------
    transforms: list
        Sequence of Transforms to be composed.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        super().__init__()

    def forward(self, X, y):
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y


def _make_collateable(transform):
    def _collate_fn(batch):
        collated_batch = default_collate(batch)
        X, y = collated_batch[:2]
        return (*transform(X, y), *collated_batch[2:])
    return _collate_fn


class AugmentedDataLoader(DataLoader):
    """A base dataloader class customized to applying augmentation Transforms.

    Parameters
    ----------
    dataset : BaseDataset
        The dataset containing the signals.
    transforms : list | Transform, optional
        Transform or sequence of Transform to be applied to each batch.
    **kwargs : dict, optional
        keyword arguments to pass to standard DataLoader class.
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "collate_fn cannot be used in this context because it is used "
                "to pass transform"
            )
        if transforms is None or (
            isinstance(transforms, list) and len(transforms) == 0
        ):
            self.collated_tr = _make_collateable(IdentityTransform())
        elif isinstance(transforms, (Transform, nn.Module)):
            self.collated_tr = _make_collateable(transforms)
        elif isinstance(transforms, list):
            self.collated_tr = _make_collateable(Compose(transforms))
        else:
            raise TypeError("transforms can be either a Transform object" +
                            " or a list of Transform objects.")

        super().__init__(
            dataset,
            collate_fn=self.collated_tr,
            **kwargs
        )

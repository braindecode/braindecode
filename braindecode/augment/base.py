# Authors: Simon Freyburger
#
# License: BSD-3

import torch
from functools import partial
import numpy as np


class Transform:
    """This is a framework that unifies transforms, so that they follow the
    structure used by most of the papers studying automatic data augmentation:
    a Transform is defined as an operation `t` applied with a magnitude `m`
    with probability `p`. Note that you can later compose Transforms using
    the Compose class of torchvision.

    Parameters
    ----------
    operation : Callable((Datum, int (optional)), Datum)
        A function taking a Datum object, and eventually a magnitude
        argument, and returning the transformed Datum. Omitting the
        `int` argument requires to set magnitude to None, and vice-versa.
    probability : int, optional
        The probability the function should be applied with, None if
        the operation is always applied.(default=None)
    magnitude : int, optional
        An amplitude parameter, define how much the transformation will
        alter the data, may be omitted if the transform does not need
        it (ex: reverse signal), (default=None)
    required_variables :
        dict(string:
                Callable((WindowsDataset or WindowsConcatDataset), any)),
        optional. A dictionary where keys are required variable name, and
        values are a function that computes this variable given an
        unaugmented dataset. (default={})
    """

    def __init__(self, operation, probability=None,
                 params={}, required_variables={}):
        self.probability = probability
        self.params = params
        if params:
            self.operation = partial(operation, params=params)
        else:
            self.operation = operation
        self.required_variables = required_variables

    def __call__(self, datum):
        """Apply the transform ``self.operation`` on the data X with
        probability ``self.probability`` and magnitude ``self.magnitude``

        Parameters
        ----------
        datum : Datum
            Data + metadata

        Returns
        -------
        Datum
            Transformed data + metadata
        """

        if self.probability is not None:
            rand_num = np.random.random()
            if rand_num >= self.probability:
                return datum
        return self.operation(datum)


"""Custom iterator for training epochs
"""

# Authors: Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD (3-clause)


class mixup_iterator(torch.utils.data.DataLoader):
    """Implements Iterator for Mixup for EEG data. See [mixup].
    Code adapted from
    #TODO ref sbbrandt
    Parameters
    ----------
    dataset: Dataset
        dataset from which to load the data.
    alpha: float
        mixup hyperparameter.
    beta_per_sample: bool (default=False)
        by default, one mixing coefficient per batch is drawn from an beta
        distribution. If True, one mixing coefficient per sample is drawn.
    References
    ----------
    ..  [mixup] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        mixup: Beyond Empirical Risk Minimization
        Online: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, dataset, alpha, beta_per_sample=False, **kwargs):
        super().__init__(dataset, collate_fn=self.mixup, **kwargs)

    def mixup(self, data):
        X, y, crop_inds = data
        x = torch.tensor(X).type(torch.float32)
        y = torch.tensor(y).type(torch.int64)
        crop_inds = torch.tensor(crop_inds).type(torch.int64)

        return x, y, crop_inds

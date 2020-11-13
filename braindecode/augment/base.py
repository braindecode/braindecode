# Authors: Simon Freyburger
#
# License: BSD-3

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
                 magnitude=None, required_variables={}):
        self.operation = operation
        self.probability = probability
        self.magnitude = magnitude
        if magnitude:
            self.transform = partial(operation, magnitude=magnitude)
        else:
            self.transfrom = operation
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
        return self.transform(datum)

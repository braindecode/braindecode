# Authors: Simon Freyburger
#
# License: BSD-3

from functools import partial
import numpy as np


class Transform:

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

        Args:
            datum (Datum): Data + metadata

        Returns:
            datum: Transformed data + metadata
        """
        if self.probability is not None:
            rand_num = np.random.random()
            if rand_num >= self.probability:
                return datum
        return self.transform(datum)

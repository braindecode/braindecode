from functools import partial
import numpy as np


class Transform:

    def __init__(self, operation, probability=None, magnitude=0):
        self.operation = operation
        self.probability = probability
        self.magnitude = magnitude
        self.transform = partial(operation, magnitude=magnitude)
        self.random_state = None

    def __call__(self, datum, random_state=None):
        """Apply the transform ``self.operation`` on the data X with
        probability ``self.probability`` and magnitude ``self.magnitude``

        Args:
            datum (Datum): Data + metadata

        Returns:
            datum: Transformed data + metadata
        """
        np.random.set_state(random_state)
        if self.probability is not None:
            rand_num = np.random.random()
            if rand_num <= self.probability:
                return self.transform(datum)
        return datum

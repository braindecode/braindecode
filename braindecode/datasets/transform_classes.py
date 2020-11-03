from functools import partial
import random


class Transform:
    
    def __init__(self, operation, probability=1, magnitude=0):
        self.operation = operation
        self.probability = probability
        self.magnitude = magnitude
        self.transform = partial(operation, magnitude=magnitude)
    
    def __call__(self, X):
        rand_num = random.random()
        if rand_num <= self.probability:
            return self.transform(X)
        else:
            return X
    
    
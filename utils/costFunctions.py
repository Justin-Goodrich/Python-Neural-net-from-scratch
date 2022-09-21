import numpy as np

def squaredError(actual,expected):
    return 0.5 * np.square(expected - actual)

def squaredErrorPrime(actual,expected):
    return (actual-expected)
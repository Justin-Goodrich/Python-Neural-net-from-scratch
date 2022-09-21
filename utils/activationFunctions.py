import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))
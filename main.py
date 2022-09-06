from Layer import Layer
from Network import Network
import numpy as np

# nummpy automatically applies the function elementwise
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))

out = Layer(2, activation=sigmoid, activationDerivative=sigmoidPrime)

input = np.matrix([[1],[0]])

NN = Network(outputLayer=out, inputNodes=2)

NN.intializeWeights()
print(NN.outputLayer.weights + NN.outputLayer.bias)

print(NN.forwardPropagate(input))


print(NN.outputLayer.bias)
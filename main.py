from Layer import Layer, outputLayer
from Network import Network
import numpy as np

from TrainingExample import TrainingExample

# nummpy automatically applies the function elementwise
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def squaredError(actual,expected):
    return 0.5 * np.square(expected - actual)

def squaredErrorPrime(actual,expected):
    return actual-expected


examples = [
    TrainingExample([[1],[0]],[[1]]),
    TrainingExample([[1],[1]],[[0]]),
    TrainingExample([[0],[0]],[[0]]),
    TrainingExample([[0],[1]],[[1]]),
]
out = outputLayer(1,activation=sigmoid, activationDerivative=sigmoidPrime)

NN = Network(outputLayer=out, cost=squaredError, costDerivative = squaredErrorPrime,learningRate=.2, inputNodes=2)
NN.addLayer(Layer(2,sigmoid,sigmoidPrime))
NN.intializeWeights()

# print("\n\n")

for i in range(2):
    NN.train(examples=examples)





print("weights")
print(NN.firstLayer.weights)

# res = NN.forwardPropagate(np.matrix([[0],[1]]))
# print(res)
# NN.firstLayer.computeError(squaredErrorPrime([[1]],res))
# NN.firstLayer.adjustWeights(.5)


NN.train(examples=examples)

print("weights")
print(NN.firstLayer.weights)


print(NN.forwardPropagate(input=[[0],[1]]))
print(NN.forwardPropagate(input=[[0],[0]]))
# print(NN.forwardPropagate(input=[[1],[1]]))




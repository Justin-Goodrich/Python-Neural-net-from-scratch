from Layer import Layer
from Network import Network
import numpy as np

# i = np.matrix([[0],[1]])
# x = Layer(2)
# x.addLayer(outputLayer(2))
# x.randomizeWeights()
# print(x.weights.getA())
# print("\n")
# print(x.computeOutput(i))

out = Layer(2)


NN = Network(outputLayer=out, inputNodes=2)

# input.randomizeWeights()
NN.intializeWeights()

print(NN.outputLayer.weights)
print(NN.outputLayer.bias)
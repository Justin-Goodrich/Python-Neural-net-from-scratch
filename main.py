from Layer import Layer, outputLayer
from Network import Network
import numpy as np

# i = np.matrix([[0],[1]])
# x = Layer(2)
# x.addLayer(outputLayer(2))
# x.randomizeWeights()
# print(x.weights.getA())
# print("\n")
# print(x.computeOutput(i))

out = outputLayer(1)
input = Layer(1)
NN = Network(inputLayer=input, outputLayer=out)
input.randomizeWeights()

print(NN.inputLayer.weights)
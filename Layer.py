from random import random
import numpy as np

class Layer:
    def __init__(self,nodes, activation, activationDerivative, next=None, prev=None):
     

        self.weights = None
        self.bias = None
        self.nodes = nodes
        self.output = 0

        self.activation = activation
        self.activationDerivative = activationDerivative
           
        self.next = next
        self.prev = prev

      


    def computeOutput(self, input):
        # recursive fuction to be called from the input layer only 

        self.output = self.weights * input
        self.output = self.output + self.bias
        self.output = self.activation(self.output)
        if self.next is None:
            return self.output
        
        return self.next.computeOutput(self.output)

    def addLayer(self,L):
        L.prev = self
        self.next = L

    
    def randomizeWeights(self, nodes):
        weightmatrix = []
        for i in range(nodes):
            row = []
            for j in range(self.nodes):
                row.append(random())
            weightmatrix.append(row)
        self.weights = np.matrix(weightmatrix)

        if self.next is not None:
            self.next.randomizeWeights(self.nodes)

    def intializeBias(self):
        self.bias = np.matrix([1 for i in range(self.nodes)])

        if self.next is not None:
            self.next.initializeBias()


# class outputLayer(Layer):
#     def computeOutput(self, input):
#         self.output = input
#         return input

#     def randomizeWeights(self, inputNodes):
#         # needed to end the recursive loop 
#         return 

#     def intiializeBias(self):
#         return 

    




    
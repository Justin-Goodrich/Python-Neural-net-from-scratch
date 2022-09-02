from random import random
import numpy as np

class Layer:

    def __init__(self,nodes,weights=None,next=None,prev=None):
        self.weights = weights
        self.nodes = nodes
        self.prev = prev
        self.next = next
        self.output = None

    def computeOutput(self, input):
        self.output = self.weights * input

        if self.next is None:
            return self.output
        
        return self.next.computeOutput(self.output)

    def addLayer(self,L):
        L.prev = self
        self.next = L

    
    def randomizeWeights(self):
        weightmatrix = []
        for i in range(self.next.nodes):
            row = []
            for j in range(self.nodes):
                row.append(random())
            weightmatrix.append(row)
        self.weights = np.matrix(weightmatrix)


class outputLayer(Layer):
    def computeOutput(self, input):
        self.output = input
        return input



    
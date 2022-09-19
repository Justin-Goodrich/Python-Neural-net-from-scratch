from random import random
import numpy as np

class Layer:
    def __init__(self,nodes, activation, activationDerivative, next=None, prev=None):
     

        self.weights = None
        self.bias = 0
        self.nodes = nodes
        self.output = 0
        self.weightedSum = 0

        self.activation = activation
        self.activationDerivative = activationDerivative
           
        self.next = next
        self.prev = prev
        self.error = 0

    def computeOutput(self, input):
        # recursive fuction to be called from the input layer only 
        # the @ symbol is the operator for matrix multiplication

        # self.weightedSum = (self.weights @ input) + self.bias
        self.weightedSum = np.matmul(self.weights,input) + self.bias



        # self.weightedSum = np.add(self.weightedSum,self.bias)
        # print("w/o")
        # print(self.weightedSum)
        # print("bias")
        # print(np.add(self.weightedSum,1))
        # self.weightedSum = self.weightedSum 

        self.output = self.activation(self.weightedSum)


        if self.next is None:
            return self.output
        
        return self.next.computeOutput(self.output)

    def addLayer(self,L):
        L.prev = self
        self.next = L

    def randomizeWeights(self, nodes):
        # weightmatrix = []
        # for i in range(self.nodes):
        #     row = []
        #     for j in range(nodes):
        #         row.append(random())
        #     weightmatrix.append(row)
        # self.weights = np.matrix(weightmatrix)

        
        self.weights = np.random.rand(self.nodes,nodes)
        if self.next is not None:
            self.next.randomizeWeights(self.nodes)

    def intializeBias(self):
        self.bias = np.matrix([1 for i in range(self.nodes)])

        if self.next is not None:
            self.next.intializeBias()

    def computeError(self):
        # self.weights.getT() @ self.next.error
        # e = np.transpose(self.next.weights) * self.next.error
        # e = np.multiply(self.next.weights.getT(),  self.next.error)
        e = np.matmul(self.next.weights.getT(),self.next.error)
        # dot(self.next.weights.getT(),self.next.error)
        # print("\n")
        # print(e)
        # print("\n")
        
        # print(e)
        self.error = np.multiply(e,self.activationDerivative(self.weightedSum))
        # print("w")
        # print(self.next.weights.getT())
        # print(self.next.error)
        # print("prod")
        # print()
        # return self.error

    def adjustWeights(self, learningRate, input=None):
        # nabla_w = np.dot(self.output.getT(), self.error)
        nabla_w = np.dot(self.output.getT(), self.error)

        # if self.prev is not None:
        #     adjustment = np.multiply(self.prev.output.getT(),self.error)
        # else:
        #     adjustment = np.multiply(input.getT(),self.error)

        # print(adjustment)
        # adjustment = np.transpose(self.weightedSum).dot(self.error)
        # print(self.weights)
        # print("\n")
        # print(adjustment)
        self.weights = np.subtract(self.weights,(learningRate * nabla_w))






class outputLayer(Layer):
    def computeError(self, costGradient):
        
        self.error = np.multiply(costGradient,self.activationDerivative(self.weightedSum))
  


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
  

def sigmoidPrime(z):
    return np.dot(sigmoid(z),(1-sigmoid(z)))



    
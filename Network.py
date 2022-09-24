from utils.TrainingExample import TrainingExample
import numpy as np
from utils.costFunctions import squaredError, squaredErrorPrime
from utils.activationFunctions import sigmoid, sigmoidPrime

class Network :
    def __init__(self,learningRate, inputNodes, activation, activationDerivative,costFunction,costDerivative) -> None:
        self.layers = []
        self.bias = []
        self.activations = []
        self.learningRate = learningRate
        self.inputNodes = inputNodes
        self.activation = activation
        self.activationDerivative = activationDerivative
        self.costFunction = costFunction
        self.costDerivative = costDerivative

    def addLayer(self, nodes):
        if len(self.layers) == 0:
            # self.layers.append(np.random.rand(nodes,self.inputNodes)+1)
            self.layers.append(np.random.rand(nodes,self.inputNodes))

        else:
            prev = self.layers[-1].shape[0]
            # self.layers.append(np.random.rand(nodes,prev)+1)
            self.layers.append(np.random.rand(nodes,prev))

            
   
        self.bias.append(np.random.rand(1,1))

        


    def forwardPropagate(self,input):
        biasIndex = 0
        for W in self.layers:
            weightedOutput = np.matmul(W,input)  
            # weightedOutput = np.dot(input,W)  
            input = self.activation(weightedOutput)
            biasIndex+=1
        return input

    def backPropagate(self, input, costGradient):
        outputs = []
        
        activation = input
        activations = [input]

        
        for W in self.layers:
            weightedOutput = np.dot(W,activation)
            # weightedOutput = np.dot(input,W) 
            outputs.append(weightedOutput)
            activation = self.activation(weightedOutput)
            activations.append(activation)

        
        error = np.multiply(costGradient,self.activationDerivative(outputs[-1]))
        nabla_w = []
        for i in range(1,len(self.layers)+1):
            w = self.layers[-i]
            delta_w = np.dot(error,activations[-i-1].transpose())
            nabla_w.append(delta_w)
            self.layers[-i] = self.layers[-i] - (self.learningRate * delta_w)
            if i < len(self.layers): 
                error = np.multiply(np.dot(w.transpose(),error), self.activationDerivative(outputs[-i-1]))





    def fit(self, trainingData, epochs):
        for e in range(epochs):
            for i in trainingData:
                actual = self.forwardPropagate(i.input)
                costGradient = self.costDerivative(actual,i.expected)
                self.backPropagate(i.input,costGradient)


            


      
        





from audioop import bias
import numpy as np

class Network :
    def __init__(self,learningRate, inputNodes, activation, activationDerivative,costFunction,costDerivative) -> None:
        self.layers = []
        self.bias = []
        self.activations = []
        self.activationDerivatives = []
        self.learningRate = learningRate
        self.inputNodes = inputNodes
        self.activation = activation
        self.activationDerivative = activationDerivative
        self.costFunction = costFunction
        self.costDerivative = costDerivative

    def addLayer(self, nodes, acivationfunction, activationDerivative):
        if len(self.layers) == 0:
            self.layers.append(np.random.rand(nodes,self.inputNodes)*2-1)

        else:
            prev = self.layers[-1].shape[0]
            self.layers.append(np.random.rand(nodes,prev)*2-1)

            
        self.bias.append(np.random.rand(1,1))
        self.activations.append(acivationfunction)
        self.activationDerivatives.append(activationDerivative)
        

        


    def forwardPropagate(self,input):
        biasIndex = 0
        for W, B, A in zip(self.layers,self.bias, self.activations):
            weightedOutput = np.matmul(W,input) + B
            input = A(weightedOutput)
            biasIndex+=1
        return input

    def backPropagate(self, input, costGradient):
        outputs = []
        
        activation = input
        activations = [input]

        
        for W, B, A in zip(self.layers,self.bias, self.activations):
            weightedOutput = np.dot(W,activation) + B
            outputs.append(weightedOutput)
            activation = A(weightedOutput)
            activations.append(activation)

        
        error = np.multiply(costGradient,self.activationDerivatives[-1](outputs[-1]))

        nabla_w = []
        for i in range(1,len(self.layers)+1):
            w = self.layers[-i]
            delta_w = np.dot(error,activations[-i-1].transpose())
            nabla_w.append(delta_w)
            self.layers[-i] = self.layers[-i] - (self.learningRate * delta_w)
            self.bias[-i] = self.bias[-i] -  (self.learningRate * error)
            if i < len(self.layers): 
                error = np.multiply(np.dot(w.transpose(),error), self.activationDerivatives[-i-1](outputs[-i-1]))





    def fit(self, trainingData, epochs):
        for e in range(epochs):
            for i in trainingData:
                actual = self.forwardPropagate(i.input)
                costGradient = self.costDerivative(actual,i.expected)
                self.backPropagate(i.input,costGradient)


            


      
        





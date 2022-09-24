import sys
# path must be changed to locate modules outside of directory
sys.path.append("../../")

from Network import Network
from utils.activationFunctions import sigmoid, sigmoidPrime
from utils.costFunctions import squaredError, squaredErrorPrime
from utils.TrainingExample import TrainingExample

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch  
np.set_printoptions(threshold=1000)
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


class Mnist_predictor(Network):
    def fit(self, trainingData, epochs):
        for e in range(epochs):
            for i in trainingData:
                output_vector = np.array([[1] if x is i[1] else [0] for x in range(10)])
                input = torch.reshape(i[0],(784,1)).numpy()
                actual = self.forwardPropagate(input)
                costGradient = self.costDerivative(actual,output_vector)
                print(actual.tolist())
                print()

                # print(actual-output_vector)
                # print("epoch: {} error:{}".format(e, self.costFunction(output_vector,actual).flatten()))
                self.backPropagate(input,costGradient)


            
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




        

learningRate = 0.02

MNIST = Mnist_predictor(learningRate,784,sigmoid,sigmoidPrime,squaredError,squaredErrorPrime)

MNIST.addLayer(128)
MNIST.addLayer(64)
MNIST.addLayer(10)

test = test_data[0]

prediction = MNIST.forwardPropagate(torch.reshape(test[0],(784,1)).numpy())
# print(prediction)
# print("untrained prediction: ")
# print("number: {}\nprediction:{}".format(test[1],prediction))

MNIST.fit(test_data,100)

# newprediction = MNIST.forwardPropagate(torch.reshape(test[0],(784,1)).numpy())

# print("trained prediction: ")
# print("number: {}\nprediction:{}".format(test[1],newprediction))



# only outputing ones 
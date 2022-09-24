import sys
# path must be changed to locate modules outside of directory
sys.path.append("../../")
from Network import Network
from utils.TrainingExample import TrainingExample
from utils.costFunctions import squaredError, squaredErrorPrime
from utils.activationFunctions import sigmoid, sigmoidPrime

examples = [
    TrainingExample([[1],[0]],[[1]]),
    TrainingExample([[1],[1]],[[0]]),
    TrainingExample([[0],[0]],[[0]]),
    TrainingExample([[0],[1]],[[1]]),
]



if __name__ == "__main__":

    N = Network(.5,2,activation=sigmoid,activationDerivative=sigmoidPrime,costFunction=squaredError,costDerivative=squaredErrorPrime)
    N.addLayer(2)
    N.addLayer(1)

    print("before training predicitons: ")
    for i in examples:
        print("input: {} prediction: {}".format(i.input.flatten(),N.forwardPropagate(i.input)))


    N.fit(examples,100000)

    print("\nafter training predicitons: ")
    for i in examples:
        print("input: {} prediction: {}".format(i.input.flatten(),N.forwardPropagate(i.input)))

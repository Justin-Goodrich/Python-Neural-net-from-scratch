from Layer import Layer


class Network:

    def __init__(self, outputLayer,inputNodes=0):
        self.firstLayer = outputLayer
        self.inputNodes = inputNodes
        self.outputLayer = outputLayer
        self.loss = None

    def intializeWeights(self):
        self.firstLayer.randomizeWeights(self.inputNodes)
        self.firstLayer.intializeBias()

    def forwardPropagate(self,input):
        return self.firstLayer.computeOutput(input)
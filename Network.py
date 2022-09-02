from Layer import Layer


class Network:
    #each network shouldbe required to have one input and output layer
    def __init__(self, inputLayer,outputLayer):
        inputLayer.next = outputLayer
        outputLayer.prev = inputLayer

        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.loss = None

    
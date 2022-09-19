from Layer import Layer


class Network:

    def __init__(self, outputLayer, cost, costDerivative, learningRate, inputNodes=0):
        self.firstLayer = outputLayer
        self.inputNodes = inputNodes
        self.outputLayer = outputLayer
        self.loss = None
        self.cost = cost
        self.costDerivative = costDerivative
        self.learningRate = learningRate

    def addLayer(self,newLayer):
        newLayer.next = self.firstLayer
        self.firstLayer.prev = newLayer
        self.firstLayer = newLayer

    def intializeWeights(self):
        self.firstLayer.randomizeWeights(self.inputNodes)
        # self.firstLayer.intializeBias()

    def forwardPropagate(self,input):
        return self.firstLayer.computeOutput(input)

    def train(self, examples):
        for i in examples:
            actual = self.forwardPropagate(i.input)
            layer = self.outputLayer
        
            while layer is not None:
                if layer.next is None:
                    layer.computeError(self.costDerivative(actual,i.expected)) 
                else: 
                    layer.computeError()
                
                if layer.prev is None:
                    layer.adjustWeights(self.learningRate, i.input)
                else:
                    layer.adjustWeights(self.learningRate)
                layer = layer.prev

                

                    

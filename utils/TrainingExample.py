import numpy as np

class TrainingExample:
   def  __init__(self, input, expected):
    self.input = np.matrix(input)
    self.expected = np.matrix(expected)


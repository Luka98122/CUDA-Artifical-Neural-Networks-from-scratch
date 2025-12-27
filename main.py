import sys
import numpy as np
import matplotlib

np.random.seed(0)

# Constants   
X = [[1, 2, 3, 2.5], 
     [2.0,5.0, -1.0, 2.0],
     [-1.5,2.7,3.3,-0.8]]

class Neuron_Layer():
    def __init__(self, n_inputs,n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        pass

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases
        pass

layer1 = Neuron_Layer(4,5) 
layer2 = Neuron_Layer(5,2)

# Input of layer2 has to match the output of layer1

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)

# TODO: Add activation functions.
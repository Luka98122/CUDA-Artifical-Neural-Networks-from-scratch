import sys
import numpy as np
import matplotlib.pyplot as plt
import nnfs
nnfs.init()

np.random.seed(0)


def create_data(points, classes): # Creates a dataset of spiral shaped clusters
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')

    for class_number in range(classes):
        ix = range(points * class_number,
                   points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4,
                        (class_number + 1) * 4,
                        points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5),
                      r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y


class Neuron_Layer():
    def __init__(self, n_inputs,n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        pass

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases
        pass
    
class Activation_ReLU():
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

"""
# Constants   
X = [[1, 2, 3, 2.5], 
     [2.0,5.0, -1.0, 2.0],
     [-1.5,2.7,3.3,-0.8]]
layer1 = Neuron_Layer(4,5) 
layer2 = Neuron_Layer(5,2)

# Input shape of layer2 has to match the output shape of layer1

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
"""

X,Y = create_data(100,3)

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_ReLU()

layer1 = Neuron_Layer(2,16)
layer1.forward(X)
activation1.forward(layer1.output)

layer2 = Neuron_Layer(16,16)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3 = Neuron_Layer(16,3)
layer3.forward(activation2.output)
activation3.forward(layer3.output)

print(activation3.output)
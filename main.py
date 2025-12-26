import sys
import numpy as np
import matplotlib

class Neuron():
    def __init__(self, inputs,weights,bias):
        output = 0
        for i in range(len(inputs)):
            output+=inputs[i]*weights[i]
        
        output += bias
        self.output = output

# Constants   
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5    
 
""" 
All neurons in a layer have the same inputs (the output of the previous layer),
but output different values because of their unique weights and biases.
"""

neuron1 = Neuron(inputs,weights1,bias1)
neuron2 = Neuron(inputs,weights2,bias2)
neuron3 = Neuron(inputs,weights3,bias3)

print([neuron1.output,neuron2.output,neuron3.output])
import sys
import numpy as np
import matplotlib.pyplot as plt
import math


np.random.seed(0)
# Used for sample data, loosely coppied from
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
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights)+self.biases
        pass
    
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

    
class Activation_ReLU():
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        self.inputs = inputs
        
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax():
    def forward(self,inputs):
        exps = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        normalised = exps / np.sum(exps,axis=1,keepdims=True)
        self.output = normalised
    
    def backward(self,dvalues): 
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
                              
            # Calculate sample-wise gradient.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Loss can be implemented as -log(softmax_output[x]), where x is the target index of the
# correct classification.     
class Loss:
    
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        
        if len(y_true.shape)==1: # Labels are class indicies
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2: # One hot encoding
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        loss = -np.log(correct_confidences)
        return loss
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape)==1:
            # If it is sparse, convert to one-hot
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues # Calculate gradient
        self.dinputs = self.dinputs/samples # Normalize gradient
        
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    
    def backward(self,dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape)==2: # If one hot, turn it into sparse (discrete)
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs / samples    

X,Y = create_data(100,3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

layer1 = Neuron_Layer(2,3)
activation1 = Activation_ReLU()

layer2 = Neuron_Layer(3,3) # Output layer

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
loss = loss_activation.forward(layer2.output, Y)

predictions = np.argmax(loss_activation.output,axis=1)

loss_activation.backward(loss_activation.output, Y)
layer2.backward(loss_activation.dinputs)
activation1.backward(layer2.dinputs)
layer1.backward(activation1.dinputs)

print(layer1.dweights)
print(layer1.dbiases)
print(layer2.dweights)
print(layer2.dbiases)


if len(Y.shape)==2:
    Y = np.argmax(Y,axis=1)
accuracy = np.mean(predictions==Y)
print(accuracy)

"""
Two ways of doing the same thing
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(activation2.output, Y)
dvalues1 = softmax_loss.dinputs


# --- Separate Softmax + Loss (full Jacobian) ---
activation = Activation_Softmax()
activation.output = activation2.output

loss = Loss_CategoricalCrossentropy()
loss.backward(activation2.output, Y)
activation.backward(loss.dinputs)

dvalues2 = activation.dinputs


# --- Compare ---
print('Combined (dL/dz):')
print(dvalues1[:5])

print('Separate (dL/dz):')
print(dvalues2[:5])

print('Max abs diff:',
      np.max(np.abs(dvalues1 - dvalues2)))
"""

# TODO: Reformat to jupyter notebook.
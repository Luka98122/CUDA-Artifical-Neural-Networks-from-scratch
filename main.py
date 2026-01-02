import sys
import numpy as np
import matplotlib.pyplot as plt
import nnfs
import math
nnfs.init()

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
        self.output = np.dot(inputs,self.weights)+self.biases
        pass
    
    
class Activation_ReLU():
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax():
    def forward(self,inputs):
        exps = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        normalised = exps / np.sum(exps,axis=1,keepdims=True)
        self.output = normalised

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
        
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        loss = -np.log(correct_confidences)
        return loss
    

X,Y = create_data(100,3)

layer1 = Neuron_Layer(2,3)
activation1 = Activation_ReLU()

layer2 = Neuron_Layer(3,3) # Output layer
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

predictions = np.argmax(activation2.output,axis=1)

if len(Y.shape)==2:
    Y = np.argmax(Y,axis=1)
accuracy = np.mean(predictions==Y)
print(accuracy)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output,Y)
print("Loss:", loss)

import sys
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(42)

def createDataset(points,classes):
    X = np.random.uniform(-1, 1, size=(points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')

    for i, (x, y_val) in enumerate(X):
        if x >= 0 and y_val >= 0:
            y[i] = 0  # Quad 1
        elif x < 0 and y_val >= 0:
            y[i] = 1  # Quad 2
        elif x < 0 and y_val < 0:
            y[i] = 2  # Quad 3
        else:
            y[i] = 3  # Quad 4

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

def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)


X, y = createDataset(points=1000, classes=4)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=30)

# Axes lines
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Quadrant Classification Dataset")
plt.grid(True)
plt.show()

w1 = 0.1*np.random.randn(2, 8)
w2 = 0.1*np.random.randn(8, 4)
bw1 = w1
bw2 = w2


bloss =  9999999


layer1 = Neuron_Layer(2,8)
layer1.weights = w1
activation1 = Activation_ReLU()
layer2 = Neuron_Layer(8,4)
layer2.weights = w2
activation2 = Activation_Softmax()

bb1 = layer1.biases
bb2 = layer2.biases

for i in range(50001):
    
    nw1 = bw1 + 0.0005*np.random.randn(2, 8)
    nw2 = bw2 + 0.0005*np.random.randn(8, 4)
    
    nb1 = bb1 + 0.0005*np.random.randn(1, 8)
    nb2 = bb2 + 0.0005*np.random.randn(1, 4)
    
    layer1.biases = nb1
    layer2.biases =  nb2
    
    layer1.weights = nw1
    layer2.weights = nw2
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss1 = Loss_CategoricalCrossentropy()
    los = loss1.calculate(activation2.output, y)
    
    acc = accuracy(activation2.output, y)

    if i % 5000 == 0:
        print(f"Iter {i:6d} | loss: {los:.4f} | acc: {acc:.4f}")
    
    if los<bloss:
        bw1 = nw1
        bw2 = nw2
        bb1 = nb1
        bb2 = nb2
        bloss = los
    else:
        layer1.weights = bw1
        layer2.weights = bw2
        layer1.biases = bb1
        layer2.biases = bb2


import sys
import numpy as np
import matplotlib.pyplot as plt
import math

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw1 = x[0]*w[0]
xw2 = x[1]*w[1]
xw3 = x[2]*w[2]

res = xw1+xw2+xw3+b

z = max(res,0) #ReLU activation
print(f"Original output: {z}")

dValue = 1.0 # Derivative from the next layer

# ReLU(Sum(Products()))

# ReLU derivative
drelu_dz = dValue * (1. if z > 0 else 0.)
print(drelu_dz)

"""
The partial derivative of f with respect to x equals y.
The partial derivative of f with respect to y equals x.
"""


# Multiply by previous layer because of chain rule.
# Derivatives of sum layer
dsum_dxw0 = 1 
dsum_dxw1 = 1 
dsum_dxw2 = 1 
dsum_db = 1 # Bias

drelu_dxw0 = drelu_dz*dsum_dxw0
drelu_dxw1 = drelu_dz*dsum_dxw1
drelu_dxw2 = drelu_dz*dsum_dxw2
drelu_db = drelu_dz *dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication:
# with respect to the x:
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
# with respect to the w:
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0 # => drelu_dx0 = drelu_dxw0 * w[0]
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# to simplify:
# drelu_dx0 = drelu_dxw0 * w[0]
# drelu_dxw0 = drelu_dz*dsum_dxw0 => drelu_dx0 = drelu_dz * dsum_dxw0 * w[0]
# dsum_dxw0 always = 1, so drelu_dx0 = drelu_dz * 1 * w[0]
# drelu_dx0 = drelu_dz * w[0]
# drelu_dx0 = dValue * (1.0 if z> 0 else 0.0) * w[0]

dx = [drelu_dx0, drelu_dx1, drelu_dx2] # Gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # Gradients on weights
db = drelu_db # Bias gradient only has 1 bias, because we are only doing 1 neuron.

# We can now intelligently decrease the output of the neuron, if we wanted to.
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
z = xw0 + xw1 + xw2 + b
# ReLU activation function
y = max(z, 0)
print(f"New output decreased: {y}")
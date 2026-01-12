#simple feedforward Backpropagation
import math

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# Input, weight, bias
x = 1.0
w = 0.5
b = 0.0

# Target output
target = 1.0

# Learning rate
eta = 0.1

print("Initial weight:", w)

# --------- Forward Pass ----------
net = x * w + b
y = sigmoid(net)
print("Output before training:", y)

# --------- Error ----------
error = target - y

# --------- Backpropagation ----------
delta = error * sigmoid_derivative(y)

# --------- Weight Update ----------
w = w + eta * delta * x

# --------- Forward Pass again ----------
net = x * w + b
y_new = sigmoid(net)

print("Updated weight:", w)
print("Output after training:", y_new)

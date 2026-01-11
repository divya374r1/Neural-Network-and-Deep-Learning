#Backpropagation Implementation for Multi-Layer Perceptron (MLP) in Python
import math

# ---------- Sigmoid and derivative ----------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    # y is sigmoid(x)
    return y * (1 - y)

# ---------- Inputs ----------
x1, x2, x3 = 1, 0, 1
target = 1
eta = 0.1

# ---------- Initial Weights ----------
# Input -> Hidden
w14, w24, w34 = 0.2, 0.4, -0.5
w15, w25, w35 = -0.3, 0.1, 0.2

# Biases (Hidden)
theta4, theta5 = 0.4, 0.2

# Hidden -> Output
w46, w56 = -0.3, -0.2
theta6 = 0.1

# ==================================================
# FORWARD PASS
# ==================================================

# Hidden layer
net_h4 = x1*w14 + x2*w24 + x3*w34 + theta4
h4 = sigmoid(net_h4)

net_h5 = x1*w15 + x2*w25 + x3*w35 + theta5
h5 = sigmoid(net_h5)

# Output layer
net_o6 = h4*w46 + h5*w56 + theta6
y6 = sigmoid(net_o6)

print("Forward Pass Output:", y6)

# ==================================================
# BACKPROPAGATION
# ==================================================

# Output error term
delta6 = (target - y6) * sigmoid_derivative(y6)

# Hidden layer error terms
delta4 = sigmoid_derivative(h4) * (delta6 * w46)
delta5 = sigmoid_derivative(h5) * (delta6 * w56)

# ==================================================
# WEIGHT UPDATES
# ==================================================

# Hidden -> Output
w46 += eta * delta6 * h4
w56 += eta * delta6 * h5

# Input -> Hidden (H4)
w14 += eta * delta4 * x1
w24 += eta * delta4 * x2
w34 += eta * delta4 * x3

# Input -> Hidden (H5)
w15 += eta * delta5 * x1
w25 += eta * delta5 * x2
w35 += eta * delta5 * x3

# ==================================================
# FORWARD PASS AGAIN (After update)
# ==================================================

net_h4 = x1*w14 + x2*w24 + x3*w34 + theta4
h4 = sigmoid(net_h4)

net_h5 = x1*w15 + x2*w25 + x3*w35 + theta5
h5 = sigmoid(net_h5)

net_o6 = h4*w46 + h5*w56 + theta6
y6_new = sigmoid(net_o6)

print("Updated Output:", y6_new)

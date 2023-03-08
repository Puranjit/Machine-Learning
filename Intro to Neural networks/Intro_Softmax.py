import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Class for defining hidden layers in the neural network
class Layer_Dense:
	def __init__(self, n_input, n_neuron):
		self.weigths = 0.1 * np.random.randn(n_input, n_neuron)
		self.biases = np.zeros((1, n_neuron))

	def forward(self, input):
		self.output = np.dot(input, self.weigths) + self.biases

# ReLU is a popular activation function that is used for calculating of final output from a neuron in the hidden layers
class Activation_ReLU:
	def forward(self, input):
		self.output = np.maximum(0, input)

# Activation function that is defined for the final output layer in neural network
class Activation_Softmax:
	def forward(self, input):
		exp_values = np.exp(input - np.max(input, axis = 1, keepdims = True))
		probabilities = exp_values/np.sum(exp_values, axis = 1, keepdims = True)
		self.output = probabilities

X, y = spiral_data(samples = 100, classes = 3)
# print(X)
# print(np.shape(X))
# print(y)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
print(dense1.output[:5])
print(np.shape(dense1.output))

dense2.forward(activation1.output)
print(activation1.output[:5])
print(np.shape(activation1.output))

activation2.forward(dense2.output)
print(dense2.output[:5])
print(np.shape(dense2.output))

print(activation2.output[:5])
print(np.shape(activation2.output))
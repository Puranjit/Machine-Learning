import numpy as np 
np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X,y = spiral_data(100,3)
# X = np.array(X)
# y = np.array(y)
# print(X.shape)
# print(y.shape)

class Layer_Dense:
	def __init__(self, n_input, n_neuron):
		# Here, n_input stands for number of features per each sample in our input
		# and n_neuron represent the number of neurons present in the next layer
		self.weights = 0.1 * np.random.randn(n_input, n_neuron) 
		self.biases = np.zeros((1,n_neuron))

	def forward(self, input):
		self.output = np.dot(input, self.weights) + self.biases



# ReLU is a popular activation function that is used for calculating of final output from a neuron in the hidden layers
# This example explains what will be the output to a specific input to the ReLU activation function
# All the inputs below 0 are rejected
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
	# if i > 0:
	# 	output.append(i)
	# else:
	# 	output.append(0)
	output.append(max(0,i))
print('Output for the given input using ReLU activation function: ', output, '\n')

class Activation_ReLU:
	def forward(self, input):
		return np.maximum(0,input)



layer1 = Layer_Dense(2,5)
layer1.forward(X)
Activation1 = Activation_ReLU()
# print(layer1.output)
out = Activation1.forward(layer1.output)
print(out)
print(out.shape)
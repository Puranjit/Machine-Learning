# Basic introduction for a neural network

import numpy as np

# This is a vanilla neural network which has a single input that is connected to 3 different neurons in the hidden layer
inputs = [1, 2, 3, 2.5]

# Each component of Input has a different weight associated with the neuron it is attached to... 
weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

# Each neuron in the hidden layer has a bias associated with it
biases = [2, 3, 0.5]

layer_output = []

# Basic operation performed to calculate the output from the neurons in the hidden layer
for neuron_weights, neuron_bias in zip(weights, biases):
	neuron_output = 0 # output of a given neuron
	for n_input, weight in zip(inputs, neuron_weights):
		# print(n_input)
		neuron_output += n_input*weight
	neuron_output += neuron_bias
	layer_output.append(neuron_output)
print(layer_output, '\n')

# one line dot product code for the above loop
output = np.dot(weights, inputs) + biases
print(output, '\n')

output = []
for i in range(len(weights)):
	out1 = (inputs[0]*weights[i][0] + inputs[1]*weights[i][1] + inputs[2]*weights[i][2] + inputs[3]*weights[i][3]) + biases[i]
	output.append(out1)
print(output, '\n')

# Example to view how zip operation works in python
numbers = [[1,2,3],[4,5,6],[7,8,9]]
letters = ['a', 'b', 'c']
zipped = zip(numbers, letters)
print(list(zipped))
for i,j in zip(numbers,letters):
	print(i,'\t', j)
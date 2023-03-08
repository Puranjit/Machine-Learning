import numpy as np

# Neural network with 3 different inputs that are connected to 2 hidden layers with 3 neurons in each hidden layer
inputs = [[1, 2, 3, 2.5],
[2.0, 5.0, -1.0, 2.0],
[-1.5, 2.7, 3.3, -0.8]]

# Hidden layer 1
weights1 = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

# Each neuron in the hidden layer has a bias associated with it
biases1 = [2, 3, 0.5]

# Hidden layer 2 
weights2 = [[0.1, -0.14, 0.5],
[-0.5, 0.12, -0.33],
[-0.44, 0.73, -0.13]]

# Each neuron in the hidden layer has a bias associated with it
biases2 = [-1, 2, -0.5]

layer_output = []

output1 = np.dot(inputs, np.array(weights1).T) + biases1
# print(output1)

output2 = np.dot(output1, np.array(weights2).T) + biases2
print(output2)

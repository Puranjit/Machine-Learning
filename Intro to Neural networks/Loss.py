# This function gives an introduction on how to calculate the cost function/loss in the neural network
# Cost function/loss - gives us information on how our model is performing on a giving input, deviation in output from actual input

import numpy as np
import math

# Example 1
softmax_output = [0.7, 0.1, 0.2]
target_class = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_class[0]+
	math.log(softmax_output[1])*target_class[1]
	+math.log(softmax_output[2])*target_class[2])
print(loss)
print(-math.log(softmax_output[1]))


# Example 2
softmax_output = np.array([[0.7, 0.1, 0.2],
				   [0.1, 0.5, 0.4],
				   [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

print(softmax_output[[0, 1, 2], class_targets])
print(softmax_output[[1],[1]])
# or
print(softmax_output[range(len(softmax_output)), class_targets])

# Final LOSS 
print(-np.log(softmax_output[range(len(softmax_output)), class_targets]))
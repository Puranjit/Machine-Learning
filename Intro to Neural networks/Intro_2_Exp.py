import math
import numpy as np

'''
Log calculation 
e (2.717) ** x = b
'''

b = 5.2
print(np.log(10))

E = math.e
print(E**2.3025)
print(np.log(0.7))

layer_outputs = [4.8, 1.21, 2.385]

# Technique 1
E = math.e
exp_values = []

for i in layer_outputs:
	exp_values.append(E**i)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for i in exp_values:
	norm_values.append(i/norm_base)
print(norm_values)
print('\n')

# Technique 2
exp_values1 = np.exp(layer_outputs)
print(exp_values1)

norm_values1 = exp_values1/np.sum(exp_values1)
print(norm_values1)


layer_outputs1 = [[4.8, 1.21, 2.385],
				  [8.9, -1.81, 0.2],
				  [1.41, 1.051, 0.026]]

exp_values1 = np.exp(layer_outputs1)
print(exp_values1)

# np.sum and use of axis = 0 (Coloumns), axis = 1 (Rows) and axis = None = Sum of all matrix
print(np.sum(exp_values1, axis = 1)) # keepdims = True - shows ooutput in a matrix

norm_values1 = exp_values1/np.sum(exp_values1, axis = 1, keepdims = True)
print(norm_values1)

# print(np.sum(exp_values1[1]))
# # print('\n')
# print(np.array(exp_values1)[0])

# norm_values1 = layer_outputs1/np.sum(layer_outputs1)
# print(norm_values1)
# print(np.shape(exp_values1))
# print(np.sum(exp_values1[0]))
# norm_values1 = []
# for j in range(np.shape(exp_values1)[0]):
# 	norm_values1.append(exp_values1[j]/np.sum(exp_values[j]))
# print(norm_values1)
print(np.exp(1000))
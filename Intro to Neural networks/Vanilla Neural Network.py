# Vanilla neural network - This is a very basic code that which explains the basic functioning/concepts of neural networks
# In this neural network we were able to reduce the loss from a factor of 1e+4 to 1e-3 factor
# Increasing number of epochs could lead to overfitting and increase the value of loss calculated

# importing the required libraries
import numpy as np
from numpy.random import randn

# Initializing weights and data
# N - Input (here we can imagine N as total number of input images inserted into a model)
# Din - (Input.size) of each image inserted into neural network
# H - Total number of neurons in the hidden layer
# Dout - Final Outputs (categories in which each image would be classified)  
N, Din, H, Dout = 64, 1000, 100, 10

# We randomnly initialize input pixels for all images in x and vector representation of each class predicted by our neural network  
x, y = randn(N, Din), randn(N, Dout)
# dim(x) = (64,1000); dim(y) = (64,10)

# Randomly initialized weights in between input layer and hidden layer in w1 and hidden layer and output layer in w2
w1, w2 = randn(Din, H), randn(H, Dout)
# dim(w1) = (1000,100); dim(w2) = (100,10) 

# Total epochs for which we will run our model to update weights in each iteration
# Epoch - iteration in which neural network completes one complete cycle of forward and backward propagation

for epoch in range(7500):
# Forward propagation
    # Using sigmoid activation function as a non-linear function to calculate the output from neurons in hidden layers
    # dim(h) = (64,100)
    h = 1.0/(1.0+np.exp(-x.dot(w1)))
    
    # Predicting classes based on inputs and activation function used in our network
    # dim(y_pred) = (64,10)
    y_pred = h.dot(w2)

# Backward propagation
# Compute loss (sigmoid activation, L2 (Euclidean) loss)
    # Calculating the total loss after each iteration [sum(deviation in predicted output from true label)]
    loss = round(np.square(y_pred-y).sum(),5)
    if epoch == 0:
        print(loss)

# Computing gradients
    # Multiplying the loss vector for each input by a scalar so that it can be used more effectively to minimize loss 
    # (lower magnitude of predicetd loss would lead us to run more number of iterations)
    # dim(dy_pred) = (64,10)
    dy_pred = 2.0*(y_pred-y)

    # Updating the w2 weights in neural network
    # dim(h.T) = (100,64); dim(dy_pred) = (64,10) : dim(dw2) = (100,10)
    dw2 = h.T.dot(dy_pred)

    # Updating the weights w1 in neural network
    dh = dy_pred.dot(w2.T)
    dw1 = x.T.dot(dh*h*(1-h))

# Stochastic Gradient (SGD) Step 
    # Updating the initialised weights (w1 & w2) for next iteration (using a learning rate paarmeter which helps in overcoming the problem of vanishing gradient problem)
    # dim(w1) & dim(w2) will remain same throughout execution
    w1 -= 1e-4*dw1
    w2 -= 1e-4*dw2

print(loss)
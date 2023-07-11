import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the feed-forward pass
def feed_forward(inputs, weights):
    hidden_layer = np.dot(inputs, weights[0])
    hidden_layer_activation = sigmoid(hidden_layer)
    
    output_layer = np.dot(hidden_layer_activation, weights[1])
    output_layer_activation = sigmoid(output_layer)
    
    return hidden_layer_activation, output_layer_activation

# Define the backward pass (backpropagation)
def backpropagation(inputs, targets, weights, learning_rate):
    hidden_layer_activation, output_layer_activation = feed_forward(inputs, weights)
    
    # Calculate the error at the output layer
    output_error = targets - output_layer_activation
    output_delta = output_error * sigmoid_derivative(output_layer_activation)
    
    # Calculate the error at the hidden layer
    hidden_error = output_delta.dot(weights[1].T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_activation)
    
    # Update the weights
    weights[1] += hidden_layer_activation.T.dot(output_delta) * learning_rate
    weights[0] += inputs.T.dot(hidden_delta) * learning_rate
    
    return weights

# Define the main training function
def train(inputs, targets, num_epochs, learning_rate):
    num_inputs = inputs.shape[1]
    num_hidden_units = 4
    num_output_units = 1
    
    # Initialize the weights with random values
    weights = [
        np.random.uniform(size=(num_inputs, num_hidden_units)),
        np.random.uniform(size=(num_hidden_units, num_output_units))
    ]
    
    # Keep track of the loss values for visualization
    loss_values = []
    
    for epoch in range(num_epochs):
        # Perform the forward and backward pass
        weights = backpropagation(inputs, targets, weights, learning_rate)
        
        # Calculate the loss (mean squared error)
        _, output = feed_forward(inputs, weights)
        loss = np.mean((targets - output) ** 2)
        loss_values.append(loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")
    
    # Plot the loss values
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

# Generate sample data for training
np.random.seed(0)
num_samples = 100
inputs = np.random.uniform(size=(num_samples, 2))
targets = np.sum(inputs, axis=1, keepdims=True) + np.random.normal(scale=0.1, size=(num_samples, 1))

# Train the MLP
num_epochs = 1000
learning_rate = 0.1
train(inputs, targets, num_epochs, learning_rate)

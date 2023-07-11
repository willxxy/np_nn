import numpy as np


"""
NOTE !!!
When we say "calculating the gradients," we are referring to the process of 
computing the derivatives of a function or model with respect to its parameters or inputs. 

Gradients indicate the direction and magnitude of the steepest ascent or descent of a function. 
By calculating gradients, we can determine how small changes in the parameters or inputs of a model affect the output or loss function. 
This information is then used to update the model's parameters to minimize the loss function and improve the model's performance.
"""


# forward pass
def forward(x, layer1, layer2):
    """
    This function takes an input data sample and the weight matrices of the layers in a MLP, 
    and computes the forward pass through the model. 
    It applies matrix multiplication and activation functions to the input data and 
    returns the intermediate outputs of each layer.

    Parameters:
    - x (np.ndarray): The input data sample of shape (num_features,).
    - layer1 (np.ndarray): The weight matrix of the first layer of the MLP.
    - layer2 (np.ndarray): The weight matrix of the second layer of the MLP.

    Returns:
    - x_layer1 (np.ndarray): The output of the first layer before applying the activation function.
    - x_relu (np.ndarray): The output of the first layer after applying the ReLU activation function.
    - x_layer2 (np.ndarray): The output of the second layer before applying the activation function.
    """
    
    # performs a matrix multiplication between the input x and the weight matrix layer1 
    # to compute the linear transformation of the input data in the first layer of the MLP.
    x_layer1 = x.dot(layer1)
    
    # applies the Rectified Linear Unit (ReLU) activation function element-wise to the outputs of the first layer's linear transformation. 
    # ReLU sets negative values to zero while preserving positive values
    x_relu = np.maximum(x_layer1, 0)
    
    # performs a matrix multiplication between the ReLU activation outputs x_relu and the weight matrix layer2
    # to compute the linear transformation of the ReLU outputs in the second layer of the MLP
    x_layer2 = x_relu.dot(layer2)
      
    return x_layer1, x_relu, x_layer2


# backward pass
def backward(out, x, y, layer2, x_logsumexp, x_relu):
    """
    Perform the backward pass of a MLP.

    This function takes the output of the forward pass, input data, true labels, and intermediate outputs of a MLP, 
    and computes the gradients of the model's parameters using backpropagation. 
    It calculates the gradients of the loss with respect to the weights of each layer.

    Parameters:
    - out (np.ndarray): The output of the forward pass of the model.
    - x (np.ndarray): The input data of shape (num_samples, num_features).
    - y (np.ndarray): The true labels of the input data, encoded as one-hot vectors of shape (num_samples, num_classes).
    - layer2 (np.ndarray): The weight matrix of the second layer of the MLP.
    - x_logsumexp (np.ndarray): The intermediate output of the log-sum-exp operation in the forward pass.
    - x_relu (np.ndarray): The intermediate output of the ReLU activation function in the forward pass.

    Returns:
    - d_layer1 (np.ndarray): The gradients of the loss with respect to the weights of the first layer.
    - d_layer2 (np.ndarray): The gradients of the loss with respect to the weights of the second layer.
    """
    d_out = -out / len(y)

    # calculates the gradient of the softmax cross-entropy loss function 
    # with respect to the input of the softmax function in the forward pass
    dx_lsm = d_out - np.exp(x_logsumexp)*d_out.sum(axis=1).reshape((-1, 1))

    # calculates the gradients of the loss with respect to the weights of the second layer.
    d_layer2 = x_relu.T.dot(dx_lsm)
    
    # calculates the gradients of the loss with respect to the ReLU activation outputs of layer1
    dx_relu = dx_lsm.dot(layer2.T)

    # calculates the gradients of the loss with respect to the weights of the first layer, 
    # considering the ReLU activation function.
    dx_layer1 = (x_relu > 0).astype(np.float32) * dx_relu

    # calculates the gradients of the loss with respect to the weights of the first layer
    # using the chain rule of derivatives.
    d_layer1 = x.T.dot(dx_layer1)
    
    return d_layer1, d_layer2

# evaluator
def numpy_eval(X_test, Y_test, layer1, layer2):
    """
    This function takes the test data and the weights of the trained layers of a MLP and computes the accuracy of the MLP's predictions on the test data. 
    The accuracy is calculated as the percentage of correct predictions.

    Parameters:
    - X_test (np.ndarray): The input test data of shape (num_samples, num_features).
    - Y_test (np.ndarray): The true labels of the test data, encoded as one-hot vectors of shape (num_samples, num_classes).
    - layer1 (np.ndarray): The weight matrix of the first layer of the MLP.
    - layer2 (np.ndarray): The weight matrix of the second layer of the MLP.

    Returns:
    - accuracy (float): The accuracy of the model's predictions on the test data, ranging from 0 to 1.
    """
    _, _, Y_test_preds_out = forward(X_test.reshape((-1, 28*28)), layer1, layer2)
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (Y_test == Y_test_preds).mean()


def layer_init(input_dim, output_dim):
    """
    This function generates and returns a weight matrix for a layer in a MLP. 
    The weight matrix is initialized using a uniform distribution between -1 and 1, 
    and then scaled by the square root of the product of the input and output dimensions. 
    The resulting matrix is of type np.float32.

    Parameters:
    - input_dim (int): The input dimension of the layer.
    - output_dim (int): The output dimension of the layer.

    Returns:
    - weight_matrix (np.ndarray): A weight matrix of shape (input_dim, output_dim) for the layer.
    """
    
    weight_matrix = np.random.uniform(-1., 1., size=(input_dim,output_dim))/np.sqrt(input_dim*output_dim)
    return weight_matrix.astype(np.float32)

def logsumexp(x):
    
    """
    This function calculates the log-sum-exp of an input array x, which is a numerically stable way to 
    compute the sum of exponentiated values. The function first finds the maximum value along each row of x and 
    subtracts it from x to avoid numerical overflow. Then, it exponentiates the modified x, computes the sum along each row, 
    and takes the logarithm of the sums.

    Parameters:
    - x (np.ndarray): The input array of shape (num_samples, num_values).

    Returns:
    - result (np.ndarray): The log-sum-exp of x, a 1D array of shape (num_samples,), containing the calculated values.
    """
    
    c = x.max(axis=1)
    return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))

def MLP(x, y, layer1, layer2):
    # training
    out = np.zeros((len(y),10), np.float32)
    out[range(out.shape[0]),y] = 1

    # forward pass
    x_layer1, x_relu, x_layer2 = forward(x, layer1, layer2)
    
    # computes the log-sum-exp of the values in x_layer2 while avoiding numerical instability. 
    # It subtracts the log-sum-exp values from x_layer2 to shift the range of the values to a numerically more stable region
    x_logsumexp = x_layer2 - logsumexp(x_layer2).reshape((-1, 1))
    
    # This step computes the element-wise product between the output values out and x_logsumexp. 
    # It then takes the mean along each row to obtain the loss values for each sample
    x_loss = (-out * x_logsumexp).mean(axis=1)
    
    # backward pass
    d_layer1, d_layer2 = backward(out, x, y, layer2, x_logsumexp, x_relu)
    
    return x_loss, x_layer2, d_layer1, d_layer2

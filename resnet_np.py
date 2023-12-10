import numpy as np

# forward pass
def forward(x, layer1, layer2):
    x_layer1 = x.dot(layer1)
    
    # Adding residual connection
    x_res = x_layer1 + x  # Residual connection
    
    x_relu = np.maximum(x_res, 0)

    x_layer2 = x_relu.dot(layer2)
      
    return x_layer1, x_res, x_relu, x_layer2


# backward pass
def backward(out, x, y, layer2, x_logsumexp, x_res, x_relu):
    d_out = -out / len(y)

    dx_lsm = d_out - np.exp(x_logsumexp)*d_out.sum(axis=1).reshape((-1, 1))

    d_layer2 = x_relu.T.dot(dx_lsm)
    
    dx_relu = dx_lsm.dot(layer2.T)

    # Adjust gradient for residual connection
    dx_res = (x_res > 0).astype(np.float32) * dx_relu

    # Gradient for layer1 considering the residual connection
    dx_layer1 = dx_res + dx_relu  # Adding the gradient from the residual connection
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

def ResNet(x, y, layer1, layer2):
    # training
    out = np.zeros((len(y),10), np.float32)
    out[range(out.shape[0]),y] = 1

    # forward pass with residual
    x_layer1, x_res, x_relu, x_layer2 = forward(x, layer1, layer2)
    
    x_logsumexp = x_layer2 - logsumexp(x_layer2).reshape((-1, 1))

    x_loss = (-out * x_logsumexp).mean(axis=1)
    
    # backward pass with residual
    d_layer1, d_layer2 = backward(out, x, y, layer2, x_logsumexp, x_res, x_relu)
    
    return x_loss, x_layer2, d_layer1, d_layer2

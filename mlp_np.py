import numpy as np


# numpy forward pass

def forward(x, layer1, layer2):
    x = x.dot(layer1)
    x = np.maximum(x, 0)
    x = x.dot(layer2)  
    return x

def numpy_eval(X_test, Y_test, layer1, layer2):
    Y_test_preds_out = forward(X_test.reshape((-1, 28*28)), layer1, layer2)
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (Y_test == Y_test_preds).mean()

# numpy forward and backward pass

def logsumexp(x):
    c = x.max(axis=1)
    return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))

def forward_backward(x, y, layer1, layer2):
    # training
    out = np.zeros((len(y),10), np.float32)
    out[range(out.shape[0]),y] = 1

    # forward pass
    x_layer1 = x.dot(layer1)
    x_relu = np.maximum(x_layer1, 0)
    x_layer2 = x_relu.dot(layer2)
    
    x_logsumexp = x_layer2 - logsumexp(x_layer2).reshape((-1, 1))
    x_loss = (-out * x_logsumexp).mean(axis=1)
    
    # backward pass
    d_out = -out / len(y)

    dx_lsm = d_out - np.exp(x_logsumexp)*d_out.sum(axis=1).reshape((-1, 1))

    d_layer2 = x_relu.T.dot(dx_lsm)
    dx_relu = dx_lsm.dot(layer2.T)

    dx_layer1 = (x_relu > 0).astype(np.float32) * dx_relu

    d_layer1 = x.T.dot(dx_layer1)
    
    return x_loss, x_layer2, d_layer1, d_layer2


# numpy training
def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
    return ret.astype(np.float32)
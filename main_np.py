import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from main_torch import MLP
from fetcher import fetch


if __name__ == '__main__':

    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    
    model = MLP()

    model.load_state_dict(torch.load('pytorch_mnist.pth'))

    l1 = model.l1.weight.detach().numpy().T
    l2 = model.l2.weight.detach().numpy().T


    # numpy forward pass

    def forward(x):
        x = x.dot(l1)
        x = np.maximum(x, 0)
        x = x.dot(l2)  
        return x

    def numpy_eval():
        Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
        Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
        return (Y_test == Y_test_preds).mean()

    # numpy forward and backward pass

    def logsumexp(x):
        c = x.max(axis=1)
        return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))

    def forward_backward(x, y):
        # training
        out = np.zeros((len(y),10), np.float32)
        out[range(out.shape[0]),y] = 1

        # forward pass
        x_l1 = x.dot(l1)
        x_relu = np.maximum(x_l1, 0)
        x_l2 = x_relu.dot(l2)
        x_lsm = x_l2 - logsumexp(x_l2).reshape((-1, 1))
        x_loss = (-out * x_lsm).mean(axis=1)

        d_out = -out / len(y)

        dx_lsm = d_out - np.exp(x_lsm)*d_out.sum(axis=1).reshape((-1, 1))

        d_l2 = x_relu.T.dot(dx_lsm)
        dx_relu = dx_lsm.dot(l2.T)

        dx_l1 = (x_relu > 0).astype(np.float32) * dx_relu

        d_l1 = x.T.dot(dx_l1)
        
        return x_loss, x_l2, d_l1, d_l2

    samp = [0,1,2,3]
    x_loss, x_l2, d_l1, d_l2 = forward_backward(X_test[samp].reshape((-1, 28*28)), Y_test[samp])


    # numpy training
    def layer_init(m, h):
        ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
        return ret.astype(np.float32)

    # reinit
    np.random.seed(1337)
    l1 = layer_init(784, 128)
    l2 = layer_init(128, 10)

    lr = 0.001
    BS = 128
    losses, accuracies = [], []
    for i in tqdm(range(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = X_train[samp].reshape((-1, 28*28))
        Y = Y_train[samp]
        x_loss, x_l2, d_l1, d_l2 = forward_backward(X, Y)
        
        cat = np.argmax(x_l2, axis=1)
        accuracy = (cat == Y).mean()
        
        # SGD
        l1 = l1 - lr*d_l1
        l2 = l2 - lr*d_l2
        
        loss = x_loss.mean()
        losses.append(loss)
        accuracies.append(accuracy)

    print(f'Training Accuracy: {max(accuracies)}')
    print(f'Training Loss: {min(losses)}')

    Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
    print(f'Test Accuracy: {numpy_eval()}')

    # Plot losses
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(-0.1, 1.1)
    plt.title('Training Loss')
    plt.savefig('loss_plot_np.png')
    plt.close()

    # Plot accuracies
    plt.plot(accuracies)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(-0.1, 1.1)
    plt.title('Training Accuracy')
    plt.savefig('accuracy_plot_np.png')
    plt.close()
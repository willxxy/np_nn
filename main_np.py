import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from mlp_torch import MLP
from fetcher import fetch
from mlp_np import MLP, numpy_eval, layer_init


if __name__ == '__main__':

    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    
    # reinit
    np.random.seed(1337)
    layer1 = layer_init(784, 512)
    layer2 = layer_init(512, 10)

    learning_rate = 0.001
    batch_size = 256
    losses, accuracies = [], []
    
    for i in tqdm(range(4000), desc = 'Training Numpy MLP'):
        sample = np.random.randint(0, X_train.shape[0], size=(batch_size))
        X = X_train[sample].reshape((-1, 28*28))
        Y = Y_train[sample]
        x_loss, x_layer2, d_layer1, d_layer2 = MLP(X, Y, layer1, layer2)
        
        cat = np.argmax(x_layer2, axis=1)
        accuracy = (cat == Y).mean()
        
        # SGD
        layer1 = layer1 - learning_rate*d_layer1
        layer2 = layer2 - learning_rate*d_layer2
        
        loss = x_loss.mean()
        losses.append(loss)
        accuracies.append(accuracy)

    print(f'Training Accuracy: {max(accuracies)}')
    print(f'Training Loss: {min(losses)}')

    print(f'Test Accuracy: {numpy_eval(X_test, Y_test, layer1, layer2)}')

    # Plot losses
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(-0.1, 1.1)
    plt.title('Training Loss')
    plt.savefig('./learning_curves/loss_plot_np.png')
    plt.close()

    # Plot accuracies
    plt.plot(accuracies)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(-0.1, 1.1)
    plt.title('Training Accuracy')
    plt.savefig('./learning_curves/accuracy_plot_np.png')
    plt.close()
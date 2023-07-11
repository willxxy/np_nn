import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from mlp_torch import MLP
from fetcher import fetch


if __name__ == '__main__':
    
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    
    model = MLP()

    negative_log_likelihood_loss = nn.NLLLoss(reduction='none')
    
    # stochastic gradient descent (sgd)
    sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    batch_size = 256
    losses, accuracies = [], []
    
    for i in tqdm(range(4000), desc = 'Training Pytorch MLP:'):
        
        sample = np.random.randint(0, X_train.shape[0], size=(batch_size))
        
        X = torch.tensor(X_train[sample].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[sample]).long()
        
        model.zero_grad()
        
        out = model(X)
        cat = torch.argmax(out, dim=1)
        
        accuracy = (cat == Y).float().mean()
        
        loss = negative_log_likelihood_loss(out, Y)
        loss = loss.mean()
        loss.backward()
        
        sgd_optimizer.step()
        
        loss, accuracy = loss.item(), accuracy.item()
        losses.append(loss)
        accuracies.append(accuracy) 
    
    print(f'Training Accuracy: {max(accuracies)}')
    print(f'Training Loss: {min(losses)}')

    Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
    
    print(f'Test Accuracy: {(Y_test == Y_test_preds).mean()}')

    torch.save(model.state_dict(), 'pytorch_mnist.pth')
    
    # Plot losses
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(-0.1, 1.1)
    plt.title('Training Loss')
    plt.savefig('loss_plot_torch.png')
    plt.close()

    # Plot accuracies
    plt.plot(accuracies)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(-0.1, 1.1)
    plt.title('Training Accuracy')
    plt.savefig('accuracy_plot_torch.png')
    plt.close()
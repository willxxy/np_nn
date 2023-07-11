import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fetcher import fetch


class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(784, 128, bias=False)
    self.l2 = nn.Linear(128, 10, bias=False)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.sm(x)
    return x



if __name__ == '__main__':
    
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    
    
    model = MLP()


    loss_function = nn.NLLLoss(reduction='none')
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    BS = 128
    losses, accuracies = [], []
    
    for i in tqdm(range(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        model.zero_grad()
        out = model(X)
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        loss = loss.mean()
        loss.backward()
        optim.step()
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
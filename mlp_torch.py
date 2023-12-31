import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    
    self.layer1 = nn.Linear(784, 512, bias=False)
    self.layer2 = nn.Linear(512, 10, bias=False)
    self.softmax = nn.LogSoftmax(dim=1)
  
  # essentiall forward() function in mlp_np.py  
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = self.layer2(x)
    x = self.softmax(x)
    return x
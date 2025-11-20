import torch
from torch import nn
from copy import deepcopy

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim << 1)
        self.fc3 = nn.Linear(hidden_dim << 1, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def copy_of(self) -> "ANN":
        return deepcopy(self)

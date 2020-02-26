import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, x_dim, y_dim, hu1, hu2, hu3, device):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_dim, hu1).to(device)
        self.fc2 = nn.Linear(hu1, hu2).to(device)
        self.fc3 = nn.Linear(hu2, hu3).to(device)
        self.fc = nn.Linear(hu3, y_dim).to(device)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc(x)
        return x

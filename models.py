'''Transfer Learning Models'''
import torch
import torch.nn as nn


class MyModelA(nn.Module):
    def __init__(self, x_dim, device):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(x_dim, 16).to(device)
        self.fc2 = nn.Linear(16, 4).to(device)
        # self.fc = nn.Linear(hu3, y_dim).to(device)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = self.fc(x)
        return x


class MyModelB(nn.Module):
    def __init__(self, z_dim, out_dim, device):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(4+z_dim, out_dim).to(device)

    def forward(self, x):
        x = self.fc1(x)
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x, z):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2

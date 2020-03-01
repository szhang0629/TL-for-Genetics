'''Transfer Learning Models for 5 Genes Together'''
import torch
import torch.nn as nn


class MyModelA(nn.Module):
    def __init__(self, x_dim, device):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(x_dim, 32).to(device)
        self.fc2 = nn.Linear(32, 4).to(device)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class MyModelB(nn.Module):
    def __init__(self, z_dim, out_dim, device):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(20+z_dim, 5).to(device)
        self.fc2 = nn.Linear(5, out_dim).to(device)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x, z):
        x1 = [self.modelA[i](x[i]) for i in range(len(x))]
        x5 = torch.cat((x1[0], x1[1], x1[2], x1[3], x1[4], z), 1)
        x2 = self.modelB(x5)
        return x2

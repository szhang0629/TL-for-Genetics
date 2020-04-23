"""Transfer Learning Models"""
import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self, input_dim):
        super(Model1, self).__init__()
        self.fc = nn.Linear(input_dim, 8)
        self.fd = nn.Linear(8, 1)

    def forward(self, x, z):
        xz = torch.cat([x, z], 1)
        x1 = torch.sigmoid(self.fc(xz))
        return self.fd(x1)


class Model2(nn.Module):
    def __init__(self, input_dim):
        super(Model2, self).__init__()
        self.fc = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fd = nn.Linear(8, 1)

    def forward(self, x, z):
        xz = torch.cat([x, z], 1)
        x1 = torch.sigmoid(self.fc(xz))
        x2 = torch.sigmoid(self.fc2(x1))
        return self.fd(x2)

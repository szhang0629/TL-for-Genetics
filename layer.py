from abc import ABC

import torch
from torch import nn as nn


class LayerA(nn.Module, ABC):
    """
    A class to represent a layer of vanilla neural network
    """
    def __init__(self, in_dim, out_dim, z_dim=0):
        torch.manual_seed(629)
        super(LayerA, self).__init__()
        self.fc = nn.Linear(in_dim + z_dim, out_dim)
        # self.do = nn.Dropout()

    def forward(self, x, z=None):
        # x = self.do(x)
        if z is not None:
            x = torch.cat([x, z], 1)
        return self.fc(x)

    def pen(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                    # and not name.endswith(".bias"):
                penalty += (param ** 2).sum()
        return penalty


class LayerB(nn.Module, ABC):
    def __init__(self, bs0, bs1, bias=True):
        self.bs0, self.bs1 = bs0, bs1
        self.index = None
        torch.manual_seed(629)
        super(LayerB, self).__init__()
        self.fc0 = nn.Linear(self.bs0.n_basis, self.bs1.n_basis, bias=bias)

    def forward(self, x):
        x = self.fc0((x @ self.bs0.mat) / self.bs0.length)
        # x = self.fc0.bias
        if self.index is None:
            return x @ self.bs1.mat.t()
        if type(self.index) is not int:
            x = x[self.index]
        x = x * self.bs1.mat
        return x.sum(1, keepdim=True)

    def pen(self, lamb0=1, lamb1=1):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                if name.endswith(".weight"):
                    penalty += self.bs1.pen_2d(param, self.bs0, lamb0=lamb0,
                                               lamb1=lamb1)
                if name.endswith(".bias"):
                    penalty += self.bs1.pen_1d(param, lamb1=lamb1)
        return penalty


class LayerC(nn.Module, ABC):
    def __init__(self, bs0):
        self.bs0 = bs0
        torch.manual_seed(629)
        super(LayerC, self).__init__()
        self.fc0 = nn.Linear(self.bs0.n_basis+self.bs0.linear, 1)

    def forward(self, x):
        return self.fc0((x @ self.bs0.mat) / self.bs0.length)

    def pen(self, lamb0=1, lamb1=1):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                if name.endswith(".weight"):
                    penalty += self.bs0.pen_1d(param.reshape((-1,)), lamb1)
                if name.endswith(".bias"):
                    penalty += param[0] ** 2
        return penalty

    def plot(self):
        self.bs0.plot(self.fc0.weight)


class LayerD(nn.Module, ABC):
    def __init__(self, bs1):
        self.bs1 = bs1
        self.index = None
        torch.manual_seed(629)
        super(LayerD, self).__init__()
        self.fc0 = nn.Linear(1, self.bs1.n_basis)

    def forward(self, x):
        x = self.fc0.bias
        if self.index is None:
            return x @ self.bs1.mat.t()
        x = x * self.bs1.mat
        return x.sum(1, keepdim=True)

    def pen(self, lamb0=1, lamb1=1):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                if name.endswith(".bias"):
                    penalty += self.bs1.pen_1d(param, lamb1=lamb1)
        return penalty

    def plot(self):
        self.bs1.plot(self.fc0.bias)


# class LayerB(nn.Module, ABC):
#     def __init__(self, n_basis_0, n_basis_1):
#         self.bs0, self.bs1 = Basis(n_basis_0), Basis(n_basis_1)
#         self.index = None
#         torch.manual_seed(629)
#         super(LayerB, self).__init__()
#         self.fc0 = nn.Linear(self.bs0.n_basis, self.bs1.n_basis)

#     def forward(self, x):
#         x = self.fc0((x @ self.bs0.mat) / self.bs0.length)
#         if self.index is None:
#             return x @ self.bs1.mat.t()
#         if type(self.index) is not int:
#             x = x[self.index]
#         x = x * self.bs1.mat
#         return x.sum(1, keepdim=True)

#     def pen(self, lamb0=1, lamb1=1):
#         penalty = 0
#         for name, param in self.named_parameters():
#             if param.requires_grad and "fc" in name:
#                 if name.endswith(".weight"):
#                     penalty += self.bs1.pen_2d(param, self.bs0,
#                     lamb0=lamb0, lamb1 = lamb1)
#                 if name.endswith(".bias"):
#                     penalty += self.bs1.pen_1d(param, lamb1=lamb1)
#         return penalty

#     def plot(self):
#         self.bs1.plot(self.fc0.bias)

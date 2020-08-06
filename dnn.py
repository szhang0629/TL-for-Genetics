import copy
from abc import ABC

import torch
from torch import nn as nn

from net import Net


class Layer(nn.Module, ABC):
    def __init__(self, in_dim, out_dim, z_dim=0):
        torch.manual_seed(0)
        super(Layer, self).__init__()
        self.fc = nn.Linear(in_dim + z_dim, out_dim)

    def forward(self, x, z=None):
        if z is not None:
            x = torch.cat([x, z], 1)
        return self.fc(x)

    def pen(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                # and not name.endswith(".bias"):
                penalty += torch.sum(param ** 2)
        return penalty


class DNN(Net, ABC):
    def __init__(self, models):
        super(DNN, self).__init__()
        self.layers = len(models)
        self.hyper_lamb = [10 ** x for x in range(-1, 3)]
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        res = self.model0(dataset.x, dataset.z)
        for i in range(1, self.layers):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
        return res*self.std + self.mean

    def fit_init(self, dataset):
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.size = dataset.x.shape[0]

    def fit_end(self):
        print(self.epoch, self.loss)

    def penalty(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                # and not name.endswith(".bias"):
                penalty += torch.sum(param ** 2)
        return penalty * self.lamb

    def transfer(self, dataset):
        res = self.model0(dataset.x, dataset.z)
        for i in range(1, self.layers-1):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
            i += 1
        transfer_set = copy.deepcopy(dataset)
        transfer_set.x = torch.sigmoid(res)
        return transfer_set

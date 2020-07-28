import numpy as np
import torch
from torch import nn as nn

from basis import Basis
from net import Net


class MyModelB(nn.Module):
    def __init__(self, n_basis_0, n_basis_1):
        self.bs0, self.bs1 = Basis(n_basis_0), Basis(n_basis_1)
        self.index = None
        torch.manual_seed(0)
        super(MyModelB, self).__init__()
        self.fc0 = nn.Linear(self.bs0.n_basis, self.bs1.n_basis)

    def forward(self, x):
        x = self.fc0((x @ self.bs0.mat) / self.bs0.length)
        if self.index is None:
            return x @ self.bs1.mat.t()
        if type(self.index) is not int:
            x = x[self.index]
        x = x * self.bs1.mat
        return torch.sum(x, 1, keepdim=True)

    def pen(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                if name.endswith(".weight"):
                    penalty += self.bs1.pen_2d(param, self.bs0)
                if name.endswith(".bias"):
                    penalty += self.bs1.pen_1d(param)
        return penalty


class FDNN(Net):
    def __init__(self, models):
        self.realize = False
        super(FDNN, self).__init__()
        self.layers = len(models)
        self.hyper_lamb = [10**x for x in range(-2, 2)]
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        if not self.realize:
            self.realization(dataset)
        res = self.model0(dataset.x)
        for i in range(1, self.layers):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
        return res * self.std + self.mean

    def realization(self, dataset):
        device = dataset.x.device
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            if i == 0:
                pos = dataset.pos
                pos0 = dataset.pos0 if hasattr(dataset, 'pos0') else min(pos)
                pos1 = dataset.pos1 if hasattr(dataset, 'pos1') else max(pos)
                model.bs0.length = pos1 - pos0 + 1
                pos = (pos - pos0) / (pos1 - pos0)
            else:
                pos = np.arange(0.005, 1, 0.01)
                model.bs0.length = len(pos)
            model.bs0.evaluate(pos)
            model.bs0.to(device)
            if hasattr(self, 'model' + str(i + 1)):
                model.index = None
                loc = np.arange(0.005, 1, 0.01)
            else:
                model.index = -1
                loc = dataset.loc
                loc0 = dataset.loc0 if hasattr(dataset, 'loc0') else min(loc)
                loc1 = dataset.loc1 if hasattr(dataset, 'loc1') else max(loc)
                loc = (loc - loc0) / (loc1 - loc0)
            model.bs1.evaluate(loc)
            model.bs1.to(device)

    def fit_init(self, dataset):
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.size = dataset.x.shape[0]
        self.realization(dataset)
        self.realize = True

    def fit_end(self):
        self.realize = False
        print(self.epoch, self.loss)

    def penalty(self):
        penalty = 0
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            penalty += model.pen()
        return penalty

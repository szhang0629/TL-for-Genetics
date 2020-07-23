import numpy as np
import torch

from skfda.representation.basis import BSpline, Fourier
from torch import nn as nn

from solution import Net


class Basis:
    def __init__(self, n_basis):
        self.n_basis = n_basis
        self.bss = Fourier((0, 1), n_basis=n_basis, period=2)
        pen0 = self.bss.gram_matrix() * 2
        vec = np.append(0.5, np.repeat(np.arange(1, n_basis//2+1), 2))
        vec = vec.reshape(n_basis, 1)
        gram = vec @ vec.T
        self.pen0 = torch.from_numpy(pen0).float()
        self.pen2 = torch.from_numpy(pen0 * gram).float()
        self.mat = None
        self.length = None
        bs0 = Fourier((0, 1), n_basis=1, period=1)
        self.integral = \
            torch.from_numpy(self.bss.inner_product(bs0)).float() * (2**0.5)

    def evaluate(self, point):
        self.mat = torch.from_numpy(self.bss.evaluate(point).T).float() * \
                   (2**0.5)

    def to(self, device):
        self.mat = self.mat.to(device)
        self.pen0 = self.pen0.to(device)
        self.pen2 = self.pen2.to(device)
        self.integral = self.integral.to(device)


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
        if self.index is not -1:
            x = x[self.index]
        x = x * self.bs1.mat
        return torch.sum(x, 1, keepdim=True)


class FDNN(Net):
    def __init__(self, models):
        self.realize = False
        super(FDNN, self).__init__()
        for i in range(len(models)):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        if not self.realize:
            self.realization(dataset)
        res = self.model0(dataset.x)
        i = 1
        while hasattr(self, 'model' + str(i)):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
            i += 1
        return res * self.std + self.mean

    def realization(self, dataset):
        device = dataset.x.device
        i = 0
        while hasattr(self, 'model' + str(i)):
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
            i += 1

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
        i = 0
        while hasattr(self, 'model' + str(i)):
            model = getattr(self, 'model' + str(i))
            for name, param in model.named_parameters():
                if param.requires_grad and "fc" in name:
                    if name.endswith(".weight"):
                        pen = torch.trace(model.bs0.pen0 @
                                          ((param @ model.bs1.pen2) @
                                           param.t()) + model.bs1.pen0 @
                                          ((param.t() @ model.bs0.pen2) @
                                           param)) * 0.5
                        mean = torch.sum(param * (model.bs0.integral.t() @
                                                  model.bs1.integral))
                        var = torch.trace((model.bs0.pen0 @
                                           ((param @ model.bs1.pen0) @
                                            param.t()))) - (mean ** 2)
                        pen = pen / var
                        penalty += pen
                    if name.endswith(".bias"):
                        pen = param @ model.bs1.pen2 @ param
                        mean = param @ model.bs1.integral
                        var = param @ model.bs1.pen0 @ param - mean @ mean
                        pen = pen / var
                        penalty += pen
            i += 1
        return penalty

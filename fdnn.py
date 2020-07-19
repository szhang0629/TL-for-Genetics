import numpy as np
import torch
import torch.nn as nn
from solution import Net


class MyModelB(nn.Module):
    def __init__(self, bs0, bs1):
        self.bs0, self.b0 = bs0, None
        # self.p0_0 = torch.from_numpy(bs0.penalty(0)) * bs0.n_basis
        # self.p2_0 = torch.from_numpy(bs0.penalty(2)) * bs0.n_basis / 100 \
        #             + self.p0_0 * 0.05
        self.p0_0 = torch.from_numpy(bs0.penalty(0)) * 2
        self.p2_0 = torch.from_numpy(bs0.penalty(2)) * 2
        self.bs1, self.b1 = bs1, None
        # self.p0_1 = torch.from_numpy(bs1.penalty(0)) * bs1.n_basis
        # self.p2_1 = torch.from_numpy(bs1.penalty(2)) * bs1.n_basis / 100 \
        #             + self.p0_1 * 0.05
        self.p0_1 = torch.from_numpy(bs0.penalty(0)) * 2
        self.p2_1 = torch.from_numpy(bs0.penalty(2)) * 2
        self.index = None
        super(MyModelB, self).__init__()
        self.fc0 = nn.Linear(self.bs0.n_basis, self.bs1.n_basis)

    def forward(self, x):
        x = self.fc0((x @ self.b0) / self.b0.shape[0])
        if self.index is None:
            return x @ self.b1.t()
        if self.index is not -1:
            x = x[self.index]
        x = x * self.b1
        return torch.sum(x, 1, keepdim=True)


class FDNN(Net):
    def __init__(self, *args):
        self.realize = False
        super(FDNN, self).__init__()
        i = 0
        for arg in args:
            setattr(self, "model" + str(i), arg)
            i += 1

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
            point = dataset.pos if i == 0 else np.arange(0.005, 1, 0.01)
            # model.b0 = torch.from_numpy(model.bs0.evaluate(point).T) * \
            #            (model.bs0.n_basis ** 0.5)
            model.b0 = torch.from_numpy(model.bs0.evaluate(point).T) * (2**0.5)
            model.b0 = model.b0.to(device)
            model.p0_0, model.p2_0 = model.p0_0.to(device), \
                                     model.p2_0.to(device)
            point = None if hasattr(self, 'model' + str(i + 1)) else dataset.loc
            if point is None:
                model.index = None
                point = np.arange(0.005, 1, 0.01)
            else:
                model.index = -1
            # model.b1 = torch.from_numpy(model.bs1.evaluate(point).T) * \
            #            (model.bs1.n_basis ** 0.5)
            model.b1 = torch.from_numpy(model.bs1.evaluate(point).T) * (2**0.5)
            model.b1 = model.b1.to(device)
            model.p0_1, model.p2_1 = model.p0_1.to(device), \
                                     model.p2_1.to(device)
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
                    if "weight" in name:
                        penalty += torch.trace(
                            (model.p0_0 @ ((param @ model.p2_1)
                                           @ param.t()))) * 0.5
                        penalty += torch.trace(
                            (model.p0_1 @ ((param.t() @ model.p2_0)
                                           @ param))) * 0.5
                    if "bias" in name:
                        penalty += torch.trace(
                            (torch.unsqueeze(param, 0) @ model.p2_1) @
                            torch.unsqueeze(param, 1))
            i += 1
        return penalty

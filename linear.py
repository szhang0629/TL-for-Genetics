import torch
import copy
import numpy as np


class Linear:
    def __init__(self, dataset=None, base=False, lamb=1.0):
        self.size, self.loss, self.lamb = None, None, lamb
        self.base = base
        self.beta = None
        if dataset is not None:
            if base:
                self.fit(dataset, base)
            else:
                cache = self.train(dataset, lamb)
                self.__dict__.update(cache.__dict__)

    def __call__(self, dataset):
        bias = torch.ones(dataset.y.shape, device=dataset.y.device)
        if self.base:
            return self.beta * bias
        else:
            x = torch.cat([bias, dataset.x], 1)
            return x @ self.beta

    def fit(self, dataset, base=False, lamb=1.0):
        self.base, self.size, self.lamb = base, dataset.y.shape[0], lamb
        if self.base:
            self.beta = torch.mean(dataset.y.double())
        else:
            bias = torch.ones(self.size, 1, device=dataset.x.device)
            x = torch.cat([bias, dataset.x], 1)
            x_t = x.transpose(0, 1)
            mat = torch.eye(x.shape[1], device=dataset.x.device)
            mat[0, 0] = 0
            self.beta = torch.inverse((x_t @ x) + lamb * mat) @ x_t @ dataset.y
        self.loss = torch.nn.MSELoss()(self(dataset), dataset.y).tolist()

    def train(self, dataset, lamb):
        if type(lamb) is not list:
            net = copy.deepcopy(self)
            net.fit(dataset, lamb=lamb)
            return net
        trainset, validset = dataset.split_seed()
        valid = [validset.loss(self.train(trainset, decay)).tolist()
                 for decay in lamb]
        print(valid)
        lamb_opt = lamb[np.argmin(np.array(valid))]
        return self.train(dataset, lamb_opt)

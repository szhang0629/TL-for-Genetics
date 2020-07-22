import copy
import os

import pandas as pd
import torch
from torch import nn as nn
from torch import optim as optim

from data import DiscreteData2, FunctionalData2


class Solution:
    def __init__(self):
        self.size, self.loss, self.lamb = None, float('Inf'), 1.0

    def hyper_train(self, dataset, lamb):
        if type(lamb) is not list or len(lamb) == 1:
            if type(lamb) is list:
                lamb = lamb[0]
            net = copy.deepcopy(self)
            net.fit(dataset, lamb=lamb)
            return net
        trainset, validset = dataset.split_seed()
        valid = [validset.loss(self.hyper_train(trainset, decay)).tolist()
                 for decay in lamb]
        print(valid)
        lamb_opt = lamb[torch.argmin(torch.FloatTensor(valid))]
        return self.hyper_train(dataset, lamb_opt)

    def to_df(self, testset, method):
        return pd.DataFrame(data={'method': [method], 'pen': [self.lamb],
                                  'train': [self.loss],
                                  'test': [testset.loss(self).tolist()]})


class Linear(Solution):
    def __init__(self, dataset=None, base=False, lamb=None):
        super(Linear, self).__init__()
        if lamb is None:
            lamb = [10 ** x for x in range(-4, 4)]
        self.base = base
        self.beta = None
        if dataset is not None:
            self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
            if base:
                self.fit(dataset, base)
            else:
                self.scalar_mean, self.scalar_std = None, None
                cache = self.hyper_train(dataset, lamb)
                self.__dict__.update(cache.__dict__)
        else:
            self.mean, self.std = 0, 1

    def __call__(self, dataset):
        bias = torch.ones(dataset.y.shape, device=dataset.y.device)
        if self.base:
            return self.mean * bias
        else:
            x_scalar = (dataset.x-self.scalar_mean)/self.scalar_std
            return (x_scalar @ self.beta) * self.std + self.mean

    def fit(self, dataset, base=False, lamb=1.0):
        self.base, self.size, self.lamb = base, dataset.y.shape[0], lamb
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        if not self.base:
            x = dataset.x
            self.scalar_mean = torch.mean(x, 0)
            # self.scalar_std = torch.std(x, 0)
            self.scalar_std = torch.ones(x.shape[1], device=x.device)
            x = (x-self.scalar_mean)/self.scalar_std
            x_t = x.transpose(0, 1)
            mat = torch.eye(x.shape[1], device=dataset.x.device)
            self.beta = torch.inverse((x_t @ x) + lamb * mat) @ x_t @ dataset.y
        self.loss = torch.nn.MSELoss()(self(dataset), dataset.y).tolist()


class Net(nn.Module, Solution):
    def __init__(self):
        self.epoch, self.mean, self.std = 0, 0, 1
        super(Net, self).__init__()

    def fit(self, dataset, lamb):
        self.fit_init(dataset)
        self.epoch, self.lamb, k = 0, lamb, 0
        cache = copy.deepcopy(self)
        optimizer = optim.Adam(self.parameters())  # , weight_decay=self.lamb)
        risk_min = float('Inf')
        while True:
            optimizer.zero_grad()
            loss = dataset.loss(self)
            self.loss = loss.tolist()
            self.eval()
            risk = loss / (self.std ** 2) + self.penalty()*self.lamb
            if self.epoch % 10 == 0:
                if risk < risk_min:
                    cache = copy.deepcopy(self)
                    risk_min = risk.tolist()
                    k = 0
                if k == 100:
                    break
                if self.epoch % 1000 == 0:
                    print(self.epoch, self.loss, risk.tolist())
            risk.backward()
            optimizer.step()
            self.epoch, k = self.epoch+1, k+1
        self.__dict__.update(cache.__dict__)
        self.fit_end()

    def save(self, folder, name, keep=None):
        if keep is not None:
            i = 0
            while hasattr(self, 'model' + str(i + keep)):
                for param in getattr(self, 'model' + str(i)).parameters():
                    param.requires_grad = False
                i += 1
        os.makedirs(folder, exist_ok=True)
        torch.save(self, folder + name)

    def pre_train(self, name, gene):
        method = self.__class__.__name__
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dir_pre = os.path.join("..", "Models", method, "")
        if os.path.exists(dir_pre + name):
            net_pre = torch.load(dir_pre + name, map_location=device)
            net_pre.eval()
        else:
            if method == "DNN":
                # oldset = DiscreteData1(gene, data="ukb", race=1001,
                #                        target=data)
                oldset = DiscreteData2(gene, data="ukb")
                net_pre = self.hyper_train(oldset,
                                          [10 ** (x / 2) for x in
                                           range(-8, -4)])
            if method == "FDNN":
                oldset = FunctionalData2(gene, data="ukb")
                net_pre = self.hyper_train(oldset,
                                          [10 ** x for x in range(-6, -3)])
            net_pre.save(dir_pre, name, 1)
        return net_pre



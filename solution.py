import copy
import os

import numpy as np
import pandas as pd
import torch


class Solution:
    """
    A solution to present a regression model
    """
    def __init__(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.size, self.loss = None, float('Inf')
        self.lamb, self.lamb_sca, self.hyper_lamb = None, None, 1.0
        self.epoch, self.str_units = 0, "0"

    def hyper_train(self, dataset, lamb=None):
        if lamb is None:
            lamb = self.hyper_lamb
        if type(lamb) is not list or len(lamb) == 1:
            if type(lamb) is list:
                lamb = lamb[0]
            net = copy.deepcopy(self)
            net.fit(dataset, lamb)
            return net
        dataset.split_seed()
        valid = [dataset.test.loss(
            self.hyper_train(dataset.train, decay)).tolist() for decay in lamb]
        print(valid)
        lamb_opt = lamb[np.argmin(valid)]
        return self.hyper_train(dataset, lamb_opt)

    def to_df(self, testset, method):
        return pd.DataFrame(data={'method': [method], 'pen': [self.lamb],
                                  'hidden': [self.str_units],
                                  'epoch': [self.epoch], 'train': [self.loss],
                                  'test': [testset.loss(self).tolist()]})

    def to_csv(self, testset, method=None):
        if method is None:
            method = self.__class__.__name__
        os.makedirs(os.path.dirname(testset.out_path), exist_ok=True)
        res = self.to_df(testset, method)
        print(res)
        if os.path.isfile(testset.out_path):
            methods = pd.read_csv(testset.out_path).loc[:, 'method'].values
            if method not in methods:
                res.to_csv(testset.out_path, index=False,
                           mode='a', header=False)
        else:
            res.to_csv(testset.out_path, index=False)

    def fit(self, *args):
        return

    def to(self, *args):
        return self


class Base(Solution):
    def __init__(self, dataset=None):
        super(Base, self).__init__()
        self.lamb = 1.
        if dataset is not None:
            self.mean, self.std = dataset.y.mean(), dataset.y.std()
            self.fit(dataset)
        else:
            self.mean, self.std = 0, 1

    def __call__(self, dataset):
        bias = torch.ones(dataset.y.shape, device=dataset.y.device)
        return self.mean * bias

    def fit(self, dataset, lamb=None):
        self.size = dataset.y.shape[0]
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.loss = torch.nn.MSELoss()(self(dataset), dataset.y).tolist()


class Ridge(Solution):
    def __init__(self, dataset=None, lamb=None):
        super(Ridge, self).__init__()
        if lamb is None:
            self.hyper_lamb = [(10**4) * (3**x)  for x in range(-2, 3)]
        self.beta = None
        if dataset is not None:
            self.mean, self.std = dataset.y.mean(), dataset.y.std()
            self.scalar_mean, self.scalar_std = None, None
            cache = self.hyper_train(dataset, lamb)
            self.__dict__.update(cache.__dict__)
        else:
            self.mean, self.std = 0, 1

    def __call__(self, dataset):
        x_scalar = (dataset.x-self.scalar_mean)/self.scalar_std
        return (x_scalar @ self.beta) * self.std + self.mean

    def fit(self, dataset, lamb=None):
        if lamb is not None:
            self.lamb = lamb
        self.size = dataset.y.shape[0]
        self.lamb_sca = self.lamb / (self.size ** 0.5)
        self.mean, self.std = dataset.y.mean(), dataset.y.std()
        x = dataset.x
        self.scalar_mean = torch.mean(x, 0)
        # self.scalar_std = torch.std(x, 0)
        self.scalar_std = torch.ones(x.shape[1], device=x.device)
        x = (x-self.scalar_mean)/self.scalar_std
        x_t = x.transpose(0, 1)
        mat = torch.eye(x.shape[1], device=dataset.x.device)
        self.beta = ((x_t @ x) + self.lamb_sca * mat).inverse() @ x_t @ dataset.y
        self.loss = torch.nn.MSELoss()(self(dataset), dataset.y).tolist()


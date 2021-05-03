import os
import math

import pandas as pd
import torch


class Solution:
    """
    A solution to present a regression model
    """
    def __init__(self, lamb=None):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.size, self.loss = None, float('Inf')
        self.hyper_lamb = 1. if lamb is None else lamb
        self.lamb, self.lr = None, 0.
        self.epoch, self.str_units = 0, "0"
        self.mean, self.std = 0, 1
        self.method_name = self.__class__.__name__
        self.criterion = torch.nn.MSELoss()

    def to_df(self, testset):
        return pd.DataFrame(
            data={'method': [self.method_name],
                  'pen': [None if self.lamb is None else math.log10(self.lamb)],
                  'hidden': [self.str_units], 'epoch': [self.epoch],
                  'lr': [self.lr],
                  'train': [self.loss], 'test': [testset.loss(self).tolist()]})

    def to_csv(self, testset):
        os.makedirs(os.path.dirname(testset.out_path), exist_ok=True)
        res = self.to_df(testset)
        pd.options.display.float_format = '{:,.8f}'.format
        print(res)
        if os.path.isfile(testset.out_path):
            # methods = pd.read_csv(testset.out_path).loc[:, 'method'].values
            # # pens = pd.read_csv(testset.out_path).loc[:, 'pen'].values
            # if self.method_name not in methods:
            res.to_csv(testset.out_path, index=False, mode='a', header=False)
        else:
            res.to_csv(testset.out_path, index=False)

    def fit(self, *args):
        return

    def to(self, *args):
        return self


class Base(Solution):
    def __init__(self, lamb=None):
        super(Base, self).__init__(lamb)

    def __call__(self, dataset):
        bias = torch.ones(dataset.y.shape, device=dataset.y.device)
        return self.mean * bias

    def fit(self, dataset, lamb=None):
        self.size = dataset.y.shape[0]
        self.mean, self.std = dataset.y.mean(), dataset.y.std()
        self.loss = self.criterion(self(dataset), dataset.y).tolist()
        print(self.loss, self.mean.tolist(), self.std.tolist(), self.mean)


class Ridge(Solution):
    def __init__(self, lamb=None):
        super(Ridge, self).__init__(lamb)
        if lamb is None:
            self.hyper_lamb = [(10**4) * (3**x) for x in range(-2, 3)]
        else:
            self.hyper_lamb = lamb
        self.beta = None

    def __call__(self, dataset):
        x_scalar = (dataset.x-self.scalar_mean)/self.scalar_std
        return (x_scalar @ self.beta) * self.std + self.mean

    def fit(self, dataset, lamb=None):
        if lamb is not None:
            self.lamb = lamb
        self.size = dataset.y.shape[0]
        self.mean, self.std = dataset.y.mean(), dataset.y.std()
        x = dataset.x
        self.scalar_mean = x.mean(0)
        # self.scalar_std = torch.std(x, 0)
        self.scalar_std = torch.ones(x.shape[1], device=x.device)
        x = (x-self.scalar_mean)/self.scalar_std
        x_t = x.transpose(0, 1)
        mat = self.lamb / (self.size ** 0.5) * \
              torch.eye(x.shape[1], device=dataset.x.device)
        self.beta = ((x_t @ x) + mat).inverse() @ x_t @ dataset.y
        self.loss = self.criterion(self(dataset), dataset.y).tolist()

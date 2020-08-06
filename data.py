import copy
import random

import torch
from torch import nn as nn


class Data:
    """
    A class to represent a group of data for prediction
    ...
    Attributes
    ----------
    y : numpy array or tensor
        response variable
    x : numpy array or tensor
        predictor variable (SNP data)
    z : numpy array or tensor
        predictor variable (covariates)
    pos : numpy array
        position of SNP data corresponding to x
    loc : numpy array or None
        coordinate of predictor variableAAAAA
    """
    def __init__(self, data):
        """
        The constructor for Data class
        :param data(list) : An ordered list of essential data
        """
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.y, self.x, self.z, self.pos, self.loc = data

    def to_tensor(self):
        self.y = torch.from_numpy(self.y).double()
        self.x = torch.from_numpy(self.x)
        if self.z is not None:
            self.z = torch.from_numpy(self.z)

    def to_numpy(self):
        self.y = self.y.cpu().numpy()
        self.x = self.x.cpu().numpy()
        if self.z is not None:
            self.z = self.z.cpu().numpy()

    def to(self, device):
        self.y, self.x = self.y.to(device), self.x.to(device)
        self.z = None if self.z is None else self.z.to(device)

    def split_seed(self, seed=0, split_ratio=0.8):
        sequence = list(range(self.x.shape[0]))
        random.Random(seed).shuffle(sequence)
        point = round(len(sequence) * split_ratio)
        return self.split([sequence[:point], sequence[point:]])

    def split(self, seq):
        res = []
        for seq_ in seq:
            res_ = copy.deepcopy(self)
            if res_.z is not None:
                res_.z = self.z[seq_]
            res_.y, res_.x = res_.y[seq_], res_.x[seq_]
            if res_.loc is not None:
                res_.loc = res_.loc[seq_]
            res.append(res_)
        return res

    def func(self, column='age'):
        self.loc = self.z[[column]].to_numpy().ravel()

    def scale_std(self):
        x_mean, x_std = self.x.mean(), self.x.std()
        points_ratio = len(self.pos)/(max(self.pos) - min(self.pos))
        std_ratio = (points_ratio*(x_mean**2 + x_std**2) -
                     (points_ratio**2) * (x_mean**2)) ** 0.5
        return 1 / std_ratio / 10

    def process(self):
        self.to_tensor()
        self.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.x = self.x.double()
        self.z = None

    def loss(self, model, criterion=nn.MSELoss()):
        return criterion(model(self), self.y)

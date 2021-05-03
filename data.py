import copy
import math
import random
import numpy as np

import torch
from sklearn.metrics import roc_auc_score


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
        self.pos0, self.pos1, self.loc0, self.loc1 = None, None, None, None
        self.train, self.test, self.seed = None, None, None
        self.out_path = None

    def to_tensor(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.y = torch.from_numpy(self.y.astype(np.float64))
        self.x = torch.from_numpy(self.x.astype(np.float64))
        if self.z is not None:
            self.z = torch.from_numpy(self.z.astype(np.float64))
        # device = torch.device("cpu")
        device = \
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def to_numpy(self):
        self.y = self.y.cpu().numpy()
        self.x = self.x.cpu().numpy()
        if self.z is not None:
            self.z = self.z.cpu().numpy()

    def to(self, device):
        self.y, self.x = self.y.to(device), self.x.to(device)
        self.z = None if self.z is None else self.z.to(device)

    def split_seed(self, seed=0, split_ratio=0.8, out=False):
        y = self.y.cpu().numpy().ravel().tolist()
        seq_0 = [i for i in range(len(y)) if y[i] == 0]
        seq_1 = [i for i in range(len(y)) if y[i] != 0]
        random.Random(seed).shuffle(seq_0)
        random.Random(seed).shuffle(seq_1)
        point_0, point_1 = \
            round(len(seq_0) * split_ratio), round(len(seq_1) * split_ratio)
        self.seed = seed
        self.out_path += str(seed) + ".csv"
        return self.split([seq_0[:point_0] + seq_1[:point_1],
                           seq_0[point_0:] + seq_1[point_1:]])

    def row_select(self, seq, out=True):
        if not out:
            if self.z is not None:
                self.z = self.z[seq]
            self.y, self.x = self.y[seq], self.x[seq]
            if self.loc is not None:
                self.loc = self.loc[seq]
            return
        res = copy.deepcopy(self)
        if self.z is not None:
            res.z = self.z[seq]
        res.y, res.x = self.y[seq], self.x[seq]
        if self.loc is not None:
            res.loc = self.loc[seq]
        return res

    def split(self, seq):
        return [self.row_select(seq_) for seq_ in seq]

    def bootstrap(self, seed=0, out=False):
        self.seed = seed
        self.out_path += str(seed) + ".csv"
        np.random.seed(0)
        length = self.x.shape[0]
        train = np.random.choice(range(length), length, True)
        valid = list(set(range(length)) - set(train))
        if out:
            return self.split([train, valid])
        else:
            self.train, self.test = self.split([train, valid])

    def func(self, column='age'):
        self.loc = self.z[[column]].to_numpy().ravel()

    def scale_std(self):
        x_mean, x_std = self.x.mean(), self.x.std()
        points_ratio = len(self.pos)/(max(self.pos) - min(self.pos))
        std_ratio = (points_ratio*(x_mean**2 + x_std**2) -
                     (points_ratio**2) * (x_mean**2)) ** 0.5
        return 1 / std_ratio

    def process(self):
        y = self.y.ravel()
        seq = range(len(y))
        seq = [num for num in seq if y[num] < y.mean() + 3 * y.std()]
        self.to_tensor()
        self.z = None
        self.row_select(seq, out=False)

    def loss(self, model):
        return model.criterion(model(self), self.y)

    def auc(self, model):
        y_score = model(self)
        yy = y_score[:, 1] > y_score[:, 0]
        return roc_auc_score(self.y.cpu().detach().numpy(), yy)

    def find_interval(self, target):
        if target is None:
            delta_pos = (max(self.pos) - min(self.pos)) / 100
            self.pos0, self.pos1 = \
                min(self.pos) - delta_pos, max(self.pos) + delta_pos
            if self.loc is not None:
                delta_loc = (max(self.loc) - min(self.loc)) / 100
                self.loc0, self.loc1 = \
                    min(self.loc) - delta_loc, max(self.loc) + delta_loc
        else:
            self.pos0, self.pos1 = target.pos0, target.pos1
            self.loc0, self.loc1 = target.loc0, target.loc1

    # def tuning(self, model, lamb=None):
    #     model.to(self.y.device)
    #     trainset = self.train
    #     model_tuned = trainset.hyper_train(model, lamb)
    #     model_tuned.to_csv(self.test)
    #     return model_tuned

    def hyper_train(self, model, lamb=None):
        if lamb is None:
            lamb = model.hyper_lamb
        if type(lamb) is not list or len(lamb) == 1:
            if type(lamb) is list:
                lamb = lamb[0]
            net = copy.deepcopy(model)
            net.fit(self, lamb)
            return net
        trainset, validset = self.split_seed(split_ratio=0.8)
        # trainset, validset = self.bootstrap(out=True)
        baseline = model.criterion(
            trainset.y.mean() *
            torch.ones(validset.y.shape, device=validset.y.device),
            validset.y).tolist()
        print((trainset.y == 0).sum() / trainset.x.shape[0],
              (validset.y == 0).sum() / validset.x.shape[0],
              model.criterion(
                  trainset.y.mean() *
                  torch.ones(trainset.y.shape, device=trainset.y.device),
                  trainset.y).tolist(),
              baseline)
        # valid = []
        lamb_cache, valid_cache = lamb[0], float('Inf')
        for decay in lamb:
            model_ = copy.deepcopy(model)
            method = trainset.hyper_train(model_, decay)
            # valid.append(validset.loss(method).tolist())
            valid = validset.loss(method).tolist()
            print("lamb:", math.log10(decay), "valid:", valid)
            if valid > valid_cache or valid > baseline:
                break
            else:
                lamb_cache, valid_cache = decay, valid
        # # methods = [trainset.hyper_train(model, decay) for decay in lamb]
        # # valid = [validset.loss(method).tolist() for method in methods]
        # print(valid)
        # # method = methods[np.argmin(valid)]
        # lamb_opt = lamb[np.argmin(valid)]
        # method = trainset.hyper_train(model, lamb_cache)
        return self.hyper_train(model, lamb_cache)

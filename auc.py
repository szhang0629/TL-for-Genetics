import copy
import random
import torch

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class SeedSequence:
    def __init__(self, seed, n, split_ratio=0.8):
        self.sequence = list(range(n))
        random.Random(seed).shuffle(self.sequence)
        point = round(n * split_ratio)
        self.sequence_split = [self.sequence[:point], self.sequence[point:]]


def ann(data_set, net, lamb, criterion):
    """penalty hyper parameter selection"""
    length = len(lamb)
    if length > 1:
        valid_list = np.zeros(length)
        train, valid = SeedSequence(629, data_set.y.shape[0]).sequence_split
        dataset_train, dataset_valid = data_set.split([train, valid])
        for i in range(length):
            net_copy = copy.deepcopy(net)
            net_copy.my_train(dataset_train.data, lamb[i], criterion)
            valid_list[i] = criterion(net_copy(dataset_valid.g, dataset_valid.x), dataset_valid.y).tolist()

        print(valid_list)
        lamb = lamb[np.argmin(valid_list)]
    else:
        lamb = lamb[0]

    net.my_train(data_set.data, lamb, criterion)
    return net


def inn(data_train, data_test, net, lamb_hyper, criterion):
    net_result = ann(data_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_result(data_train.g, data_train.x), data_train.y).tolist()
    loss_test = criterion(net_result(data_test.g, data_test.x), data_test.y).tolist()
    return pd.DataFrame(data={'method': ["NN"], 'train': [loss_train], 'test': [loss_test]})


def base(y_train, y_test, criterion):
    mean_train = torch.mean(y_train)
    loss_train = criterion(mean_train * torch.ones(y_train.shape, device=y_train.device), y_train).cpu().tolist()
    loss_test = criterion(mean_train * torch.ones(y_test.shape, device=y_train.device), y_test).cpu().tolist()
    return pd.DataFrame(data={'method': ["Base"], 'train': [loss_train], 'test': [loss_test]})


def my_lasso(data, data_test, lamb, criterion):
    if type(lamb) is list:
        length = len(lamb)
        if length > 1:
            valid_list = np.zeros(length)
            train, valid = SeedSequence(629, data.y.shape[0]).sequence_split
            data_train, data_valid = data.split([train, valid])
            for i in range(length):
                valid_list[i] = my_lasso(data_train, data_valid, lamb[i], criterion)['test'][0]

            print(valid_list)
            lamb = lamb[np.argmin(valid_list)]
        else:
            lamb = lamb[0]

    data = data.numpy()
    data_test = data_test.numpy()
    clf = linear_model.Lasso(alpha=lamb)
    clf.fit(data[1], data[0])
    loss_train = criterion(torch.from_numpy(clf.predict(data[1])[None].T), torch.from_numpy(data[0])).cpu().tolist()
    loss_test = criterion(torch.from_numpy(clf.predict(data_test[1])[None].T), torch.from_numpy(data_test[0])).cpu().tolist()
    return pd.DataFrame(data={'method': ["LM"], 'train': [loss_train], 'test': [loss_test]})


def my_log(data, data_test):
    clf = LogisticRegression(random_state=0, C=0.00001).fit(data.g, data.y)
    loss_train = log_loss(data_test.y, clf.predict(data_test.g))
    loss_test = log_loss(data.y, clf.predict(data.g))

    return pd.DataFrame(data={'method': ["LR"], 'train': [loss_train], 'test': [loss_test]})
import copy
import random
import torch

import numpy as np
import pandas as pd


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
            net_copy.fit(dataset_train, lamb[i], criterion)
            valid_list[i] = criterion(net_copy(dataset_valid), dataset_valid.y).tolist()

        print(valid_list)
        lamb = lamb[np.argmin(valid_list)]
    else:
        lamb = lamb[0]

    net.fit(data_set, lamb, criterion)
    return net, lamb


def base(y_train, y_test, criterion, classification=False):
    if classification:
        mean_train = torch.mean(y_train.float()).cpu().tolist()
        mean_train_ = torch.tensor([[mean_train, 1 - mean_train]], device=y_train.device)
        loss_train = criterion(torch.cat(y_train.shape[0]*[mean_train_]), y_train).tolist()
        loss_test = criterion(torch.cat(y_test.shape[0]*[mean_train_]), y_test).tolist()
    else:
        mean_train = torch.mean(y_train)
        loss_train = criterion(mean_train * torch.ones(y_train.shape, device=y_train.device), y_train).tolist()
        loss_test = criterion(mean_train * torch.ones(y_test.shape, device=y_train.device), y_test).tolist()

    return pd.DataFrame(data={'method': ["Base"], 'pen': [0], 'train': [loss_train], 'test': [loss_test]})

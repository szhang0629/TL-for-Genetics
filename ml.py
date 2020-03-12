"""Apply Transfer Learning in Genetics"""
# !/usr/bin/env python
# coding: utf-

import copy

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from data2 import *
from auc import *
from penalty_parameters import penalty_parameters


def ml(data_set, data_old, net, lamb, criterion, ratio_old=1):
    length = len(lamb)
    if length > 1:
        x, z, y = data_set
        valid_list = np.zeros(length)
        train, valid = sample_index(629, y)
        y_train, y_valid = y[train], y[valid]
        if z is None:
            z_train, z_valid = None, None
        else:
            z_train, z_valid = z[train], z[valid]
        if type(x) is list:
            x_train, x_valid = [x_[train] for x_ in x], [x_[valid] for x_ in x]
        else:
            x_train, x_valid = x[train], x[valid]
        data_train = [x_train, z_train, y_train]
        for i in range(length):
            net_cache = ml1(data_train, data_old, copy.deepcopy(net), lamb[i], criterion, ratio_old)
            valid_list[i] = criterion(net_cache(x_valid), y_valid).tolist()

        print(valid_list)
        lamb = lamb[np.argmin(valid_list)]
    else:
        lamb = lamb[0]

    return ml1(data_set, data_old, copy.deepcopy(net), lamb, criterion, ratio_old)


def ml1(data_set, data_old, net, lamb, criterion, ratio_old):
    g, z, y = data_set
    g_old, z_old, y_old = data_old
    optimizer = optim.Adam(net.parameters())
    epoch, k, = 0, 0
    loss_min = np.float('Inf')

    while k < 100:
        optimizer.zero_grad()
        output1, output2 = net(g, g_old)
        mse = criterion(output1, y) + criterion(output2, y_old)*ratio_old
        penalty = penalty_parameters(net.modelA) / (y.shape[0] + y_old.shape[0]) +\
            penalty_parameters(net.modelB1) / y.shape[0] +\
            penalty_parameters(net.modelB2) / y_old.shape[0]
        loss = mse + penalty * lamb
        if epoch % 10 == 0:
            # if epoch % 1000 == 0:
            #     print(epoch, k, mse.tolist(), penalty.tolist())
            loss_cpu = loss.tolist()
            if loss_cpu < loss_min * 0.999:
                net_cache = copy.deepcopy(net)
                loss_min, k = loss_cpu, 0
            else:
                k += 1
        loss.backward()
        optimizer.step()
        epoch += 1

    output1 = net_cache(g)
    mse = criterion(output1, y)
    pen = penalty_parameters(net_cache.modelA) / (y.shape[0] + y_old.shape[0]) +\
        penalty_parameters(net_cache.modelB1) / y.shape[0]
    print(epoch - 100 * 10 - 1, "mse:", mse.tolist(), "pen:", lamb * pen.tolist())
    return net_cache

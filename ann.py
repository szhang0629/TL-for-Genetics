import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from penalty_parameters import penalty_parameters
from sample_r import *


def ann(x, z, y, net, lamb, criterion):
    length = len(lamb)
    if length > 1:
        valid_list = np.zeros(length)
        train, valid = sample_index(629, y)
        y_train, y_valid = y[train], y[valid]
        z_train, z_valid = z[train], z[valid]
        if type(x) is list:
            x_train, x_valid = [], []
            for i in range(len(x)):
                x_train.append(x[i][train])
                x_valid.append(x[i][valid])
        else:
            x_train, x_valid = x[train], x[valid]
        for i in range(length):
            net_cache = ann1(x_train, z_train, y_train, copy.deepcopy(net), lamb[i], criterion)
            valid_list[i] = criterion(net_cache(x_valid, z_valid), y_valid).tolist()

        print(valid_list)
        lamb = lamb[np.argmin(valid_list)]
    else:
        lamb = lamb[0]

    return ann1(x, z, y, copy.deepcopy(net), lamb, criterion)

    
def ann1(x, z, y, net, lamb, criterion):
    # optimizer = optim.Adam(net.parameters())
    optimizer = optim.Adadelta(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epoch, k, lamb0 = 0, 0, lamb/y.shape[0]
    loss_min = np.float('Inf')

    while k < 100:
        optimizer.zero_grad()
        output = net(x, z)
        mse, penalty = criterion(output, y), penalty_parameters(net)*lamb0
        loss = mse + penalty
        if epoch % 10 == 0:
            # if epoch % 100 == 0:
            #     print(epoch, k, loss_min, mse.tolist(), penalty.tolist())
            if loss.cpu().detach().numpy() < loss_min:
                loss_min = loss.cpu().detach().numpy()
                net_cache = copy.deepcopy(net)
                k = 0
            else:
                k += 1
        loss.backward()
        optimizer.step()
        epoch += 1

    output = net_cache(x, z)
    mse, pen = criterion(output, y).tolist(), penalty_parameters(net_cache).tolist()*lamb0
    print(epoch-100*10-1, "mse:", mse, "pen:", pen, "loss:", loss_min)
    return net_cache

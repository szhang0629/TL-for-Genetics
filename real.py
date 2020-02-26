#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from Net import Net
from data import data
from ann import *
from sample_r import *


def real(seed_index, name):
    g, x, y = data(name, seed_index)
    train, test = sample_index(seed_index, y)
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    net = Net(x.shape[1], y.shape[1], 16)
    criterion = nn.MSELoss()
    print(torch.std(y_train), torch.std(y_test), criterion(torch.mean(y_train) * torch.ones(y_test.shape), y_test))

    lamb_hyper = [0.01, 0.1, 1, 10, 100]
    net_result = ann(x_train, y_train, net, lamb_hyper, criterion)
    loss_train = criterion(net_result(x_train), y_train)
    loss_test = criterion(net_result(x_test), y_test)
    df = pd.DataFrame(data={'train': [loss_train.tolist()], 'test': [loss_test.tolist()]})
    print(df)
    # path_output = "../Output_nn/" + name + "/" + str(16)
    # if not os.path.exists(path_output):
    #     os.makedirs(path_output)

    # df.to_csv(path_output + "/" + str(seed_index) + ".csv", index=False)
    return

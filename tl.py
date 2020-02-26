'''Apply Transfer Learning in Genetics'''
#!/usr/bin/env python
# coding: utf-

import copy
import os

import pandas as pd
import torch.nn as nn

from ann import *
from data import data
from data2 import *
from models import *
from sample_r import *


def tl(seed_index, name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g, x, y = data(name, seed_index)
    xx = x[:, 0].numpy()
    old = np.arange(len(xx))[xx.astype(bool)]
    new = np.arange(len(xx))[(1-xx).astype(bool)]
    g_old, g_new, y_old, y_new = g[old], g[new], y[old], y[new]

    # g_new, y_new = data2(name, "ea")
    # g_old, y_old = data2(name, "ukb")
    train, test = sample_index(seed_index, g_new)
    g_train, g_test, y_train, y_test = g_new[train], g_new[test], y_new[train], y_new[test]
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()
    mean_train = torch.mean(y_train)
    loss_train = criterion(mean_train*torch.ones(y_train.shape), y_train).tolist()
    loss_test = criterion(mean_train*torch.ones(y_test.shape), y_test).tolist()
    df = pd.DataFrame(data={'method': ["Base"],
                            'train': [loss_train], 'test': [loss_test]})
    g_train, y_train, g_test, y_test = g_train.to(device), y_train.to(device), g_test.to(device), y_test.to(device)
    torch.manual_seed(629)

    model1 = MyModelA(g_new.shape[1], device)
    model2 = MyModelB(device)
    net = MyEnsemble(model1, model2)
    lamb_hyper = [1e-1, 1e0, 1e1, 1e2]
    # lamb_hyper = [1e3]
    net_result = ann(g_train, y_train, net, lamb_hyper, criterion)
    loss_train = criterion(net_result(g_train), y_train).cpu().tolist()
    loss_test = criterion(net_result(g_test), y_test).cpu().tolist()
    df = df.append(pd.DataFrame(data={'method': ["NN"], 
                                      'train': [loss_train], 'test': [loss_test]}))

    g_old, y_old = g_old.to(device), y_old.to(device)
    net_res = ann(g_old, y_old, net, lamb_hyper, criterion)
    for param in net_res.modelA.parameters():
        param.requires_grad = False

    net_result = ann(g_train, y_train, net_res, lamb_hyper, criterion)
    loss_train = criterion(net_result(g_train), y_train).cpu().tolist()
    loss_test = criterion(net_result(g_test), y_test).cpu().tolist()
    df = df.append(pd.DataFrame(data={'method': ["TL"], 
                                      'train': [loss_train], 'test': [loss_test]}))

    path_output = "../Output_nn/TL/" + name
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + "/" + str(seed_index) + ".csv", index=False)
    print(df)
    return

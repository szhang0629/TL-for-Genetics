#!/usr/bin/env python
# coding: utf-8

import copy
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from ann import *
from data_c import *
from data5_c import *
from model5 import *
from sample_r import *


def tl5(seed_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g, x, y = data("CHRNA3")
    g = [g]
    for name in ["CHRNA5", "CHRNA6", "CHRNB3", "CHRNB4"]:
        g_ = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
        g.append(torch.from_numpy(g_.values).float())

    x_race_indicator = x[:, 0].numpy()
    old = np.arange(len(x_race_indicator))[x_race_indicator.astype(bool)]
    new = np.arange(len(x_race_indicator))[(1 - x_race_indicator).astype(bool)]
    x = x[:, 1:3]
    y_old, y_new, x_old, x_new = y[old], y[new], x[old], x[new]
    g_old, g_new = [g_[old] for g_ in g], [g_[new] for g_ in g]

    # g_new, x_new, y_new = data5("ea")
    # g_old, x_old, y_old = data5("ukb")
    # x_new, x_old = x_new[:, 0:2], x_old[:, 0:2]
    train, test = sample_index(seed_index, y_new)
    y_train, y_test = y_new[train], y_new[test]
    x_train, x_test = x_new[train], x_new[test]
    g_train = [g_[train] for g_ in g_new]
    g_test = [g_[test] for g_ in g_new]

    criterion = nn.CrossEntropyLoss()
    y_train, y_test = y_train.to(device), y_test.to(device)
    x_train, x_test = x_train.to(device), x_test.to(device)
    g_train, g_test = [g.to(device) for g in g_train], [g.to(device) for g in g_test]

    torch.manual_seed(629)
    model1 = [MyModelA(g_.shape[1], device) for g_ in g_new]
    model2 = MyModelB(x_new.shape[1], 2, device)
    net = MyEnsemble(model1, model2)
    lamb_hyper = [1e-2, 1e-1, 1e0, 1e1]
    net_result = ann(g_train, x_train, y_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_result(g_train, x_train), y_train).cpu().tolist()
    loss_test = criterion(net_result(g_test, x_test), y_test).cpu().tolist()
    df = pd.DataFrame(data={'method': ["NN"], 
                            'train': [loss_train], 'test': [loss_test]})

    for i in range(5):
        g_old[i] = g_old[i].to(device)

    y_old = y_old.to(device)
    x_old = x_old.to(device)
    net_res = ann(g_old, x_old, y_old, copy.deepcopy(net), lamb_hyper, criterion)
    for i in range(5):
        for param in net_res.modelA[i].parameters():
            param.requires_grad = False

    net_result = ann(g_train, x_train, y_train, net_res, lamb_hyper, criterion)
    loss_train = criterion(net_result(g_train, x_train), y_train).cpu().tolist()
    loss_test = criterion(net_result(g_test, x_test), y_test).cpu().tolist()
    df = df.append(pd.DataFrame(data={'method': ["TL"], 
                                      'train': [loss_train], 'test': [loss_test]}))

    path_output = "../Output_nn/TL5/CLF/"
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + "/" + str(seed_index) + ".csv", index=False)
    print(df)
    return

#!/usr/bin/env python
# coding: utf-8

import copy
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from ann import *
from data import *
from data5 import *
from model5 import *
from sample_r import *


def tl5(seed_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g, x, y = data("CHRNA3", seed_index)
    g = [g]
    for name in ["CHRNA5", "CHRNA6", "CHRNB3", "CHRNB4"]:
        g_ = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
        g.append(torch.from_numpy(g_.values).float())
    
    xx = x[:, 0].numpy()
    old = np.arange(len(xx))[xx.astype(bool)]
    new = np.arange(len(xx))[(1 - xx).astype(bool)]
    y_old, y_new = y[old], y[new]
    g_old, g_new = [g_[old] for g_ in g], [g_[new] for g_ in g]

    #g_new, y_new = data5("ea")
    #g_old, y_old = data5("ukb")
    train, test = sample_index(seed_index, y_new)
    y_train, y_test = y_new[train], y_new[test]
    g_train = [g_[train] for g_ in g_new]
    g_test = [g_[test] for g_ in g_new]

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    mean_train = torch.mean(y_train)
    loss_train = criterion(mean_train * torch.ones(y_train.shape), y_train).tolist()
    loss_test = criterion(mean_train * torch.ones(y_test.shape), y_test).tolist()
    df = pd.DataFrame(data={'method': ["Base"],
                            'train': [loss_train], 'test': [loss_test]})
    y_train, y_test = y_train.to(device), y_test.to(device)
    g_train, g_test = [g.to(device) for g in g_train], [g.to(device) for g in g_test]

    torch.manual_seed(629)
    model1 = []
    for i in range(5):
        model1.append(MyModelA(g_new[i].shape[1], device))
    model2 = MyModelB(device)
    net = MyEnsemble(model1, model2)
    lamb_hyper = [1e-1, 1e0, 1e1, 1e2]
    net_result = ann(g_train, y_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_result(g_train), y_train).cpu().tolist()
    loss_test = criterion(net_result(g_test), y_test).cpu().tolist()
    df = df.append(pd.DataFrame(data={'method': ["NN"], 
                                      'train': [loss_train], 'test': [loss_test]}))

    for i in range(5):
        g_old[i] = g_old[i].to(device)

    y_old = y_old.to(device)
    net_res = ann(g_old, y_old, copy.deepcopy(net), lamb_hyper, criterion)
    for i in range(5):
        for param in net_res.modelA[i].parameters():
            param.requires_grad = False

    net_result = ann(g_train, y_train, net_res, lamb_hyper, criterion)
    loss_train = criterion(net_result(g_train), y_train).cpu().tolist()
    loss_test = criterion(net_result(g_test), y_test).cpu().tolist()
    df = df.append(pd.DataFrame(data={'method': ["TL"], 
                                      'train': [loss_train], 'test': [loss_test]}))

    path_output = "../Output_nn/TL/"
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + "/" + str(seed_index) + ".csv", index=False)
    print(df)
    return

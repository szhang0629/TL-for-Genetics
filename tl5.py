#!/usr/bin/env python
# coding: utf-8

import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from ann import ann
from inn import inn
from base import base
from data import Dataset
from data5 import *
from model5 import *
from auc import *


def tl5(seed_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y, g, x = Dataset("CHRNA3").data
    g = [g]
    for name in ["CHRNA5", "CHRNA6", "CHRNB3", "CHRNB4"]:
        g_ = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
        g.append(torch.from_numpy(g_.values).float())

    x_race_indicator = x[:, 0].numpy()
    old = np.arange(len(x_race_indicator))[x_race_indicator.astype(bool)]
    new = np.arange(len(x_race_indicator))[(1 - x_race_indicator).astype(bool)]

    x = x[:, 1:3]
    y_old, y_new, x_old, x_new = y[old].to(device), y[new].to(device), x[old].to(device), x[new].to(device)
    g_old, g_new = [g_[old].to(device) for g_ in g], [g_[new].to(device) for g_ in g]

    # g_new, x_new, y_new = data5("ea")
    # g_old, x_old, y_old = data5("ukb")
    # x_new, x_old = x_new[:, 0:2], x_old[:, 0:2]
    train, test = SeedSequence(seed_index, y_new.shape[0]).sequence_split
    x_train, y_train = x_new[train], y_new[train]
    x_test, y_test = x_new[test], y_new[test]
    g_train = [g_[train] for g_ in g_new]
    g_test = [g_[test] for g_ in g_new]

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.CrossEntropyLoss()

    df = base(y_train, y_test, criterion)

    torch.manual_seed(629)
    model1 = [MyModelA(g_.shape[1], device) for g_ in g_new]
    model2 = MyModelB(x_new.shape[1], 1, device)
    net = MyEnsemble(model1, model2)
    lamb_hyper = [1e-1]  #, 1e0, 1e1, 1e2]
    data_train = [y_train, g_train, x_train]
    data_test = [y_test, g_test, x_test]
    df = df.append(inn(data_train, data_test, copy.deepcopy(net), lamb_hyper, criterion))

    net_res = ann([y_old, g_old, x_old], copy.deepcopy(net), lamb_hyper, criterion)
    for i in range(5):
        for param in net_res.modelA[i].parameters():
            param.requires_grad = False

    df_ = inn(data_train, data_test, copy.deepcopy(net_res), lamb_hyper, criterion)
    df_.method = "TL"
    df = df.append(df_)

    path_output = "../Output_nn/TL/"
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + "/" + str(seed_index) + ".csv", index=False)
    print(df)
    return


tl5(1)
"""Apply Transfer Learning in Genetics"""
# !/usr/bin/env python
# coding: utf-8

import copy
import os
import random

import pandas as pd

from ml import ml
from data import Dataset
from data2 import *
from models import *
from auc import *


def main(seed_index, name="CHRNA5", classification=False):
    device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(name, classification, device)
    x_race_indicator = dataset.x[:, 0].cpu().numpy()
    dataset.x = dataset.x[:, 1:3]
    old = np.arange(len(x_race_indicator))[x_race_indicator.astype(bool)]
    new = np.arange(len(x_race_indicator))[(1-x_race_indicator).astype(bool)]
    dataset_new, dataset_old = dataset.split([new, old])

    train, test = SeedSequence(seed_index, dataset.y.shape[0]).sequence_split
    dataset_train, dataset_test = dataset.split([train, test])

    path_output = "../Output_nn/TL/SAGE/" + name
    lamb_hyper = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # g_new, y_new = data2(name, "ea", classification)
    # g_old, y_old = data2(name, "ukb", classification)
    # g_old, y_old = g_old.to(device), y_old.to(device)
    # train, test = sample_index(seed_index, g_new)
    # g_train, y_train = g_new[train].to(device), y_new[train].to(device)
    # g_test, y_test = g_new[test].to(device), y_new[test].to(device)
    # x_old, x_train, x_test = None, None, None
    # path_output = "../Output_nn/TL/UKB/" + name

    if classification:
        path_output += "/CLS/"
        criterion = nn.CrossEntropyLoss()
        # df = pd.DataFrame(columns=['method', 'train', 'test'])
        df = my_log(dataset_train, dataset_test)
        torch.manual_seed(629)
        net = Net(dataset.g.shape[1], 2).to(device)
        # nets = Nets(dataset.g.shape[1], 2).to(device)
    else:
        path_output += "/PRD/"
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        df = base(dataset_train.y, dataset_test.y, criterion)
        df = df.append(my_lasso(dataset_train, dataset_test, lamb_hyper, criterion))
        torch.manual_seed(629)
        net = Net(dataset.g.shape[1], 1).to(device)
        # nets = Nets(dataset.g.shape[1], 1).to(device)

    df = df.append(inn(dataset, dataset_test, copy.deepcopy(net), lamb_hyper, criterion))
    #
    # net_res = ann(Dataset(data=dataset_old, device=device), copy.deepcopy(net), lamb_hyper, criterion)
    # for param in net_res.modelA.parameters():
    #     param.requires_grad = False
    #
    # df_ = inn(dataset, dataset_test, copy.deepcopy(net_res), lamb_hyper, criterion)
    # df_.method = "TL"
    # df = df.append(df_)

    # ratio_list = [1, 0.75, 0.5, 0.25]
    # for i in range(len(ratio_list)):
    #     net_res = ml(data_train, data_old, copy.deepcopy(nets), lamb_hyper, criterion, ratio_list[i])
    #     loss_train = criterion(net_res(g_train), y_train).tolist()
    #     loss_test = criterion(net_res(g_test), y_test).tolist()
    #     df = df.append(pd.DataFrame(data={'method': ["ML"+str(i)], 'train': [loss_train], 'test': [loss_test]}))

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(df)
    return

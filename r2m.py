# !/usr/bin/env python
# coding: utf-8

import copy
import os
import sys

import pandas as pd
import torch.nn as nn

from auc import *
from data import Dataset, Dataset2
from models import MyEnsemble
from models.model_4 import ModelA, ModelB, ModelC


def r2m(seed_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_old = Dataset2(device, "rat")
    dataset = Dataset2(device, "mice")
    path_output = "../Output_nn/R2M/"

    train, test = SeedSequence(seed_index, dataset.y.shape[0]).sequence_split
    dataset_train, dataset_test = dataset.split([train, test])
    lamb_hyper = [1e-1, 1e0, 1e1, 1e2, 1e3]

    criterion = nn.MSELoss()
    df = base(dataset_train.y, dataset_test.y, criterion)
    # df = df.append(my_lasso(dataset_train, dataset_test, [1e-3, 1e-2, 1e-1, 1e0], criterion))
    torch.manual_seed(seed_index)
    net = MyEnsemble(ModelA(dataset.x), ModelB(), ModelC(1)).to(device)

    net_result = ann(dataset_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_result(dataset_train.x, dataset_train.z), dataset_train.y).tolist()
    loss_test = criterion(net_result(dataset_test.x, dataset_test.z), dataset_test.y).tolist()
    df = df.append(pd.DataFrame(data={'method': ["NN"], 'train': [loss_train], 'test': [loss_test]}))

    # net_old = MyEnsemble(ModelA(dataset_old.x), ModelB(), ModelC(1, dataset.z.shape[1])).to(device)
    # net_res = ann(dataset_old, copy.deepcopy(net_old), lamb_hyper, criterion)
    # net_tf = MyEnsemble(net_result.ModelA, net_res.ModelB, net_result.ModelC).to(device)
    # for param in net_tf.ModelB.parameters():
    #     param.requires_grad = False
    #
    # df_ = inn(dataset_train, dataset_test, copy.deepcopy(net_tf), lamb_hyper, criterion)
    # df_.method = "TL"
    # df = df.append(df_)

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(df)
    return

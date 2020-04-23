# !/usr/bin/env python
# coding: utf-8

import copy
import os

import pandas as pd
import torch.nn as nn

from auc import *
from data import Dataset
from models.model_8 import Model1, Model2


def real(seed_index, name="CHRNA5", classification=False):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    dataset = Dataset(name, classification, device)
    path_output = "../Output_nn/Real/"

    train, test = SeedSequence(seed_index, dataset.y.shape[0]).sequence_split
    dataset_train, dataset_test = dataset.split([train, test])
    lamb_hyper = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    criterion = nn.MSELoss()
    df = base(dataset_train.y, dataset_test.y, criterion)
    input_dim = dataset.x.shape[1] + dataset.z.shape[1]
    torch.manual_seed(seed_index)

    net = Model1(input_dim)
    df_ = inn(dataset_train, dataset_test, copy.deepcopy(net), lamb_hyper, criterion)
    df_.method = "NN1"
    df = df.append(df_)

    net = Model2(input_dim)
    df_ = inn(dataset_train, dataset_test, copy.deepcopy(net), lamb_hyper, criterion)
    df_.method = "NN2"
    df = df.append(df_)

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(df)
    return

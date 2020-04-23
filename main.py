"""Apply Transfer Learning in Genetics"""
# !/usr/bin/env python
# coding: utf-8

import copy
import os
import sys

import pandas as pd
import torch.nn as nn

from auc import *
from data import Dataset
from models import MyEnsemble
from models.model_4 import ModelA, ModelC


def main(seed_index, name="CHRNA5", classification=False, name_data="ukb"):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    if name_data == "ukb":
        dataset = Dataset(name, classification, device, name_data="ea")
        dataset_old = Dataset(name, classification, device, name_data="ukb")
        path_output = "../Output_nn/UKB/"
    else:
        dataset_all = Dataset(name, classification, device)
        race_indicator = dataset_all.z[:, 0].cpu().numpy()
        dataset_all.z = dataset_all.z[:, 1:3]
        old = np.arange(len(race_indicator))[race_indicator.astype(bool)]
        new = np.arange(len(race_indicator))[(1-race_indicator).astype(bool)]
        dataset, dataset_old = dataset_all.split([new, old])
        path_output = "../Output_nn/SAGE/"

    train, test = SeedSequence(seed_index, dataset.y.shape[0]).sequence_split
    dataset_train, dataset_test = dataset.split([train, test])
    lamb_hyper = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    if classification:
        path_output += "/CLS/" + name + "/"
        criterion = nn.CrossEntropyLoss()
        # df = pd.DataFrame(columns=['method', 'train', 'test'])
        # df = my_log(dataset_train, dataset_test)
        df = base(dataset_train.y, dataset_test.y, criterion, classification=True)
        torch.manual_seed(seed_index)
        if dataset.z is None:
            # net = Net(dataset.g.shape[1], 2).to(device)
            net = MyEnsemble(ModelA(dataset.x), ModelC(2)).to(device)
        else:
            # net = Net(dataset.g.shape[1], 2, dataset.x.shape[1]).to(device)
            net = MyEnsemble(ModelA(dataset.x), ModelC(2, dataset.z.shape[1])).to(device)
    else:
        path_output += "/PRD/" + name + "/"
        criterion = nn.MSELoss()
        df = base(dataset_train.y, dataset_test.y, criterion)
        # df = df.append(my_lasso(dataset_train, dataset_test, [1e-3, 1e-2, 1e-1, 1e0], criterion))
        torch.manual_seed(seed_index)
        if dataset.x is None:
            net = MyEnsemble(ModelA(dataset.x.shape[1]), ModelC(1)).to(device)
        else:
            net = MyEnsemble(ModelA(dataset.x), ModelC(1, dataset.z.shape[1])).to(device)

    df = df.append(inn(dataset_train, dataset_test, copy.deepcopy(net), lamb_hyper, criterion))
    net_res = ann(dataset_old, copy.deepcopy(net), lamb_hyper, criterion)
    for param in net_res.ModelA.parameters():
        param.requires_grad = False

    # if type(name) is list:
    #     path_model = ("clf_" if classification else "prd_") + name_data+"_tf.md"
    # else:
    #     path_model = ("clf_" if classification else "prd_") + name_data+"_tf_"+name+".md"
    # if os.path.exists(path_model):
    #     net_res = torch.load(path_model)
    #     net_res.eval()
    # else:
    #     net_res = ann(dataset_old, copy.deepcopy(net), lamb_hyper, criterion)
    #     for param in net_res.ModelA.parameters():
    #         param.requires_grad = False
    #     torch.save(net_res, path_model)

    df_ = inn(dataset_train, dataset_test, copy.deepcopy(net_res), lamb_hyper, criterion)
    df_.method = "TL"
    df = df.append(df_)

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(df)
    return

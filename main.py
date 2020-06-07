# !/usr/bin/env python
# coding: utf-8
import os
import torch
import numpy as np
import pandas as pd

from data import Dataset1
from models import ModelA, ModelC, MyEnsemble


def main(seed_index, name="CHRNA5", name_data="sage", hidden_units=4):
    if name is None:
        name = ["CHRNA5", "CHRNA3", "CHRNA6", "CHRNB3", "CHRNB4"]
    classification = False
    out_dim = 1 + classification
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name_data == "ukb":
        dataset = Dataset1(name, name_data="ea")
        dataset.to(device)
        dataset.process(classification)
        dataset.z = None
        dataset_old = Dataset1(name, name_data="ukb")
        dataset_old.to(device)
        dataset_old.process(classification)
        dataset_old.z = None
        path_output = "../Output_nn/UKB/"
    else:
        dataset = Dataset1(name, name_data="aa")
        dataset.to(device)
        dataset.process(classification)
        dataset.z = None
        dataset_old = Dataset1(name, name_data="ea")
        dataset_old.to(device)
        dataset_old.process(classification)
        dataset_old.z = None
        path_output = "../Output_nn/SAGE/"

    dataset_train, dataset_test = dataset.split_seed(seed_index)
    lamb_hyper = 10. ** np.arange(-2, 3)
    path_output += ("All" if type(name) is list else name) + "/"
    in_dim = [x_.shape[1] for x_ in dataset.x] if type(dataset.x) is list else dataset.x.shape[1]

    if not os.path.exists(path_output + str(seed_index) + ".csv"):
        os.makedirs(path_output, exist_ok=True)
    y_base = torch.mean(dataset_train.y)
    res = pd.DataFrame(data={'method': ["Base"], 'pen': [1.0],
                             'train': [dataset_train.base(y_base)], 'test': [dataset_test.base(y_base)]})
    torch.manual_seed(seed_index)
    net = MyEnsemble(ModelC(in_dim, out_dim)).to(device)
    net_lm, lamb = net.ann(dataset_train, lamb_hyper)
    res = res.append(pd.DataFrame(data={'method': ["LM"], 'pen': [lamb],
                                        'train': [dataset_train.loss(net_lm)], 'test': [dataset_test.loss(net_lm)]}))
    torch.manual_seed(seed_index)
    model_a = ModelA(in_dim, hidden_units)
    model_c = ModelC([hidden_units] * len(dataset.x) if type(dataset.x) is list else hidden_units, out_dim)
    net = MyEnsemble(model_a, model_c).to(device)
    net_nn, lamb = net.ann(dataset_train, lamb_hyper)
    res = res.append(pd.DataFrame(data={'method': ["NN"], 'pen': [lamb],
                                        'train': [dataset_train.loss(net_nn)], 'test': [dataset_test.loss(net_nn)]}))
    net_ml, lamb = net.ann(dataset_train, lamb_hyper, dataset_old)
    res = res.append(pd.DataFrame(data={'method': ["ML"], 'pen': [lamb],
                                        'train': [dataset_train.loss(net_ml)], 'test': [dataset_test.loss(net_ml)]}))
    torch.manual_seed(seed_index)
    net_old, lamb = net.ann(dataset_old, lamb_hyper)
    for param in net_old.model0.parameters():
        param.requires_grad = False
    net_tl, lamb = net_old.ann(dataset_train, lamb_hyper)
    res = res.append(pd.DataFrame(data={'method': ["TL"], 'pen': [lamb],
                                        'train': [dataset_train.loss(net_tl)], 'test': [dataset_test.loss(net_tl)]}))
    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

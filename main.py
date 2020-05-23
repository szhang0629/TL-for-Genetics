# !/usr/bin/env python
# coding: utf-8
import os
import torch
import numpy as np
import pandas as pd

from data import Dataset1
from models import ModelA, ModelC, MyEnsemble


def main(seed_index, name="CHRNA5", name_data="sage", hidden_units=2):
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
        path_output = "../Output_nn/UKB/"

    dataset_train, dataset_test = dataset.split_seed(seed_index)
    lamb_hyper = 10. ** np.arange(-1, 4)
    path_output += "/CLS/" if classification else "/PRD/" + ("" if type(name) is list else name)
    path_output += "/"

    if not os.path.exists(path_output + str(seed_index) + ".csv"):
        os.makedirs(path_output, exist_ok=True)
        y_base = torch.mean(dataset_train.y)
        df = pd.DataFrame(data={'method': ["Base"], 'pen': [0],
                                'train': [dataset_train.base(y_base)], 'test': [dataset_test.base(y_base)]})
        torch.manual_seed(seed_index)
        net = MyEnsemble(ModelC(sum([x_.shape[1] for x_ in dataset_train.x]) if type(dataset_train.x) is list
                                else dataset_train.x.shape[1], out_dim)).to(device)
        net_lm, lamb = net.ann(dataset_train, lamb_hyper)
        df = df.append(pd.DataFrame(data={'method': ["LM"], 'pen': [lamb],
                                          'train': [dataset_train.loss(net_lm)], 'test': [dataset_test.loss(net_lm)]}))
        df.to_csv(path_output + str(seed_index) + ".csv", index=False)
        print(df)

    path_output += str(hidden_units) + "/"
    torch.manual_seed(seed_index)
    net = MyEnsemble(ModelA(dataset.x, hidden_units), ModelC(hidden_units * (len(dataset.x) if type(dataset.x) is list
                                                                             else 1), out_dim)).to(device)
    net_nn, lamb = net.ann(dataset_train, lamb_hyper)
    df = pd.DataFrame(data={'method': ["NN"], 'pen': [lamb],
                            'train': [dataset_train.loss(net_nn)], 'test': [dataset_test.loss(net_nn)]})

    net_ml, lamb = net.ann(dataset_train, lamb_hyper, dataset_old)
    df = df.append(pd.DataFrame(data={'method': ["ML"], 'pen': [lamb],
                                      'train': [dataset_train.loss(net_ml)], 'test': [dataset_test.loss(net_ml)]}))

    # folder_model = "../Models/" + ("clf/" if classification else "prd/") + name_data + \
    #                ("/" if type(name) is list else "/" + name + "/")
    # model_name = str(hidden_units) + ".md"
    # if os.path.exists(folder_model + model_name):
    #     net_old = torch.load(folder_model + model_name)
    #     net_old.eval()
    # else:
    #     torch.manual_seed(0)
    #     net_old, lamb = net.ann(dataset_old, lamb_hyper)
    #     for param in net_old.model0.parameters():
    #         param.requires_grad = False
    #     os.makedirs(folder_model, exist_ok=True)
    #     torch.save(net_old, folder_model + model_name)
    #
    # net_tl, lamb = net_old.ann(dataset_train, lamb_hyper)
    # df = df.append(pd.DataFrame(data={'method': ["TL"], 'pen': [lamb],
    #                                   'train': [dataset_train.loss(net_tl)], 'test': [dataset_test.loss(net_tl)]}))
    os.makedirs(path_output, exist_ok=True)
    df.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(df)
    return

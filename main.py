# !/usr/bin/env python
# coding: utf-8
import os

import torch.nn as nn

from auc import *
from data import Dataset0, Dataset1
from models import MyEnsemble
from models.model_4 import ModelA, ModelC


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
        dataset_all = Dataset0(name)
        dataset_all.to(device)
        dataset_all.process(classification)
        race_indicator = dataset_all.z[:, 0].cpu().numpy()
        # dataset_all.z = dataset_all.z[:, 1:3]
        dataset_all.z = None
        old = np.arange(len(race_indicator))[race_indicator.astype(bool)]
        new = np.arange(len(race_indicator))[(1-race_indicator).astype(bool)]
        dataset, dataset_old = dataset_all.split([new, old])
        path_output = "../Output_nn/SAGE/"

    train, test = SeedSequence(seed_index, dataset.y.shape[0]).sequence_split
    dataset_train, dataset_test = dataset.split([train, test])
    lamb_hyper = [1e-1, 1e0, 1e1, 1e2, 1e3]
    path_output += "/CLS/" if classification else "/PRD/" + ("" if type(name) is list else name)
    path_output += "/"
    criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()

    if not os.path.exists(path_output + str(seed_index) + ".csv"):
        os.makedirs(path_output, exist_ok=True)
        df = base(dataset_train.y, dataset_test.y, criterion, classification)
        torch.manual_seed(seed_index)
        net = MyEnsemble(ModelC(sum([x_.shape[1] for x_ in dataset_train.x]) if type(dataset_train.x) is list
                                else dataset_train.x.shape[1], out_dim)).to(device)
        net_lm, lamb = ann(dataset_train, copy.deepcopy(net), lamb_hyper, criterion)
        loss_train = criterion(net_lm(dataset_train), dataset_train.y).tolist()
        loss_test = criterion(net_lm(dataset_test), dataset_test.y).tolist()
        df = df.append(pd.DataFrame(data={'method': ["LM"], 'pen': [lamb], 'train': [loss_train], 'test': [loss_test]}))
        df.to_csv(path_output + str(seed_index) + ".csv", index=False)
        print(df)

    path_output += str(hidden_units) + "/"
    torch.manual_seed(seed_index)
    net = MyEnsemble(ModelA(dataset.x, hidden_units), ModelC(hidden_units * (len(dataset.x) if type(dataset.x) is list
                                                                             else 1), out_dim)).to(device)
    net_nn, lamb = ann(dataset_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_nn(dataset_train), dataset_train.y).tolist()
    loss_test = criterion(net_nn(dataset_test), dataset_test.y).tolist()
    df = pd.DataFrame(data={'method': ["NN"], 'pen': [lamb], 'train': [loss_train], 'test': [loss_test]})

    folder_model = "../Models/" + ("clf/" if classification else "prd/") + name_data + \
                   ("/" if type(name) is list else "/" + name + "/")
    model_name = str(hidden_units) + ".md"
    if os.path.exists(folder_model + model_name):
        net_old = torch.load(folder_model + model_name)
        net_old.eval()
    else:
        torch.manual_seed(0)
        net_old, lamb = ann(dataset_old, copy.deepcopy(net), lamb_hyper, criterion)
        for param in net_old.model0.parameters():
            param.requires_grad = False
        os.makedirs(folder_model, exist_ok=True)
        torch.save(net_old, folder_model + model_name)

    net_tl, lamb = ann(dataset_train, copy.deepcopy(net_old), lamb_hyper, criterion)
    loss_train = criterion(net_tl(dataset_train), dataset_train.y).tolist()
    loss_test = criterion(net_tl(dataset_test), dataset_test.y).tolist()
    df = df.append(pd.DataFrame(data={'method': ["TL"], 'pen': [lamb], 'train': [loss_train], 'test': [loss_test]}))
    os.makedirs(path_output, exist_ok=True)
    df.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(df)
    return

#
# for num in [4, 8, 16, 32, 64]:
#     main(1, hidden_units=num)

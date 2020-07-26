# !/usr/bin/env python
# coding: utf-8
import os

from dataset import Data1, Data3
from dnn import DNN, Layer
from fdnn import FDNN, MyModelB
from solution import Base, Ridge


def main(seed_index, gene="CHRNA6"):
    # data, race = "sage", None
    data, race = "ukb", 4
    # data, race = "mice", 0
    hidden_units = [17]
    str_units = str("_".join(str(x) for x in hidden_units))
    if data == "ukb":
        dataset = Data1(gene, data=data, race=race)
        race = {1002: "irish", 4: "black"}[race]
        filename = os.path.join("..", "Output", data, race, gene, str_units, "")
    else:
        dataset = Data1(gene, data=data, race=race)
        # dataset = Data3(gene, data=data)
        filename = os.path.join("..", "Output", data, gene, str_units, "")
    filename += str(seed_index) + ".csv"
    dataset.x = dataset.x * dataset.scale_ratio
    trainset, testset = dataset.split_seed(seed_index)
    device = dataset.x.device

    bl = Base(trainset)
    bl.to_csv(testset, filename)

    # rd = Ridge(trainset)
    # rd.to_csv(testset, filename)
    #
    # dims = [dataset.x.shape[1]] + hidden_units + [1]
    # net = DNN([Layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    # net = net.to(device)
    #
    # net_nn = net.hyper_train(trainset)
    # net_nn.to_csv(testset, filename)
    #
    # name_pre = data + "_" + str_units + ".md"
    # net_pre = net.pre_train(name_pre, gene, dataset)
    # net_tl = net_pre.hyper_train(trainset, [10 ** x for x in range(-4, 2)])
    # net_tl.to_csv(testset, filename, "TL")

    # trainset_tl, testset_tl = \
    #     net_pre.transfer(trainset), net_pre.transfer(testset)
    # lm = Ridge(trainset_tl)
    # lm.to_df(testset_tl, filename, "TRD")

    dims = [hidden_units[0]] + hidden_units + [hidden_units[-1]]
    model_flm = FDNN([MyModelB(dims[0], dims[-1])]).to(device)
    net_flm = model_flm.hyper_train(trainset)
    net_flm.to_csv(testset, filename, "FLM")

    net = FDNN([MyModelB(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    net = net.to(device)
    net_fnn = net.hyper_train(trainset)
    net_fnn.to_csv(testset, filename)
    #
    # name_pre = data + "_" + str_units + ".md"
    # net_pre = net.pre_train(name_pre, gene, dataset)
    # net_tl = net_pre.hyper_train(trainset)
    # net_tl.to_csv(testset, filename, "FTL")

    return

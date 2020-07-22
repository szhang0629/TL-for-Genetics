# !/usr/bin/env python
# coding: utf-8
import os

import torch

from data import DiscreteData1, DiscreteData2
from dnn import DNN, Layer
from solution import Linear


def main(seed_index, gene="CHRNA6", data="sage", race=1002, hidden_units=[4]):
    # torch.set_default_tensor_type(torch.DoubleTensor)
    if data == "ukb":
        dataset = DiscreteData1(gene, data=data, race=race)
        race = {1002: "irish", 4: "black"}[race]
        path_output = os.path.join("..", "Output", "nn", data, race, gene, "")
    else:
        dataset = DiscreteData1(gene, data=data)
        path_output = os.path.join("..", "Output", "nn", data, gene, "")
    # dataset = DiscreteData2(gene, data="sage")
    # path_output = os.path.join("..", "Output", "nn", "sage", gene, "")

    trainset, testset = dataset.split_seed(seed_index)
    if hidden_units is None:
        hidden_units = [dataset.x.shape[1]]
    dims = [dataset.x.shape[1]] + hidden_units + [1 + dataset.classification]
    bl = Linear(trainset, True)
    res = bl.to_df(testset, "Base")
    rd = Linear(trainset)
    res = res.append(rd.to_df(testset, "RD"))

    net = DNN([Layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    net = net.to(dataset.y.device)
    net_nn = net.hyper_train(trainset, [10 ** (x / 2) for x in range(-7, 3)])
    res = res.append(net_nn.to_df(testset, "NN"))

    # name_pre = gene + "_" + str("_".join(str(x) for x in hidden_units)) + ".md"
    # net_pre = net.pre_train(name_pre, gene)
    # net_tl = net_pre.hyper_train(trainset,
    #                              [10 ** (x / 2) for x in range(-8, 2)])
    # res = res.append(net_tl.to_df(testset, "TL"))
    # trainset_tl, testset_tl = net_pre.transfer(trainset), \
    #                           net_pre.transfer(testset)
    # lm = Linear(trainset_tl)
    # res = res.append(lm.to_df(testset_tl, "LM"))
    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

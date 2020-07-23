#!/usr/bin/env python
# coding: utf-8
import os

from data import Data1, Data2
from fdnn import FDNN, MyModelB
from solution import Linear


def main(seed_index, gene="CHRNA3", data="ukb", race=1002, hidden_units=[9]):
    str_units = str("_".join(str(x) for x in hidden_units))
    if data == "ukb":
        dataset = Data1(gene, data=data, race=race, functional=True)
        race = {1002: "irish", 4: "black"}[race]
        path_output = os.path.join("..", "Output", "FDNN",
                                   data, race, gene, str_units, "")
    else:
        dataset = Data1(gene, data=data, functional=True)
        path_output = os.path.join("..", "Output", "FDNN",
                                   data, gene, str_units, "")
    # dataset = Data2(gene, data, functional=True)
    # path_output = os.path.join("..", "Output", "FDNN",
    #                            "sage", gene, str_units, "")
    device = dataset.x.device
    lamb_hyper = [10**x for x in range(-3, 0)]
    trainset, testset = dataset.split_seed(seed_index)
    dims = [hidden_units[0]] + hidden_units + [hidden_units[-1]]
    bl = Linear(trainset, True)
    res = bl.to_df(testset, "Base")
    # TODO: new LM model without extra parameters
    model_flm = FDNN([MyModelB(dims[0], dims[-1])]).to(device)
    net_flm = model_flm.hyper_train(trainset, lamb_hyper)
    res = res.append(net_flm.to_df(testset, "FLM"))

    net = FDNN([MyModelB(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    net = net.to(dataset.y.device)
    net_fnn = net.hyper_train(trainset, lamb_hyper)
    res = res.append(net_fnn.to_df(testset, "FDNN"))

    # name_pre = data + "_" + str_units + ".md"
    # net_pre = net.pre_train(name_pre, gene, dataset)
    # net_tl = net_pre.hyper_train(trainset, [lamb/10 for lamb in lamb_hyper])
    # res = res.append(net_tl.to_df(testset, "TL"))

    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

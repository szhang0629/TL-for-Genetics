#!/usr/bin/env python
# coding: utf-8
import os

from data import Data3
from fdnn import FDNN, MyModelB
from solution import Linear


def main(seed_index, gene="0", hidden_units=[9]):
    str_units = str("_".join(str(x) for x in hidden_units))
    dataset = Data3(gene)
    path_output = os.path.join("..", "Output", "FDNN", gene, str_units, "")
    device = dataset.x.device
    lamb_hyper = [10**x for x in range(-1, 2)]
    trainset, testset = dataset.split_seed(seed_index)
    dims = [hidden_units[0]] + hidden_units + [hidden_units[-1]]
    bl = Linear(trainset, True)
    res = bl.to_df(testset, "Base")
    model_flm = FDNN([MyModelB(dims[0], dims[-1])]).to(device)
    net_flm = model_flm.hyper_train(trainset,
                                    [10**(x/2) for x in range(-5, -2)])
    res = res.append(net_flm.to_df(testset, "FLM"))

    net = FDNN([MyModelB(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    net = net.to(device)
    net_fnn = net.hyper_train(trainset, lamb_hyper)
    res = res.append(net_fnn.to_df(testset, "FNN"))

    name_pre = str_units + ".md"
    net_pre = net.pre_train(name_pre, gene, dataset)
    net_tl = net_pre.hyper_train(trainset, [lamb/10 for lamb in lamb_hyper])
    res = res.append(net_tl.to_df(testset, "TL"))

    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

# !/usr/bin/env python
# coding: utf-8
from simulation import Simulation
from dataset import Data1, Data2
from dnn import DNN
from fdnn import FDNN
from solution import Base, Ridge


def main(seed_index, gene="CHRNA6"):
    data, race, hidden = "sage", None, [8, 4]
    # data, race, hidden = "ukb", 4, [16, 4]
    # data, race, hidden = "mice", None, [128]
    if gene is None:
        dataset = Simulation(seed_index)
    elif data == "ukb":
        dataset = Data1(gene, data=data, race=race)
    elif data == "mice":
        dataset = Data2(gene, data=data)
    else:
        dataset = Data1(gene, data=data, race=race)
    dataset.split_seed(seed_index)

    base = Base()
    ridge = Ridge()
    net = DNN([dataset.x.shape[1]] + hidden + [1])
    # fbase = FDNN([64])
    # model_flm = FDNN([hidden[0]] + [hidden[-1]])
    # model_flm = FDNN([64] + [64])
    # net_f = FDNN([64] + hidden + [64])
    # net_f = FDNN([64] + hidden + [1])

    bl = dataset.tuning(base)
    rd = dataset.tuning(ridge)
    net_dnn = dataset.tuning(net, [10 ** x for x in range(-4, 0)])
    # net_bl = dataset.tuning(fbase,  [10**x for x in range(-3, 1)], "FBS")
    # net_flm = dataset.tuning(model_flm, [10**x for x in range(-3, 1)], "FLM")
    # net_fdnn = dataset.tuning(net_f, [10 ** x for x in range(-2, 2)])

    net_ = net.pre_train(dataset, [10 ** x for x in range(-6, -3)])
    net_tl = dataset.tuning(net_, [10 ** x for x in range(-4, 0)], "TL")
    # net_f = net_f.pre_train(dataset, [10 ** x for x in range(-3, 1)])
    # net_ftl = dataset.tuning(net_f, [10 ** x for x in range(-2, 2)], "FTL")

    return

# !/usr/bin/env python
# coding: utf-8
from simulation import Simulation
from dataset import Data1, Data2
# from nn import NN
from fnn import FNN
from solution import Base, Ridge


def main(seed_index, gene="CHRNA5"):
    # # data, race = "sage", None
    data, race = "ukb", 1002
    # data, race, hidden = "mice", None
    if data == "ukb":
        dataset = Data1(gene, data=data, race=race)
    elif data == "mice":
        dataset = Data2(gene, data=data)
    else:
        dataset = Data1(gene, data=data, race=race)
    # dataset = Simulation(seed_index)
    dataset.split_seed(seed_index)

    base = Base()
    # ridge = Ridge([1e2 * (3**x)  for x in range(-2, 3)])
    # net = NN([dataset.x.shape[1]] + [4] + [1],
    # [1e0 * (3 ** x) for x in range(-2, 3)])
    # net_ = net.pre_train(dataset, [1e-1 * (3 ** x) for x in range(-1, 2)])
    # net_.method_name = "TL"
    fnet_0 = FNN([32])
    fnet_0.method_name = "FBase"
    # fnet_1 = FNN([64] + [64], [1e0 * (3 ** x) for x in range(-1, 3)])
    # fnet_1.method_name = "FLM"
    fnet_2 = FNN([128] + [4] + [32],
                 [1e-4 * (3 ** x) for x in range(-2, 3)])
    # fnet_tl = fnet_2.pre_train(dataset,
    # [1e-1 * (3 ** x) for x in range(-2, 3)])
    # fnet_tl.method_name = "FTL"

    methods = [dataset.tuning(method) for method in [base, fnet_0, fnet_2]]

    return methods

# !/usr/bin/env python
# coding: utf-8
from simulation import Simulation
from dataset import Data1, Data2
from nn import NN
from fnn import FNN
from solution import Base, Ridge


def main(seed_index, gene="CHRNB3"):
    # data, race = "sage", None
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

    # ridge = Ridge([1e2 * (3**x) for x in range(-2, 3)])
    hidden1 = [dataset.x.shape[1]] + [16, 1] + [1]
    # hidden2 = [dataset.x.shape[1]] + [16, 4] + [1]
    # net = NN(hidden1, [(2 ** (x)) for x in range(-6, 2)])
    # # net_2 = dataset.tuning(net)
    # net_ = net.pre_train(dataset, [(2 ** (x)) for x in range(-6, 2)])
    # net_.method_name = "TL"
    # fnet_0 = FNN([32])
    # fnet_1 = FNN([64] + [64], [1e0 * (3 ** x) for x in range(-1, 3)])
    # fnet_1.method_name = "FLM"
    # layer0 = [32]
    # layer1 = [128] + [32]
    # layer2 = [128] + [8] + [32]

    # base = dataset.tuning(Base())
    # fnet_0 = dataset.tuning(FNN([32], [1e2 * (3 ** x) for x in range(-2, 3)]))
    # fnet_0 = dataset.tuning(FNN([32], [1e-1]))
    # fnet_2 = FNN(layer2, [1e3 * (10 ** x) for x in range(0, 5)])
    # fnet_tl = fnet_2.pre_train(dataset, [1e3])
    methods = [dataset.tuning(method) for method in
               [Base(),
                NN(hidden1, [(2 ** (x/2)) for x in range(-2, 4)])]]
                # NN(hidden2, [1e-1 * (10 ** x) for x in range(0, 5)])]]
                # FNN(layer0, [1e-2 * (10 ** x) for x in range(0, 5)]),
                # FNN(layer1, [1e2]),
                # FNN(layer2, [1e-1]),
                # FNN(layer2, [1e1]),
                # FNN(layer2, [1e3]),
                # fnet_tl]]

    # dataset_pre = Data1(gene, data="ukb", race=1001)
    # x_pre = dataset_pre.x - dataset_pre.x.mean(0)
    # y_pre = dataset_pre.y - dataset_pre.y.mean()
    # top = y_pre.T @ x_pre
    # below = x_pre.T @ x_pre
    # beta = top / below.diag()
    # beta = beta.abs()
    # beta = beta/beta.std()
    #
    # dataset.x = dataset.x * beta
    # dataset.split_seed(seed_index)
    #
    # hidden1 = [dataset.x.shape[1]] + [4] + [1]
    # net = NN(hidden1, [(10 ** (x/2)) for x in range(-4, 1)])
    # net.method_name = "TL"
    #
    # methods += [dataset.tuning(net)]

    return

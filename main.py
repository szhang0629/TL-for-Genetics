# !/usr/bin/env python
# coding: utf-8
# from simulation import Simulation
from dataset import Data1, Data2
from nn import NN
from solution import Base, Ridge
import torch


def main(seed_index, gene="CHRNA6"):
    torch.set_printoptions(precision=8)
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
    print(dataset.x.shape)
    train, test = dataset.split_seed(seed_index, split_ratio=0.8)
    print((train.y == 0).sum(), train.x.shape[0],
          (test.y == 0).sum(), test.x.shape[0])

    base = Base()
    # ridge = Ridge([1e2 * (3**x) for x in range(-2, 3)])
    hidden1 = [dataset.x.shape[1]] + [64, 16, 4] + [1]
    net = NN(hidden1, [(10 ** (x/2)) for x in range(6, 5, -1)])
    net_ = net.pre_train(dataset, 100)
    net_.method_name = "TL"

    methods = [train.hyper_train(method) for method in
               [base,
                net_,
                NN(hidden1, [(10 ** (x/2)) for x in range(6, 5, -1)])]]
    for model in methods:
        model.to_csv(test)
    return methods

    # base.criterion = CrossEntropyLoss()
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

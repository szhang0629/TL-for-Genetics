# !/usr/bin/env python
# coding: utf-8
import os
import torch

from data import DiscreteData1, DiscreteData2
from dnn import Layer, DNN
from solution import Linear


def main(seed_index, gene="CHRNA5", data="sage", race=1002, hidden_units=None):
    torch.set_default_tensor_type(torch.DoubleTensor)
    # if data == "ukb":
    #     dataset = DiscreteData1(gene, data=data, race=race)
    #     race = {1002: "irish", 4: "black"}[race]
    #     path_output = os.path.join("..", "Output", "nn", data, race, gene, "")
    # else:
    #     dataset = DiscreteData1(gene, data=data)
    #     path_output = os.path.join("..", "Output", "nn", data, gene, "")
    dataset = DiscreteData2(gene, data="sage")
    path_output = os.path.join("..", "Output", "nn", "sage", gene, "")

    trainset, testset = dataset.split_seed(seed_index)
    if hidden_units is None:
        hidden_units = [dataset.x.shape[1]]
    dims = [dataset.x.shape[1]] + hidden_units + [1 + dataset.classification]
    bl = Linear(trainset, True)
    res = bl.to_df(testset, "Base")
    rd = Linear(trainset, lamb=[10 ** x for x in range(-5, 5)])
    res = res.append(rd.to_df(testset, "RD"))
    net = DNN([Layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    net = net.to(dataset.y.device)
    net_nn = net.hyper_train(trainset, [10 ** (x / 2) for x in range(-7, 3)])
    res = res.append(net_nn.to_df(testset, "NN"))
    dir_pre = os.path.join("..", "Models", "nn", "ukb", "")
    name_pre = gene + "_" + str("_".join(str(x) for x in hidden_units)) + ".md"
    if os.path.exists(dir_pre + name_pre):
        net_pre = torch.load(dir_pre + name_pre, map_location=dataset.y.device)
        net_pre.eval()
    else:
        # oldset = DiscreteData1(gene, data="ukb", race=1001, target=data)
        oldset = DiscreteData2(gene, data="ukb")
        net_pre = net.hyper_train(oldset,
                                  [10 ** (x / 2) for x in range(-8, -4)])
        net_pre.save(dir_pre, name_pre, 1)

    net_tl = net_pre.hyper_train(trainset,
                                 [10 ** (x / 2) for x in range(-8, 2)])
    res = res.append(net_tl.to_df(testset, "TL"))
    trainset_tl, testset_tl = net_pre.transfer(trainset), \
                              net_pre.transfer(testset)
    lm = Linear(trainset_tl, lamb=[10 ** x for x in range(-5, 5)])
    res = res.append(lm.to_df(testset_tl, "LM"))
    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

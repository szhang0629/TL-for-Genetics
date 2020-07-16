# !/usr/bin/env python
# coding: utf-8
import os
import torch

from data import Dataset1, Dataset2
from net import Layer, MyEnsemble
from linear import Linear


def main(seed_index, gene="CHRNA6", data="sage", race=1002, hidden_units=[4]):
    torch.set_default_tensor_type(torch.DoubleTensor)
    # if data == "ukb":
    #     dataset = Dataset1(gene, data=data, race=race)
    #     race = {1002: "irish", 4: "black"}[race]
    #     path_output = os.path.join("..", "Output", "nn", data, race, gene, "")
    # else:
    #     dataset = Dataset1(gene, data=data)
    #     path_output = os.path.join("..", "Output", "nn", data, gene, "")
    dataset = Dataset2(gene, data="sage")
    path_output = os.path.join("..", "Output", "nn", "sage", gene, "")

    trainset, testset = dataset.split_seed(seed_index)
    hidden_units = [dataset.x.shape[1]]
    dims = [dataset.x.shape[1]] + hidden_units + [1 + dataset.classification]
    res = testset.to_df(Linear(trainset, True), "Base")
    res = res.append(testset.to_df(Linear(trainset,
                                          lamb=[10 ** x for x in range(-10, 0)]), "RD"))
    net = MyEnsemble([Layer(dims[i], dims[i+1]) for i in range(len(dims)-1)])
    net = net.to(dataset.y.device)
    net_nn = net.ann(trainset, [10**(x/2) for x in range(-7, 3)])
    res = res.append(testset.to_df(net_nn, "NN"))
    dir_pre = os.path.join("..", "Models", "nn", data, "")
    name_pre = gene + "_" + str("_".join(str(x) for x in hidden_units)) + ".md"
    if os.path.exists(dir_pre + name_pre):
        net_pre = torch.load(dir_pre + name_pre, map_location=dataset.y.device)
        net_pre.eval()
    else:
        # oldset = Dataset1(gene, data="ukb", race=1001, target=data)
        oldset = Dataset2(gene, data="ukb")
        net_pre = net.ann(oldset, [10**(x/2) for x in range(-8, -4)])
        net_pre.save(dir_pre, name_pre, 1)

    net_tl = net_pre.ann(trainset, [10**(x/2) for x in range(-8, 2)])
    res = res.append(testset.to_df(net_tl, "TL"))
    trainset_tl, testset_tl = net_pre.transfer(trainset), net_pre.transfer(testset)
    res = res.append(testset_tl.to_df(Linear(trainset_tl,
                                             lamb=[10**x for x in range(-10, 0)]), "LM"))
    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

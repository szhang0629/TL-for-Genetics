#!/usr/bin/env python
# coding: utf-8
import os

import torch

from data import FunctionalData2
from fdnn import FDNN, MyModelB
from solution import Linear


def main(seed_index, gene="CHRNB3"):
    # torch.set_default_tensor_type(torch.DoubleTensor)
    dataset = FunctionalData2(gene, data="sage")
    path_output = os.path.join("..", "Output", "fnn", "sage", gene, str(9), "")
    device = dataset.x.device
    lamb_hyper = [10**x for x in range(-2, 1)]
    trainset, testset = dataset.split_seed(seed_index)
    bl = Linear(trainset, True)
    res = bl.to_df(testset, "Base")
    # TODO: new LM model without extra parameters
    model_flm = FDNN(MyModelB(9, 9)).to(device)
    net_flm = model_flm.hyper_train(trainset, lamb_hyper)
    #  [10**x for x in range(-1, 2)])
    res = res.append(net_flm.to_df(testset, "FLM"))

    net = FDNN(MyModelB(9, 9), MyModelB(9, 9)).to(device)
    net_fnn = net.hyper_train(trainset, lamb_hyper)
    res = res.append(net_fnn.to_df(testset, "FNN"))

    name_pre = gene + "_" + str(9) + ".md"
    net_pre = net.pre_train(name_pre, gene)
    net_tl = net_pre.hyper_train(trainset, lamb_hyper)
    res = res.append(net_tl.to_df(testset, "TL"))

    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

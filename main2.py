#!/usr/bin/env python
# coding: utf-8
import os
import torch

from skfda.representation.basis import BSpline, Fourier
from data import FunctionalData2
from fdnn import MyModelB, FDNN
from solution import Linear


def main(seed_index, name="CHRNB3"):
    torch.set_default_tensor_type(torch.DoubleTensor)
    # bss = BSpline(n_basis=16, order=4)
    bss = Fourier((0, 1), n_basis=15, period=2)
    dataset = FunctionalData2(name, data="sage")
    path_output = os.path.join("..", "Output", "fnn", "sage", name,
                               str(bss.n_basis), "")
    device = dataset.x.device
    lamb_hyper = [10**x for x in range(-5, -2)]
    trainset, testset = dataset.split_seed(seed_index)
    bl = Linear(trainset, True)
    res = bl.to_df(testset, "Base")
    # TODO: new LM model without extra parameters
    model_flm = FDNN(MyModelB(bss, bss)).to(device)
    net_flm = model_flm.hyper_train(trainset, lamb_hyper)
    res = res.append(net_flm.to_df(testset, "FLM"))
    net = FDNN(MyModelB(bss, bss), MyModelB(bss, bss)).to(device)
    net_fnn = net.hyper_train(trainset, lamb_hyper)
    res = res.append(net_fnn.to_df(testset, "FNN"))

    folder_model = os.path.join("..", "Models", "fnn", "ukb", "")
    model_name = name + "_" + str(bss.n_basis) + ".md"
    if os.path.exists(folder_model + model_name):
        net_old = torch.load(folder_model + model_name, map_location=device)
        net_old.eval()
    else:
        # oldset = Dataset1(bss, name, data="ukb", race=1001, target="sage")
        oldset = FunctionalData2(name, data="ukb")
        net_old = net.hyper_train(oldset, [10**x for x in range(-6, -3)])
        i = 0
        while hasattr(net_old, 'model' + str(i+1)):
            model_ = getattr(net_old, 'model' + str(i))
            model_.b0, model_.b1 = None, None
            for param in model_.parameters():
                param.requires_grad = False
            i += 1
        model_ = getattr(net_old, 'model' + str(i))
        model_.b0, model_.b1 = None, None
        os.makedirs(folder_model, exist_ok=True)
        torch.save(net_old, folder_model + model_name)

    net_tl = net_old.hyper_train(trainset, lamb_hyper)
    res = res.append(net_tl.to_df(testset, "TL"))
    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return

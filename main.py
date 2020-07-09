# !/usr/bin/env python
# coding: utf-8
import os
import torch

from data import Dataset1
from net import Layer, MyEnsemble


def main(seed_index, gene, data="ukb", hidden_units=[8]):
    torch.set_default_tensor_type(torch.DoubleTensor)
    classification = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if data == "ukb":
        dataset = Dataset1(gene, data=data, race=4002)
    else:
        dataset = Dataset1(gene, data=data)
    dataset.process(classification, device)
    trainset, testset = dataset.split_seed(seed_index)
    dims = [dataset.x.shape[1]] + hidden_units + [1 + classification]
    res = testset.to_df(trainset.base(), "Base", trainset)
    net = MyEnsemble(
        [Layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]).to(device)
    net_nn = net.ann(trainset, [10**(x/2) for x in range(-5, 1)])
    res = res.append(testset.to_df(net_nn, "NN", trainset))
    folder_model = os.path.join("..", "Models", data, "")
    model_name = gene + "_" + str("_".join(str(x) for x in hidden_units)) + ".md"
    if os.path.exists(folder_model + model_name):
        net_old = torch.load(folder_model + model_name, map_location=device)
        net_old.eval()
    else:
        oldset = Dataset1(gene, data="ukb", race=1001, target=data)
        oldset.process(classification, device)
        net_old = net.ann(oldset, [10**(x/2) for x in range(-7, -1)])
        i = 0
        while hasattr(net_old, 'model' + str(i+1)):
            for param in getattr(net_old, 'model' + str(i)).parameters():
                param.requires_grad = False
            i += 1
        os.makedirs(folder_model, exist_ok=True)
        torch.save(net_old, folder_model + model_name)

    net_tl = net_old.ann(trainset, [10**(x/2) for x in range(-7, -1)])
    res = res.append(testset.to_df(net_tl, "TL", trainset))
    path_output = os.path.join("..", "Output_nn", data, gene, "")
    os.makedirs(path_output, exist_ok=True)
    res.to_csv(path_output + str(seed_index) + ".csv", index=False)
    print(res)
    return


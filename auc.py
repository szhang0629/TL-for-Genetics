import copy
import random
import torch

import numpy as np
import pandas as pd
import torch.optim as optim


class SeedSequence:
    def __init__(self, seed, n, split_ratio=0.8):
        self.sequence = list(range(n))
        random.Random(seed).shuffle(self.sequence)
        point = round(n * split_ratio)
        self.sequence_split = [self.sequence[:point], self.sequence[point:]]


def ann(data_set, net, lamb, criterion):
    """penalty hyper parameter selection"""
    length = len(lamb)
    if length > 1:
        valid_list = np.zeros(length)
        train, valid = SeedSequence(629, data_set.y.shape[0]).sequence_split
        dataset_train, dataset_valid = data_set.split([train, valid])
        for i in range(length):
            net_copy = copy.deepcopy(net)
            net_copy = my_train(net_copy, dataset_train, lamb[i], criterion)
            valid_list[i] = criterion(net_copy(dataset_valid.x, dataset_valid.z), dataset_valid.y).tolist()

        print(valid_list)
        lamb = lamb[np.argmin(valid_list)]
    else:
        lamb = lamb[0]

    net = my_train(net, data_set, lamb, criterion)
    return net


def inn(data_train, data_test, net, lamb_hyper, criterion):
    net_result = ann(data_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_result(data_train.x, data_train.z), data_train.y).tolist()
    loss_test = criterion(net_result(data_test.x, data_test.z), data_test.y).tolist()
    return pd.DataFrame(data={'method': ["NN"], 'train': [loss_train], 'test': [loss_test]})


def base(y_train, y_test, criterion, classification=False):
    if classification:
        mean_train = torch.mean(y_train.float()).cpu().tolist()
        mean_train_ = torch.tensor([[mean_train, 1-mean_train]], device=y_train.device)
        loss_train = criterion(torch.cat(y_train.shape[0]*[mean_train_]), y_train).tolist()
        loss_test = criterion(torch.cat(y_test.shape[0]*[mean_train_]), y_test).tolist()
    else:
        mean_train = torch.mean(y_train)
        loss_train = criterion(mean_train * torch.ones(y_train.shape, device=y_train.device), y_train).tolist()
        loss_test = criterion(mean_train * torch.ones(y_test.shape, device=y_train.device), y_test).tolist()

    return pd.DataFrame(data={'method': ["Base"], 'train': [loss_train], 'test': [loss_test]})


def decay(x, y):
    return y * (torch.abs(x) > torch.abs(y)) + x * (torch.abs(x) < torch.abs(y))


def my_train(net, dataset, lamb, criterion):
    optimizer = optim.Adam(net.parameters())
    # optimizer = optim.Adadelta(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    epoch, loss_min, lamb_ = 0, np.float('Inf'), lamb / (dataset.y.shape[0] ** 0.5)
    k, net_cache = 0, copy.deepcopy(net)

    while True:
        optimizer.zero_grad()
        net.eval()
        output = net(dataset.x, dataset.z)
        # pen_fun = [pen_l2, pen_l2, pen_l2]
        # pen_param = [net.ModelA, net.ModelB, net.ModelC]
        mse = criterion(output, dataset.y)
        # pen = sum(map(lambda fun, param_fun: fun(param_fun), pen_fun, pen_param)) * lamb_
        pen = pen_l2(net) * lamb_
        loss = mse + pen
        if epoch % 10 == 0:
            # if epoch % 1000 == 0:
            #     print(epoch, loss.tolist())
            if loss < loss_min*0.999:
                k, loss_min, net_cache = 0, loss.tolist(), copy.deepcopy(net)
            else:
                k += 1
                if k == 100:
                    break
        # ModelA_copy = copy.deepcopy(net.ModelA)
        loss.backward()
        # for name, param in net.ModelA.named_parameters():
        #     if param.requires_grad and ('weight' in name):
        #         if pen_fun[0] == pen_l1:
        #             x_grad = copy.deepcopy(param.grad)*(param == 0.)
        #             x_grad.requires_grad = True
        #             y_grad = torch.sum(torch.abs(x_grad))
        #             y_grad.backward()
        #             param.grad -= decay(param.grad, x_grad.grad * lamb_)
        #         elif pen_fun[0] == pen_group:
        #             x_grad = copy.deepcopy(param.grad)*((torch.sum(param**2, dim=0) == 0.).repeat(param.shape[0], 1))
        #             x_grad.requires_grad = True
        #             y_grad = torch.sum((x_grad.shape[0] * torch.sum(x_grad ** 2, dim=0)) ** 0.5)
        #             y_grad.backward()
        #             param.grad -= decay(param.grad, x_grad.grad * lamb_)

        optimizer.step()
        # if pen_fun[0] == pen_l1 or pen_fun[0] == pen_group:
        #     for param, param_old in zip(net.ModelA.parameters(), ModelA_copy.parameters()):
        #         if param.dim() > 1:
        #             param.requires_grad = False
        #             param -= param * ((torch.sign(param)*torch.sign(param_old)) < 0)
        #             param.requires_grad = True

        epoch += 1

    # zeroes = [torch.sum(param == 0, dim=0).tolist() for param in net_cache.ModelA.parameters()]
    # zeroes = [torch.sum(param ** 2, dim=0).tolist() for param in net_cache.ModelA.parameters()]
    print(epoch - 100 * 10, "loss:", loss_min, "lamb:", lamb)
    return net_cache


def pen_l1(net):
    """l1 penalty on weight parameters"""
    penalty = 0
    for name, param in net.named_parameters():
        # penalty += torch.sum(torch.mul(param, param))
        if 'bias' not in name:
            penalty += torch.sum(torch.abs(param))

    return penalty


def pen_l2(net):
    """l2 penalty on weight parameters"""
    # return sum([torch.sum(param**2) for param in net.parameters()])
    penalty = 0
    for name, param in net.named_parameters():
        # penalty += torch.sum(torch.mul(param, param))
        if 'weight' in name:
            penalty += torch.sum(param**2)

    return penalty


def pen_group(net):
    """group penalty on weight parameters"""
    penalty = 0
    for name, param in net.named_parameters():
        # penalty += torch.sum(torch.mul(param, param))
        if 'weight' in name:
            penalty += torch.sum((param.shape[0]*torch.sum(param**2, dim=0))**0.5)

    return penalty

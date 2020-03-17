import copy
import random
import torch

import numpy as np
import pandas as pd
import torch.optim as optim

# from sklearn import linear_model
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss


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
            net_copy = my_train(net_copy, dataset_train.data, lamb[i], criterion)
            valid_list[i] = criterion(net_copy(dataset_valid.g, dataset_valid.x), dataset_valid.y).tolist()

        print(valid_list)
        lamb = lamb[np.argmin(valid_list)]
    else:
        lamb = lamb[0]

    net = my_train(net, data_set.data, lamb, criterion)
    return net


def inn(data_train, data_test, net, lamb_hyper, criterion):
    net_result = ann(data_train, copy.deepcopy(net), lamb_hyper, criterion)
    loss_train = criterion(net_result(data_train.g, data_train.x), data_train.y).tolist()
    loss_test = criterion(net_result(data_test.g, data_test.x), data_test.y).tolist()
    return pd.DataFrame(data={'method': ["NN"], 'train': [loss_train], 'test': [loss_test]})


def base(y_train, y_test, criterion, classification=False):
    if classification:
        mean_train = torch.mean(y_train.float()).cpu().tolist()
        mean_train_ = torch.tensor([[mean_train, 1-mean_train]], device=y_train.device)
        loss_train = criterion(torch.cat(y_train.shape[0]*[mean_train_]), y_train).cpu().tolist()
        loss_test = criterion(torch.cat(y_test.shape[0]*[mean_train_]), y_test).cpu().tolist()
    else:
        mean_train = torch.mean(y_train)
        loss_train = criterion(mean_train * torch.ones(y_train.shape, device=y_train.device), y_train).cpu().tolist()
        loss_test = criterion(mean_train * torch.ones(y_test.shape, device=y_train.device), y_test).cpu().tolist()
    return pd.DataFrame(data={'method': ["Base"], 'train': [loss_train], 'test': [loss_test]})


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


def decay(x, y):
    return y * (torch.abs(x) > torch.abs(y)) + x * (torch.abs(x) < torch.abs(y))


def my_train(net, dataset, lamb, criterion):
    y, x, z = dataset
    optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
    epoch, loss_min, lamb_ = 0, np.float('Inf'), lamb / y.shape[0]
    k, net_cache = 0, copy.deepcopy(net)

    while True:
        optimizer.zero_grad()
        net.eval()
        output = net(x, z)
        mse, pen = criterion(output, y), (pen_l2(net.modelA) + pen_l2(net.modelB) + pen_l2(net.modelC)) * lamb_
        loss = mse + pen
        if epoch % 10 == 0:
            if epoch % 1000 == 0:
                print(epoch, loss.tolist())
            if loss < loss_min*0.999:
                k, loss_min, net_cache = 0, loss.tolist(), copy.deepcopy(net)
            else:
                k += 1
                if k == 100:
                    break
        # modelA_copy = copy.deepcopy(net.modelA)
        loss.backward()
        # for name, param in net.named_parameters():
        #     if param.requires_grad and ('weight' in name) and ('modelA' in name):
        #         param.grad -= torch.sign(param.grad)*torch.clamp(torch.abs(param.grad), max=lamb_)*(param == 0.)
        #         x_grad = torch.autograd.Variable(param.grad, requires_grad=True)
        #         # y_grad = torch.sum(torch.abs(x_grad))
        #         y_grad = torch.sum((x_grad.shape[0] * torch.sum(x_grad ** 2, dim=0)) ** 0.5)
        #         y_grad.backward()
        #         # param.grad -= decay(param.grad, x_grad.grad * lamb_*(param == 0.))
        #         param.grad -= decay(param.grad, x_grad.grad * lamb_ *
        #                             ((torch.sum(param**2, dim=0) == 0.).repeat(param.shape[0], 1)))
        #
        optimizer.step()
        # for param, param_old in zip(net.modelA.parameters(), modelA_copy.parameters()):
        #     if param.dim() > 1:
        #         param.requires_grad = False
        #         param -= param * ((torch.sign(param)*torch.sign(param_old)) < 0)
        #         param.requires_grad = True
        epoch += 1

    zeroes = [torch.sum(param == 0, dim=0).tolist() for param in net_cache.modelA.parameters()]
    # zeroes = [torch.sum(param ** 2, dim=0).tolist() for param in net_cache.modelA.parameters()]
    print(epoch - 100 * 10, "loss:", loss_min, "lamb:", lamb, "zeroes:", zeroes)
    return net_cache


# def my_lasso(data, data_test, lamb, criterion):
#     if type(lamb) is list:
#         length = len(lamb)
#         if length > 1:
#             valid_list = np.zeros(length)
#             train, valid = SeedSequence(629, data.y.shape[0]).sequence_split
#             data_train, data_valid = data.split([train, valid])
#             for i in range(length):
#                 valid_list[i] = my_lasso(data_train, data_valid, lamb[i], criterion)['test'][0]
#
#             print(valid_list)
#             lamb = lamb[np.argmin(valid_list)]
#         else:
#             lamb = lamb[0]
#
#     data = data.numpy()
#     data_test = data_test.numpy()
#     if lamb == 0.:
#         clf = linear_model.LinearRegression().fit(data[1], data[0])
#         loss_train = criterion(torch.from_numpy(clf.predict(data[1])), torch.from_numpy(data[0])).cpu().tolist()
#         loss_test = criterion(torch.from_numpy(clf.predict(data_test[1])),
#                               torch.from_numpy(data_test[0])).cpu().tolist()
#     else:
#         clf = linear_model.Lasso(alpha=lamb, max_iter=10000).fit(data[1], data[0])
#         loss_train = criterion(torch.from_numpy(clf.predict(data[1])[None].T),
#                                torch.from_numpy(data[0])).cpu().tolist()
#         loss_test = criterion(torch.from_numpy(clf.predict(data_test[1])[None].T),
#                               torch.from_numpy(data_test[0])).cpu().tolist()
#
#     return pd.DataFrame(data={'method': ["LM"], 'train': [loss_train], 'test': [loss_test]})
#
#
# def my_log(data, data_test):
#     clf = LogisticRegression(random_state=0, C=0.00001).fit(data.g, data.y)
#     loss_train = log_loss(data_test.y, clf.predict(data_test.g))
#     loss_test = log_loss(data.y, clf.predict(data.g))
#
#     return pd.DataFrame(data={'method': ["LR"], 'train': [loss_train], 'test': [loss_test]})

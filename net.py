import copy
import os
from abc import ABC

import torch
from torch import nn as nn
from torch import optim as optim

from solution import Solution


class Net(Solution, nn.Module, ABC):
    def __init__(self, lamb=None):
        self.layers = None
        nn.Module.__init__(self)
        Solution.__init__(self, lamb)

    def fit(self, dataset, lamb):
        self.fit_init(dataset)
        self.epoch, self.lamb, k = 0, lamb, 0
        # self.lamb_sca = self.lamb / (self.size ** 0.5)
        cache = copy.deepcopy(self)
        optimizer = optim.Adam(self.parameters())
        risk_min = float('Inf')
        while self.epoch < 1e5:
            optimizer.zero_grad()
            pen = self.penalty()
            # if pen > 1e-3:
            #     pen.backward()
            #     optimizer.step()
            #     continue
            loss = dataset.loss(self)
            self.eval()
            # risk = loss / (self.std ** 2)
            risk = loss / (self.std ** 2) + pen
            if self.epoch % 10 == 0:
                if risk < risk_min:
                    self.loss = loss.tolist()
                    cache = copy.deepcopy(self)
                    risk_min = risk.tolist()
                    k = 0
                else:
                    k += 1
                    if k == 50:
                        break
                if self.epoch % 10000 == 0:
                    print(self.epoch, loss.tolist(), pen.tolist(), risk.tolist())
            risk.backward()
            optimizer.step()
            self.epoch += 1
        self.__dict__.update(cache.__dict__)
        self.fit_end()

    def save(self, folder, name, keep=None):
        if keep is not None:
            i = 0
            while hasattr(self, 'model' + str(i + keep)):
                for param in getattr(self, 'model' + str(i)).parameters():
                    param.requires_grad = False
                i += 1
        os.makedirs(folder, exist_ok=True)
        torch.save(self, folder + name)

    def pre_train(self, target, lamb=None):
        method = self.__class__.__name__
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dir_pre = os.path.join("..", "Models", method, target.gene, "")
        oldset = target.__class__(target.gene, target=target)
        name = oldset.name + "_" + self.str_units + ".md"
        if os.path.exists(dir_pre + name):
            net_pre = torch.load(dir_pre + name, map_location=device)
            net_pre.eval()
        else:
            self.to(device)
            net_pre = self.hyper_train(oldset)
            net_pre.save(dir_pre, name, 1)

        net_pre.hyper_lamb = lamb
        return net_pre

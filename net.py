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
        torch.set_default_tensor_type(torch.DoubleTensor)

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
        device = \
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        dir_pre = os.path.join("..", "Models", method, target.gene, "")
        oldset = target.__class__(target.gene, target=target)
        name = oldset.name + "_" + self.str_units + ".md"
        lamb_cache = self.hyper_lamb
        if os.path.exists(dir_pre + name):
            net_pre = torch.load(dir_pre + name, map_location=device)
            net_pre.eval()
        else:
            self.to(device)
            net_pre = oldset.hyper_train(self, lamb)
            net_pre.save(dir_pre, name, 1)

        net_pre.hyper_lamb = lamb_cache
        # net_pre.method_name = "FTL" if method[0] == "F" else "TL"
        return net_pre

    # def fit(self, dataset, lamb):
    #     """
    #     weight decay
    #     """
    #     self.epoch, self.lamb, self.lr, k = 0, lamb, 1e-2, 0
    #     self.fit_init(dataset)
    #     cache = copy.deepcopy(self)
    #     lastlayer = getattr(self, 'model' + str(self.layers - 1))
    #     # lastunit = len(lastlayer.fc.weight.data[0])
    #     optimizer = optim.Adam(
    #         ([{'params': self.model0.parameters(),
    #            'lr': 1e-3, 'weight_decay': 0}]
    #          if self.model0.fc.weight.requires_grad else []) +
    #         [{'params': getattr(self, 'model' + str(i)).parameters(),
    #           'lr': 1e-3, 'weight_decay': 0}
    #          for i in range(1, self.layers - 1)
    #          if getattr(self, 'model' + str(i)).fc.weight.requires_grad] +
    #         [{'params': lastlayer.parameters(),
    #           'lr': 1e-4, 'weight_decay': self.lamb}])
    #     risk_min = float('Inf')
    #     while self.epoch < 1e6:
    #         optimizer.zero_grad()
    #         # pen = self.penalty() * self.lamb
    #         loss = dataset.loss(self)
    #         self.eval()
    #         risk = loss / (self.std ** 2)
    #         if risk < risk_min:
    #             self.loss = loss.tolist()
    #             cache = copy.deepcopy(self)
    #             risk_min = risk.tolist()
    #             k = 0
    #         else:
    #             k += 1
    #             if k == 3000:
    #                 break
    #         risk.backward()
    #         optimizer.step()
    #         self.epoch += 1
    #         # print(risk.tolist())
    #         # if self.epoch % 1000 == 0:
    #         #     print(self.epoch, k, risk.tolist(), self.penalty().tolist())
    #     self.__dict__.update(cache.__dict__)
    #     self.fit_end()
    #     return

    # def fit(self, dataset, lamb):
    #     """
    #     bound penalty
    #     """
    #     self.epoch, self.lamb, k = 0, lamb, 0
    #     self.lr = 1e-2 / self.lamb
    #     self.fit_init(dataset)
    #     cache = copy.deepcopy(self)
    #     lastlayer = getattr(self, 'model' + str(self.layers - 1))
    #     optimizer = optim.Adam(
    #         ([{'params': self.model0.parameters(), 'lr': 1e-3}]
    #          if self.model0.fc.weight.requires_grad else []) +
    #         [{'params': getattr(self, 'model' + str(i)).parameters(),
    #           'lr': 1e-3} for i in range(1, self.layers - 1)
    #          if getattr(self, 'model' + str(i)).fc.weight.requires_grad] +
    #         [{'params': lastlayer.parameters(), 'lr': self.lr}],
    #         eps=1e-17)
    #     # optimizer_sgd = optim.SGD(
    #     #     ([{'params': self.model0.parameters(), 'lr': 1e-3}]
    #     #      if self.model0.fc.weight.requires_grad else []) +
    #     #     [{'params': getattr(self, 'model' + str(i)).parameters(),
    #     #       'lr': 1e-3} for i in range(1, self.layers - 1)
    #     #      if getattr(self, 'model' + str(i)).fc.weight.requires_grad] +
    #     #     [{'params': lastlayer.parameters(), 'lr': self.lr}])
    #     risk_min = float('Inf')
    #     while self.epoch < 1e5:
    #         pen = self.penalty()
    #         loss = dataset.loss(self)
    #         self.eval()
    #         risk = loss / (self.std ** 2)
    #         if pen <= 1. / self.lamb:
    #             if risk < risk_min:
    #                 self.loss, risk_min, k = loss.tolist(), risk.tolist(), 0
    #                 cache = copy.deepcopy(self)
    #             else:
    #                 k += 1
    #                 if k == 3000:
    #                     break
    #             optimizer.zero_grad()
    #             risk.backward()
    #             optimizer.step()
    #         else:
    #             # k = 0
    #             # optimizer = optim.Adam(
    #             #     ([{'params': self.model0.parameters(), 'lr': 1e-3}]
    #             #      if self.model0.fc.weight.requires_grad else []) +
    #             #     [{'params': getattr(self, 'model' + str(i)).parameters(),
    #             #       'lr': 1e-3} for i in range(1, self.layers - 1)
    #             #      if
    #             #      getattr(self, 'model' + str(i)).fc.weight.requires_grad] +
    #             #     [{'params': lastlayer.parameters(), 'lr': self.lr / 10}],
    #             #     eps=1e-17)
    #             optimizer.zero_grad()
    #             pen.backward()
    #             optimizer.step()
    #             # print(self.epoch, k, risk.tolist(), "before:", pen.tolist(),
    #             #       "then", self.penalty().tolist())
    #         self.epoch += 1
    #     self.__dict__.update(cache.__dict__)
    #     self.fit_end()
    #     return

    def fit(self, dataset, lamb):
        """
        add penalty
        """
        self.epoch, self.lamb, k = 0, lamb, 0
        self.lr = 1e-2 / self.lamb
        self.fit_init(dataset)
        cache = copy.deepcopy(self)
        lastlayer = getattr(self, 'model' + str(self.layers - 1))
        optimizer = optim.Adam(
            ([{'params': self.model0.parameters(), 'lr': 1e-3}]
             if self.model0.fc.weight.requires_grad else []) +
            [{'params': getattr(self, 'model' + str(i)).parameters(),
              'lr': 1e-3} for i in range(1, self.layers - 1)
             if getattr(self, 'model' + str(i)).fc.weight.requires_grad] +
            [{'params': lastlayer.parameters(), 'lr': self.lr}], eps=1e-17)
        risk_min = float('Inf')
        changed, epoch_t = False, 0
        while self.epoch < 1e5:
            loss = dataset.loss(self)
            self.eval()
            risk = loss / (self.std ** 2) + self.penalty()
            if loss < self.loss:
                self.loss, risk_min, k = loss.tolist(), risk.tolist(), 0
                cache, epoch_t_cache = copy.deepcopy(self), epoch_t
            else:
                k += 1
                if k == 1000:
                    break
            # if self.epoch % 10000 == 0:
            #     print(self.epoch, k, risk.tolist(), self.penalty().tolist())
            optimizer.zero_grad()
            risk.backward()
            optimizer.step()
            self.epoch += 1
        self.__dict__.update(cache.__dict__)
        print(self.epoch, k, risk_min, self.penalty().tolist())
        self.fit_end()
        return

import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim


class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=0):
        torch.manual_seed(0)
        super(Layer, self).__init__()
        self.fc = nn.Linear(in_dim + z_dim, out_dim)

    def forward(self, x, z=None):
        if z is not None:
            x = torch.cat([x, z], 1)
        return self.fc(x)


class MyEnsemble(nn.Module):
    def __init__(self, models):
        self.size, self.epoch, self.lamb = None, 0, 1
        self.loss = float('Inf')
        self.mean, self.std = 0, 1
        super(MyEnsemble, self).__init__()
        for i in range(len(models)):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        res = self.model0(dataset.x, dataset.z)
        i = 1
        while hasattr(self, 'model' + str(i)):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
            i += 1
        return res*self.std + self.mean

    def fit(self, dataset, lamb):
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.size, self.epoch, self.lamb, k = dataset.x.shape[0], 0, lamb, 0
        cache = copy.deepcopy(self)
        optimizer = optim.Adam(self.parameters())  # , weight_decay=self.lamb)
        risk_min = float('Inf')
        while self.epoch < 1e5:
            optimizer.zero_grad()
            # self.loss = dataset.loss(self)
            loss = dataset.loss(self)
            self.loss = loss.tolist()
            self.eval()
            risk = loss / (self.std ** 2) + self.penalty()*self.lamb
            if self.epoch % 10 == 0:
                # if self.loss < cache.loss:
                if risk < risk_min:
                    cache = copy.deepcopy(self)
                    risk_min, k = risk.tolist(), 0
                if k == 100:
                    break
                # if self.epoch % 1000 == 0:
                #     print(self.epoch, self.loss)
            risk.backward()
            optimizer.step()
            self.epoch, k = self.epoch+1, k+1
        self.__dict__.update(cache.__dict__)
        print(self.epoch, self.loss)

    def ann(self, dataset, lamb):
        if type(lamb) is not list:
            net = copy.deepcopy(self)
            net.fit(dataset, lamb)
            return net
        trainset, validset = dataset.split_seed()
        valid = [validset.loss(self.ann(trainset, decay)).tolist()
                 for decay in lamb]
        print(valid)
        lamb_opt = lamb[np.argmin(np.array(valid))]
        return self.ann(dataset, lamb_opt)

    def penalty(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:  # and not name.endswith(".bias"):
                penalty += torch.sum(param ** 2)
        return penalty

    def save(self, folder, name, keep=None):
        if keep is not None:
            i = 0
            while hasattr(self, 'model' + str(i + keep)):
                for param in getattr(self, 'model' + str(i)).parameters():
                    param.requires_grad = False
                i += 1
        os.makedirs(folder, exist_ok=True)
        torch.save(self, folder + name)

    def transfer(self, dataset):
        res = self.model0(dataset.x, dataset.z)
        i = 1
        while hasattr(self, 'model' + str(i+1)):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
            i += 1
        transfer_set = copy.deepcopy(dataset)
        transfer_set.x = torch.sigmoid(res)
        return transfer_set

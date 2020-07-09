import copy
import numpy as np
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
        self.epoch, self.lamb = 0, 1
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
        self.epoch, self.lamb, k, loss_min = 0, lamb, 0, float('Inf')
        optimizer = optim.Adam(self.parameters(), weight_decay=self.lamb)
        while True:
            optimizer.zero_grad()
            self.eval()
            loss = dataset.loss(self) / (self.std**2)
            if self.epoch % 10 == 0:
                if loss < loss_min:
                    k, loss_min = 0, loss.tolist()
                    cache = copy.deepcopy(self)
                if k == 100:
                    break
            loss.backward()
            optimizer.step()
            self.epoch, k = self.epoch+1, k+1
        self.__dict__.update(cache.__dict__)
        print(self.epoch, loss_min)

    def ann(self, dataset, lamb):
        if type(lamb) is not list:
            net = copy.deepcopy(self)
            net.fit(dataset, lamb)
            return net
        trainset, validset = dataset.split_seed()
        valid = [validset.loss(self.ann(trainset, lamb_)) for lamb_ in lamb]
        print(valid)
        lamb_opt = lamb[np.argmin(np.array(valid))]
        return self.ann(dataset, lamb_opt)
        

# class Mnet(nn.Module):
#     def __init__(self, models, model0, model1):
#         self.epoch, self.lamb = 0, 1
#         self.mean0, self.std0 = 0, 1
#         self.mean1, self.std1 = 0, 1
#         super(Mnet, self).__init__()
#         for i in range(len(models)):
#             setattr(self, "model" + str(i), models[i])
#         self.net0 = model0
#         self.net1 = model1
#
#     def forward(self, dataset):
#         res = self.model0(dataset.x, dataset.z)
#         i = 1
#         while hasattr(self, 'model' + str(i)):
#             res = torch.sigmoid(res)
#             res = getattr(self, 'model' + str(i))(res)
#             i += 1
#         res = torch.sigmoid(res)
#         if hasattr(dataset, 'old'):
#             if dataset.old:
#                 return self.net1(res)*self.std1 + self.mean1
#         return self.net0(res)*self.std0 + self.mean0
#
#     def fit(self, dataset, lamb, oldset):
#         oldset.old = True
#         self.mean0, self.std0 = torch.mean(dataset.y), torch.std(dataset.y)
#         self.mean1, self.std1 = torch.mean(oldset.y), torch.std(dataset.y)
#         self.epoch, self.lamb, k, loss_min = 0, lamb, 0, float('Inf')
#         optimizer = optim.Adam(self.parameters(), weight_decay=self.lamb)
#         while True:
#             optimizer.zero_grad()
#             self.eval()
#             loss = dataset.loss(self, False) / (self.std0**2)
#             if self.epoch % 10 == 0:
#                 if loss < loss_min:
#                     k, loss_min = 0, loss.tolist()
#                     cache = copy.deepcopy(self)
#                 if k == 100:
#                     break
#             if oldset is not None:
#                 loss += oldset.loss(self, False) / (self.std1**2)
#             loss.backward()
#             optimizer.step()
#             self.epoch, k = self.epoch+1, k+1
#         self.__dict__.update(cache.__dict__)
#         print(self.epoch, loss_min)
#
#     def ann(self, dataset, lamb, oldset=None):
#         if type(lamb) is not list:
#             net = copy.deepcopy(self)
#             net.fit(dataset, lamb, oldset)
#             return net
#         trainset, validset = dataset.split_seed()
#         valid = [validset.loss(self.ann(trainset, lamb_, oldset)) for lamb_ in lamb]
#         print(valid)
#         lamb_opt = lamb[np.argmin(np.array(valid))]
#         return self.ann(dataset, lamb_opt, oldset)

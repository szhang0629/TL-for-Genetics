import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ModelA(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ModelA, self).__init__()
        if type(in_dim) is list:
            for i in range(len(in_dim)):
                setattr(self, "fc" + str(i), nn.Linear(in_dim[i], out_dim))
        else:
            # self.drop_layer = nn.Dropout(p=0.5)
            self.fc = nn.Linear(in_dim, out_dim)
            # self.bn = nn.BatchNorm1d(num_features=out_dim)

    def forward(self, x, z=None):
        if type(x) is list:
            x1 = [getattr(self, "fc"+str(i))(x[i]) for i in range(len(x))]
            return torch.cat(x1, 1)
        else:
            # x = self.drop_layer(x)
            return self.fc(x)


class ModelC(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=0):
        super(ModelC, self).__init__()
        x_dim = sum(in_dim) if type(in_dim) is list else in_dim
        self.fc = nn.Linear(x_dim + z_dim, out_dim)

    def forward(self, x, z=None):
        if type(x) is list:
            x = torch.cat(x, 1)
        if z is not None:
            x = torch.cat([x, z], 1)
        return self.fc(x)


class MyEnsemble(nn.Module):
    def __init__(self, *args):
        super(MyEnsemble, self).__init__()
        i = 0
        for arg in args:
            setattr(self, "model" + str(i), arg)
            i += 1
        self.model_old = copy.deepcopy(getattr(self, "model" + str(i-1)))

    def forward(self, dataset, old=False):
        res = self.model0(dataset.x, dataset.z)
        if old:
            res = torch.sigmoid(res)
            return self.model_old(res)
        if hasattr(self, 'model1'):
            res = torch.sigmoid(res)
            res = self.model1(res)
        return res

    def fit(self, dataset, lamb, dataset_old):
        criterion = nn.CrossEntropyLoss() if dataset.classification else nn.MSELoss()
        net, epoch, k, loss_min = copy.deepcopy(self), 0, 0, float('Inf')
        optimizer = optim.Adam(net.parameters()) # , weight_decay=lamb)
        while True:
            optimizer.zero_grad()
            net.eval()
            loss = criterion(net(dataset), dataset.y)
            if epoch % 10 == 0:
                if loss < loss_min:
                    k, loss_min = 0, loss.tolist()  
                    self.__dict__.update(net.__dict__)
                else:
                    k += 1
                    if k == 100:
                        break
            if dataset_old is not None:
                loss += criterion(net(dataset_old, True), dataset_old.y)
            pen = net.penalty() * lamb
            loss_plus = loss + pen
            loss_plus.backward()
            optimizer.step()
            epoch += 1
        print(epoch - 100 * 10, "loss:", loss_min)

    def ann(self, dataset, lamb, dataset_old=None):
        net = copy.deepcopy(self)
        valid_list = []
        dataset_train, dataset_valid = dataset.split_seed()
        for lamb_ in lamb:
            net_copy = copy.deepcopy(net)
            net_copy.fit(dataset_train, lamb_, dataset_old)
            valid_list.append(dataset_valid.loss(net_copy))
        print(valid_list)
        lamb = lamb[np.argmin(np.array(valid_list))]
        net.fit(dataset, lamb, dataset_old)
        return net, lamb

    def penalty(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                if not (name.endswith(".bias") and 'model1' in name):
                    penalty += torch.sum(param ** 2)
                    # penalty += torch.sum(torch.abs(param))
        return penalty

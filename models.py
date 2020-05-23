import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ModelA(nn.Module):
    def __init__(self, x, out_dim=4):
        super(ModelA, self).__init__()
        if type(x) is list:
            for i in range(len(x)):
                setattr(self, "fc" + str(i), nn.Linear(x[i].shape[1], out_dim))
        else:
            self.fc = nn.Linear(x.shape[1], out_dim)

    def forward(self, x):
        if type(x) is list:
            x1 = [torch.sigmoid(getattr(self, "fc"+str(i))(x[i])) for i in range(len(x))]
            return torch.cat(x1, 1)
        else:
            return torch.sigmoid(self.fc(x))


class ModelC(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=0):
        super(ModelC, self).__init__()
        self.fc = nn.Linear(in_dim+z_dim, out_dim)

    def forward(self, x, z=None):
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

    def forward(self, dataset, dataset_old=None):
        if hasattr(self, 'model1'):
            x1 = self.model0(dataset.x)
            y = self.model1(x1, dataset.z)
            if dataset_old is not None:
                x1_old = self.model0(dataset_old.x)
                y_old = self.model_old(x1_old, dataset_old.z)
                return torch.cat([y, y_old], 0)
            return y
        else:
            return self.model0(dataset.x, dataset.z)

    def fit(self, dataset, lamb, dataset_old):
        criterion = nn.CrossEntropyLoss() if dataset.classification else nn.MSELoss()
        net, epoch, k, mse_min, pen_min = copy.deepcopy(self), 0, 0, 0, 0
        optimizer = optim.Adam(net.parameters())
        total_num = dataset.y.shape[0] + (0 if dataset_old is None else dataset_old.y.shape[0])
        loss_min, lamb_ = float('Inf'), lamb / (total_num ** 0.5)
        y = dataset.y if dataset_old is None else torch.cat([dataset.y, dataset_old.y], 0)
        while True:
            optimizer.zero_grad()
            net.eval()
            output = net(dataset, dataset_old)
            mse = criterion(output, y)
            pen = sum(net.penalty()) * lamb_
            loss = mse + pen
            if epoch % 10 == 0:
                if loss < loss_min:
                    k, loss_min, mse_min, pen_min = 0, loss.tolist(), mse.tolist(), pen.tolist()
                    self.__dict__.update(net.__dict__)
                else:
                    k += 1
                    if k == 100:
                        break
            loss.backward()
            optimizer.step()
            epoch += 1

        print(epoch - 100 * 10, "mse:", mse_min, "pen:", pen_min, "loss:", loss_min)

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
        # return [torch.sum(param**2) for param in net.parameters()]
        # return [torch.sum(param ** 2) for name, param in self.named_parameters() if ('weight' in name)
        #         and param.requires_grad]
        # return [torch.sum(torch.abs(param)) for name, param in self.named_parameters() if param.requires_grad]
        return [torch.sum(torch.abs(param)) for name, param in self.named_parameters() if ('weight' in name)
                and param.requires_grad]

# __init__.py
import copy
import torch
import torch.nn as nn

import torch.optim as optim


class MyEnsemble(nn.Module):
    def __init__(self, *args):
        super(MyEnsemble, self).__init__()
        i = 0
        for arg in args:
            setattr(self, "model" + str(i), arg)
            i += 1

    def forward(self, dataset):
        if hasattr(self, 'model1'):
            x1 = self.model0(dataset.x)
            return self.model1(x1, dataset.z)
        else:
            return self.model0(dataset.x, dataset.z)

    def fit(self, dataset, lamb, criterion):
        k, net = 0, copy.deepcopy(self)
        optimizer = optim.Adam(net.parameters())
        epoch, loss_min, lamb_ = 0, float('Inf'), lamb / (dataset.y.shape[0] ** 0.5)

        while True:
            optimizer.zero_grad()
            net.eval()
            output = net(dataset)
            mse = criterion(output, dataset.y)
            pen = pen_l2(net) * lamb_
            loss = mse + pen
            if epoch % 10 == 0:
                if loss < loss_min:
                    k, loss_min = 0, loss.tolist()
                    self.__dict__.update(net.__dict__)
                else:
                    k += 1
                    if k == 100:
                        break
            loss.backward()
            optimizer.step()
            epoch += 1

        print(epoch - 100 * 10, "loss:", loss_min, "lamb:", lamb)


def pen_l2(net):
    """l2 penalty on weight parameters"""
    # return sum([torch.sum(param**2) for param in net.parameters()])
    penalty = 0
    for name, param in net.named_parameters():
        # penalty += torch.sum(param**2)
        if 'weight' in name:
            penalty += torch.sum(param**2)

    return penalty

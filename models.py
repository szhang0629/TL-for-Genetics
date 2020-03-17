"""Transfer Learning Models"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from auc import pen_l1, pen_l2, pen_group, decay


class Net(nn.Module):
    def __init__(self, x_dim, out_dim, z_dim=0):
        super(Net, self).__init__()
        self.modelA = nn.Sequential(
            # nn.BatchNorm1d(x_dim),
            # nn.Dropout(p=0.5),
            nn.Linear(x_dim, 16),
            nn.Sigmoid()
        )
        self.modelB = nn.Sequential(
            # nn.BatchNorm1d(16),
            # nn.Dropout(p=0.5),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )
        self.modelC = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(4+z_dim),
            nn.Linear(4+z_dim, 8),
            nn.Sigmoid(),
            nn.Linear(8, out_dim)
        )

    def forward(self, x, z):
        x = self.modelA(x)
        x = self.modelB(x)
        if z is not None:
            x = torch.cat([x, z], 1)
        x = self.modelC(x)
        return x

    # def my_train(self, dataset, lamb, criterion):
    #     y, x, z = dataset
    #     optimizer = optim.Adam(self.parameters())
    #     # optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
    #     epoch, loss_min, lamb_ = 0, np.float('Inf'), lamb / y.shape[0]
    #     k, net_cache = 0, copy.deepcopy(self)
    #
    #     while True:
    #         optimizer.zero_grad()
    #         self.eval()
    #         output = self(x, z)
    #         mse, pen = criterion(output, y), \
    #                    (pen_group(self.modelA) + pen_l2(self.modelB) + pen_l2(self.modelC)) * lamb_
    #         loss = mse + pen
    #         if epoch % 10 == 0:
    #             # if epoch % 1000 == 0:
    #             #     print(epoch, loss.tolist())
    #             if loss < loss_min*0.999:
    #                 k, loss_min, net_cache = 0, loss.tolist(), copy.deepcopy(self)
    #             else:
    #                 k += 1
    #                 if k == 100:
    #                     break
    #         modelA_copy = copy.deepcopy(self.modelA)
    #         loss.backward()
    #         for name, param in self.named_parameters():
    #             if param.requires_grad and ('weight' in name) and ('modelA' in name):
    #                 # param.grad -= torch.sign(param.grad)*torch.clamp(torch.abs(param.grad), max=lamb_)*(param == 0.)
    #                 x_grad = torch.autograd.Variable(param.grad, requires_grad=True)
    #                 # y_grad = torch.sum(torch.abs(x_grad))
    #                 y_grad = torch.sum((x_grad.shape[0] * torch.sum(x_grad ** 2, dim=0)) ** 0.5)
    #                 y_grad.backward()
    #                 # param.grad -= decay(param.grad, x_grad.grad * lamb_*(param == 0.))
    #                 param.grad -= decay(param.grad, x_grad.grad * lamb_ *
    #                                     ((torch.sum(param**2, dim=0) == 0.).repeat(param.shape[0], 1)))
    #
    #         optimizer.step()
    #         for param, param_old in zip(self.modelA.parameters(), modelA_copy.parameters()):
    #             if param.dim() > 1:
    #                 param.requires_grad = False
    #                 param -= param * ((torch.sign(param)*torch.sign(param_old)) < 0)
    #                 param.requires_grad = True
    #         epoch += 1
    #
    #     zeroes = [torch.sum(param == 0, dim=0).tolist() for param in net_cache.modelA.parameters()]  # [0]
    #     print(epoch - 100 * 10, "loss:", loss_min, "lamb:", lamb, "zeroes:", zeroes)
    #     self = copy.deepcopy(net_cache)
    #     return

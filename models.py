"""Transfer Learning Models"""
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np

from penalty_parameters import penalty_parameters


class Net(nn.Module):
    def __init__(self, x_dim, out_dim):
        super(Net, self).__init__()
        self.modelA = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(x_dim, 64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        self.modelB = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(4, 8),
            nn.Sigmoid(),
            nn.Linear(8, out_dim)
        )

    def forward(self, x, z):
        x = self.modelA(x)
        x = self.modelB(x)
        return x

    def my_train(self, dataset, lamb, criterion):
        y, x, z = dataset
        optimizer = optim.Adam(self.parameters())
        # optimizer = optim.SGD(self.parameters(), lr=3, momentum=0.9)
        epoch, loss_min = 0, np.float('Inf')
        k, net_cache = 0, copy.deepcopy(self)

        while True:
            optimizer.zero_grad()
            self.eval()
            output = self(x, z)
            mse, pen = criterion(output, y), penalty_parameters(self) * lamb / y.shape[0]
            loss = mse + pen
            if epoch % 10 == 0:
                if epoch % 1000 == 0:
                    print(epoch, loss.tolist())
                if loss < loss_min*0.995:
                    k, loss_min, net_cache = 0, loss.tolist(), copy.deepcopy(self)
                else:
                    k += 1
                    if k == 100:
                        break
            loss.backward()
            optimizer.step()
            epoch += 1

        print(epoch - 100 * 10, "loss:", loss_min)
        self = net_cache


# class Nets(nn.Module):
#     def __init__(self, x_dim, out_dim):
#         super(Nets, self).__init__()
#         self.modelA = nn.Sequential(
#             # nn.Dropout(p=0.5),
#             nn.Linear(x_dim, 64),
#             nn.Sigmoid(),
#             # nn.Dropout(p=0.5),
#             nn.Linear(64, 16),
#             nn.Sigmoid(),
#             # nn.Dropout(p=0.5),
#             nn.Linear(16, 4),
#             nn.Sigmoid()
#         )
#         self.modelB1 = nn.Sequential(
#             # nn.Dropout(p=0.5),
#             nn.Linear(4, 16),
#             nn.Sigmoid(),
#             nn.Linear(16, out_dim)
#         )
#         self.modelB2 = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(4, 16),
#             nn.Sigmoid(),
#             nn.Linear(16, out_dim)
#         )
#
#     def forward(self, x1, x2=None):
#         x1_ = self.modelA(x1)
#         y1 = self.modelB1(x1_)
#         if x2 is None:
#             return y1
#         else:
#             x2_ = self.modelA(x2)
#             y2 = self.modelB2(x2_)
#             return y1, y2

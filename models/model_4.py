"""Transfer Learning Models"""
import torch
import torch.nn as nn


class ModelA(nn.Module):
    def __init__(self, x):
        super(ModelA, self).__init__()
        if type(x) is list:
            for i in range(len(x)):
                setattr(self, "fc" + str(i), nn.Linear(x[i].shape[1], 4))
                # setattr(self, "fd" + str(i), nn.Linear(16, 4))
        else:
            self.fc = nn.Linear(x.shape[1], 4)
            # self.fd = nn.Linear(16, 4)

    def forward(self, x):
        if type(x) is list:
            x1 = [torch.sigmoid(getattr(self, "fc"+str(i))(x[i])) for i in range(len(x))]
            # x2 = [torch.sigmoid(getattr(self, "fd"+str(i))(x1[i])) for i in range(len(x1))]
            return torch.cat(x1, 1)
        else:
            x1 = torch.sigmoid(self.fc(x))
            return x1
            # return torch.sigmoid(self.fd(x1))


class ModelC(nn.Module):
    def __init__(self, out_dim, z_dim=0):
        super(ModelC, self).__init__()
        self.fc = nn.Linear(4, out_dim)

    def forward(self, x, z=None):
        # if z is not None:
        #     x = torch.cat([x, z], 1)
        return self.fc(x)



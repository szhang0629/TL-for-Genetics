import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn


class Dataset:
    def __init__(self, data, classification=False):
        self.y, self.x, self.z = data
        self.classification = classification

    def to(self, device):
        self.y = self.y.to(device)
        self.x = [x_.to(device) for x_ in self.x] if type(self.x) is list else self.x.to(device)
        self.z = None if self.z is None else self.z.to(device)

    def split(self, seq):
        res = []
        for seq_ in seq:
            x_ = [x_[seq_] for x_ in self.x] if type(self.x) is list else self.x[seq_]
            z_ = None if self.z is None else self.z[seq_]
            res_ = Dataset(data=[self.y[seq_], x_, z_])
            res_.to(self.y.device)
            res.append(res_)
        return res

    def split_seed(self, seed=0, split_ratio=0.8):
        sequence = list(range(self.y.shape[0]))
        random.Random(seed).shuffle(sequence)
        point = round(len(sequence) * split_ratio)
        return self.split([sequence[:point], sequence[point:]])

    def process(self, classification=False):
        self.classification = classification
        y = self.y
        if self.classification:
            y[y != 0] = 1
            y = y.squeeze()
            self.y = y.long()
        else:
            self.y = torch.log(self.y + 1)
            self.y = (self.y - torch.mean(self.y)) / torch.std(self.y)

    def base(self, y_base):
        if self.classification:
            criterion = nn.CrossEntropyLoss()
            y_base_ = torch.tensor([[y_base, 1 - y_base]], device=self.y.device)
            return criterion(torch.cat(self.y.shape[0]*[y_base_]), self.y).tolist()
        else:
            criterion = nn.MSELoss()
            return criterion(y_base * torch.ones(self.y.shape, device=self.y.device), self.y).tolist()

    def loss(self, model):
        criterion = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()
        return criterion(model(self), self.y).tolist()


class Dataset1(Dataset):
    def __init__(self, name="CHRNA5", name_data=None):
        if type(name) is list:
            x = [pd.read_csv("../../Data/" + name_ + "/g_" + name_data + ".csv", index_col=0) for name_ in name]
        else:
            x = pd.read_csv("../../Data/" + name + "/g_" + name_data + ".csv", index_col=0)
        z = pd.read_csv("../../Data/Phe/x_" + name_data + ".csv", index_col=0)
        z[['age']] = (z[['age']] - 13) / 70
        y = pd.read_csv("../../Data/Phe/y_" + name_data + ".csv", index_col=0)
        iid = np.intersect1d(z.index.values, y.index.values)
        if type(x) is list:
            for x_ in x:
                iid = np.intersect1d(iid, x_.index.values)
        else:
            iid = np.intersect1d(iid, x.index.values)
        z, y = torch.from_numpy(z.loc[iid].values).float(), torch.from_numpy(y.loc[iid].values).float()
        if type(x) is list:
            x = [torch.from_numpy(x_.loc[iid].values).float() for x_ in x]
        else:
            x = torch.from_numpy(x.loc[iid].values).float()
        super().__init__(data=[y, x, z])


class Dataset2(Dataset):
    def __init__(self, name="CHRNA5"):
        x = pd.read_csv("../../Data/" + name + "/g.csv", index_col=0)
        x = torch.from_numpy(x.values).float()
        z = pd.read_csv("../../Data/Phe/x.csv", index_col=0)
        y = pd.read_csv("../../Data/Phe/y.csv", index_col=0)

        z[['age_int']] = (z[['age_int']] - 13) / 70
        z = z[['race', 'sex', 'age_int']]
        z = torch.from_numpy(z.values).float()
        y = torch.from_numpy(y.values).float()
        super().__init__(data=[y, x, z])
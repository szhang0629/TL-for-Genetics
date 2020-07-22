import random

import torch
from torch import nn as nn


class Dataset:
    def __init__(self, y, x, z, classification=False):
        self.classification = classification
        self.y, self.x, self.z = y, x, z

    def to_tensor(self):
        self.y = torch.from_numpy(self.y.values).float()
        self.x = torch.from_numpy(self.x.values).float()
        if self.z is not None:
            self.z = torch.from_numpy(self.z.values).float()

    def to(self, device):
        self.y, self.x = self.y.to(device), self.x.to(device)
        self.z = None if self.z is None else self.z.to(device)

    def split_seed(self, seed=0, split_ratio=0.8):
        sequence = list(range(self.x.shape[0]))
        random.Random(seed).shuffle(sequence)
        point = round(len(sequence) * split_ratio)
        return self.split([sequence[:point], sequence[point:]])

    def process(self, classification=False):
        self.classification = classification
        self.to_tensor()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if classification:
            self.y[self.y != 0] = 1
            self.y = self.y.squeeze()
            self.y = self.y.long()
        else:
            # self.y = torch.log(self.y + 1)
            self.to(device)
            self.z = None

    def loss(self, model):
        criterion = nn.CrossEntropyLoss() if self.classification \
            else nn.MSELoss()
        return criterion(model(self), self.y)


class DiscreteData(Dataset):
    def __init__(self, data, classification=False):
        y, x, z = data
        super().__init__(y, x, z, classification=classification)

    def split(self, seq):
        res = []
        for seq_ in seq:
            if self.z is None:
                z_ = None
            else:
                z_ = self.z[seq_]
            res_ = DiscreteData(data=[self.y[seq_], self.x[seq_], z_])
            res.append(res_)
        return res


class FunctionalData(Dataset):
    def __init__(self, data, classification=False):
        y, x, z, self.pos, self.loc = data
        super().__init__(y, x, z, classification=classification)

    def split(self, seq):
        res = []
        for seq_ in seq:
            if self.z is None:
                z_ = None
            else:
                z_ = self.z[seq_]
            res_ = FunctionalData([self.y[seq_], self.x[seq_], z_, self.pos,
                                   self.loc[seq_]],
                                  classification=self.classification)
            res.append(res_)
        return res

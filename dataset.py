import random

import torch
from torch import nn as nn


class Dataset:
    def __init__(self, data, classification=False):
        # torch.set_default_tensor_type(torch.DoubleTensor)
        self.classification = classification
        self.y, self.x, self.z, self.pos = data

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
        self.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        if classification:
            self.y[self.y != 0] = 1
            self.y = self.y.squeeze()
            self.y = self.y.long()
        else:
            # self.y = torch.log(self.y + 1)
            self.z = None

    def loss(self, model):
        criterion = nn.CrossEntropyLoss() if self.classification \
            else nn.MSELoss()
        return criterion(model(self), self.y)

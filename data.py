import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer


class Dataset:
    def __init__(self, data, classification=False):
        self.y, self.x, self.z = data
        self.classification = classification

    def to(self, device):
        self.y, self.x = self.y.to(device), self.x.to(device)
        self.z = None if self.z is None else self.z.to(device)

    def split(self, seq):
        res = []
        for seq_ in seq:
            z_ = None if self.z is None else self.z[seq_]
            res_ = Dataset(data=[self.y[seq_], self.x[seq_], z_])
            res.append(res_)
        return res

    def split_seed(self, seed=0, split_ratio=0.8):
        sequence = list(range(self.y.shape[0]))
        random.Random(seed).shuffle(sequence)
        point = round(len(sequence) * split_ratio)
        return self.split([sequence[:point], sequence[point:]])

    def process(self, classification=False, device=None):
        self.classification = classification
        if self.classification:
            self.y[self.y != 0] = 1
            self.y = (self.y.squeeze()).long()
        else:
            # index = (self.y!=0).squeeze()
            # self.y, self.x = self.y[index], self.x[index]
            # self.y = torch.log(10*self.y + 1)
            self.z = None
        if device is not None:
            self.to(device)

    def base(self):
        y_base = torch.mean(self.y.double())
        
        def model_base(self, y_base=y_base):
            if self.classification:
                y_base_ = torch.tensor([[y_base, 1 - y_base]], device=self.y.device)
                return torch.cat(self.y.shape[0]*[y_base_])
            else:
                return y_base * torch.ones(self.y.shape, device=self.y.device)
        return model_base

    def loss(self, model):
        criterion = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()
        return criterion(model(self), self.y)

    def to_df(self, model, method, trainset):
        if method == "Base":
            return pd.DataFrame(data={'method': ["Base"], 'pen': [1.0],
                                      'train': [trainset.loss(model).tolist()],
                                      'test': [self.loss(model).tolist()]})
        return pd.DataFrame(data={'method': [method], 'pen': [model.lamb],
                                  'train': [trainset.loss(model).tolist()],
                                  'test': [self.loss(model).tolist()]})


class Dataset1(Dataset):
    def __init__(self, gene, data="ukb", race=None, target=None):
        folder = os.path.join("..", "..", "Data", data, "")
        x = pd.read_csv(folder + "/g_" + gene + ".csv", index_col=0)
        yz = pd.read_csv(folder + "Y.csv", index_col=0)
        if race is not None:
            if race//10 == 0:
                race_index = (yz.loc[:, 'eth_org']//1000 == race)
            else:
                race_index = (yz.loc[:, 'eth_org'] == race)
            yz = yz.loc[race_index, :]
        if (target is not None) and (data != target):
            experience = os.path.join("..", "..", "Data", target, "",
                                      "/g_" + gene + ".csv")
            x = x[pd.read_csv(experience, index_col=0).columns.values]
        else:
            x = x.loc[:, x.isnull().sum()/x.shape[0] < 0.01]
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x)
        x = pd.DataFrame(data=imp_mean.transform(x.values),
                         index=x.index, columns=x.columns)
        #x = x.dropna()
        iid = np.intersect1d(yz.index.values, x.index.values)
        x = x.loc[iid, :]
        yz.loc[:, 'age'] = (yz.loc[:, 'age'] - 13) / 70
        z = yz.loc[iid, ['sex', 'age']]
        y = yz.loc[iid, ['smk']]
        z, y = torch.from_numpy(z.values).double(), torch.from_numpy(y.values).double()
        x = torch.from_numpy(x.values).double()
        super().__init__(data=[y, x, z])

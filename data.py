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
            self.z = None
        if device is not None:
            self.to(device)

    def loss(self, model):
        criterion = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()
        return criterion(model(self), self.y)

    def to_df(self, model, method):
        return pd.DataFrame(data={'method': [method], 'pen': [model.lamb],
                                  'train': [model.loss],
                                  'test': [self.loss(model).tolist()]})


class Dataset1(Dataset):
    def __init__(self, gene, data="ukb", race=None, target=None):
        folder = os.path.join("..", "..", "Data", data, "")
        x = pd.read_csv(folder + "g_" + gene + ".csv", index_col=0)
        yz = pd.read_csv(folder + "Y.csv", index_col=0)
        if race is not None:
            if race//10 == 0:
                race_index = (yz.loc[:, 'eth_org']//1000 == race) | (yz.loc[:, 'eth_org'] == race)
            else:
                race_index = (yz.loc[:, 'eth_org'] == race)
            yz = yz.loc[race_index, :]
        if (target is not None) and (data != target):
            experience = os.path.join("..", "..", "Data", target,
                                      "g_" + gene + ".csv")
            x = x[pd.read_csv(experience, index_col=0).columns.values]
            # index_positive = (yz[['smk']].values > 0)
            # yz = yz.loc[index_positive]
        else:
            x = x.loc[:, x.isnull().sum()/x.shape[0] < 0.01]
        if target is None:  # dataset
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp_mean.fit(x)
            x = pd.DataFrame(data=imp_mean.transform(x.values),
                             index=x.index, columns=x.columns)
        else:  # oldset
            x = x.dropna()
        iid = np.intersect1d(yz.index.values, x.index.values)
        x = x.loc[iid, :]
        yz.loc[:, 'age'] = (yz.loc[:, 'age'] - 13) / 70
        z = yz.loc[iid, ['sex', 'age']]
        y = yz.loc[iid, ['smk']]
        z, y = torch.from_numpy(z.values).double(), torch.from_numpy(y.values).double()
        x = torch.from_numpy(x.values).double()
        super().__init__(data=[y, x, z])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.z = None


# class Dataset2(Dataset):
#     def __init__(self, gene, data="mice", target=None):
#         folder = os.path.join("..", "..", "Data", data, "")
#         x = pd.read_csv(folder + str(gene) + ".csv", index_col=0)
#         yz = pd.read_csv(folder + "Y.csv", index_col=0)
#         if (target is not None) and (data != target):
#             experience = os.path.join("..", "..", "Data", target, str(gene)+".csv")
#             x = x[pd.read_csv(experience, index_col=0).columns.values]
#         iid = np.intersect1d(yz.index.values, x.index.values)
#         x = x.loc[iid, :]
#         z = yz.loc[iid, ['sex']]
#         y = yz.loc[iid, ['weight']]
#         z, y = torch.from_numpy(z.values).double(), torch.from_numpy(y.values).double()
#         x = torch.from_numpy(x.values).double()
#         super().__init__(data=[y, x, z])
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.to(device)
#         self.z = None


class Dataset2(Dataset):
    def __init__(self, gene="CHRNA5", data=None):
        if data == "sage":
            x = pd.read_csv("../../Data_old/" + gene + "/g_ea.csv", index_col=0)
        else:
            x = pd.read_csv("../../Data_old/" + gene + "/g_ukb.csv", index_col=0)
        z = pd.read_csv("../../Data_old/Phe/x_" + data + ".csv", index_col=0)
        y = pd.read_csv("../../Data_old/Phe/y_" + data + ".csv", index_col=0)
        iid = np.intersect1d(z.index.values, y.index.values)
        iid = np.intersect1d(iid, x.index.values)
        z, y, x = z.loc[iid], y.loc[iid], x.loc[iid]
        x = torch.from_numpy(x.values).double()
        z = torch.from_numpy(z.values).double()
        y = torch.from_numpy(y.values).double()
        super().__init__(data=[y, x, z])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.z = None

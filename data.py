import copy
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from dataset import Dataset


class Data(Dataset):
    def __init__(self, data, classification=False):
        if len(data) == 4:
            self.functional = False
        else:
            self.functional = True
            self.loc = data.pop()
        self.scale_ratio = 1
        super().__init__(data, classification=classification)

    def split(self, seq):
        res = []
        for seq_ in seq:
            res_ = copy.deepcopy(self)
            if res_.z is not None:
                res_.z = self.z[seq_]
            res_.y, res_.x = res_.y[seq_], res_.x[seq_]
            if self.functional:
                res_.loc = res_.loc[seq_]
            res.append(res_)
        return res

    def func(self):
        self.functional = True
        # self.pos = np.asarray(self.x.columns.astype('int'))
        self.loc = self.z[['age']].to_numpy().ravel()
        # pos = np.asarray(self.x.columns.astype('int'))
        # self.pos = (pos - min(pos)) / (max(pos) - min(pos))
        # loc = self.z[['age']].to_numpy().ravel()
        # self.loc = (loc - 13) / 70

    def scale_std(self):
        x_mean, x_std = np.mean(self.x.values), np.std(self.x.values)
        points_ratio = len(self.pos)/(max(self.pos) - min(self.pos))
        std_ratio = (points_ratio*(x_mean**2 + x_std**2) -
                     (points_ratio**2) * (x_mean**2)) ** 0.5
        self.scale_ratio = 1 / std_ratio / 5
        return self.scale_ratio


class Data1(Data):
    def __init__(self, gene, data="ukb", race=1001, target=None,
                 functional=False):
        folder = os.path.join("..", "Data", data, "")
        x = pd.read_csv(folder + gene + ".csv", index_col=0)
        yz = pd.read_csv(folder + "Y.csv", index_col=0)
        if race is not None:
            if race//10 == 0:
                index1 = (yz.loc[:, 'eth_org']//1000 == race)
                index2 = (yz.loc[:, 'eth_org'] == race)
                race_index = index1 | index2
            else:
                race_index = (yz.loc[:, 'eth_org'] == race)
            yz = yz.loc[race_index, :]
        if target is None:
            x = x.loc[:, x.isnull().sum() / x.shape[0] < 0.01]
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp_mean.fit(x)
            x = pd.DataFrame(data=imp_mean.transform(x.values),
                             index=x.index, columns=x.columns)
        else:
            x = x[target.pos.astype('str')]
            x = x.dropna()
            functional = target.functional
        iid = np.intersect1d(yz.index.values, x.index.values)
        x = x.loc[iid, :]
        yz = yz.loc[iid, :]
        z = yz.loc[:, ['sex', 'age']]
        y = yz.loc[:, ['smk']]
        pos = np.asarray(x.columns.astype('int'))
        super().__init__(data=[y, x, z, pos])
        self.x = self.scale_std() * self.x
        if functional:
            self.func()
        self.process()


class Data3(Data):
    def __init__(self, gene, target=None):
        if target is None:
            data = "mice"
        else:
            data = "rat"
        folder = os.path.join("..", "Data", data, "")
        x = pd.read_csv(folder + str(gene) + ".csv", index_col=0)
        yz = pd.read_csv(folder + "Y.csv", index_col=0)
        iid = np.intersect1d(yz.index.values, x.index.values)
        x = x.loc[iid, :]
        yz = yz.loc[iid, :]
        y = yz.loc[iid, ['weight']]
        pos = np.asarray(x.columns.astype('int'))
        if target is not None:
            pos = pos[np.where(min(target.pos) <= pos)]
            pos = pos[np.where(pos <= max(target.pos))]
            x = x[pos.astype('str')]
            z = yz.loc[:, ['sex']]
            z.insert(1, 'age', np.mean(target.loc))
        else:
            z = yz.loc[:, ['sex', 'age']]
            self.pos0, self.pos1 = min(pos), max(pos)
        super().__init__(data=[y, x, z, pos])
        if target is not None:
            scale_ratio = target.scale_ratio
        else:
            scale_ratio = self.scale_std()
        self.x = scale_ratio * self.x
        self.func()
        if target is None:
            self.loc0, self.loc1 = min(self.loc), max(self.loc)
        else:
            self.loc0, self.loc1 = target.loc0, target.loc1
        self.process()


class Data2(Data):
    def __init__(self, gene="CHRNA5", data="ukb", functional=False):
        if data == "sage":
            x = pd.read_csv("../Data_old/" + gene + "/g_ea.csv", index_col=0)
        else:
            x = pd.read_csv("../Data_old/" + gene + "/g_ukb.csv", index_col=0)
        z = pd.read_csv("../Data_old/Phe/x_" + data + ".csv", index_col=0)
        y = pd.read_csv("../Data_old/Phe/y_" + data + ".csv", index_col=0)
        iid = np.intersect1d(z.index.values, y.index.values)
        iid = np.intersect1d(iid, x.index.values)
        z, y, x = z.loc[iid], y.loc[iid], x.loc[iid]
        pos = np.asarray(x.columns.astype('int'))
        super().__init__(data=[y, x, z, pos])
        self.x = self.scale_std() * self.x
        if functional:
            self.func()
        self.process()

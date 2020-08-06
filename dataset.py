import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from data import Data


class Data1(Data):
    def __init__(self, gene, data="ukb", race=1001, target=None):
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
        iid = np.intersect1d(yz.index.values, x.index.values)
        x = x.loc[iid, :]
        yz = yz.loc[iid, :]
        z = yz.loc[:, ['sex', 'age']]
        y = yz.loc[:, ['smk']]
        pos = np.asarray(x.columns.astype('int'))
        loc = z[['age']].to_numpy().ravel()
        super().__init__(data=[y.values, x.values, z.values, pos, loc])
        if target is None:
            self.pos0, self.pos1 = min(pos), max(pos)
            self.loc0, self.loc1 = min(self.loc), max(self.loc)
        else:
            self.pos0, self.pos1 = target.pos0, target.pos1
            self.loc0, self.loc1 = target.loc0, target.loc1
        self.process()


class Data3(Data):
    def __init__(self, gene, data="rat", target=None):
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
        loc = z[['age']].to_numpy().ravel()
        super().__init__(data=[y.values, x.values, z.values, pos, loc])
        if target is None:
            delta_pos = (max(pos) - min(pos)) / 100
            delta_loc = (max(loc) - min(loc)) / 100
            self.pos0, self.pos1 = min(pos) - delta_pos, max(pos) + delta_pos
            self.loc0, self.loc1 = \
                min(self.loc) - delta_loc, max(self.loc) + delta_loc
        else:
            self.pos0, self.pos1 = target.pos0, target.pos1
            self.loc0, self.loc1 = target.loc0, target.loc1
        self.process()

import os

import numpy as np
import pandas as pd
# from sklearn.impute import SimpleImputer

from data import Data


class Data1(Data):
    """
    Initialization of data from UKB or SAGE
    """
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
            # x = x.loc[:, x.isnull().sum() / x.shape[0] < 0.01]
            # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            # imp_mean.fit(x)
            # x = pd.DataFrame(data=imp_mean.transform(x.values),
            #                  index=x.index, columns=x.columns)
            x = x.dropna()  
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
        self.name = data
        self.gene = gene
        self.find_interval(target)
        self.process()
        if data == "ukb":
            race = {1001: "british", 1002: "irish", 4: "black"}[race]
            self.out_path = os.path.join("..", "Output", data, race, gene, "")
        else:
            self.out_path = os.path.join("..", "Output", data, gene, "")


class Data2(Data):
    """
    Initialization of data from rat or mice
    """
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
        self.name = data
        self.gene = gene
        self.find_interval(target)
        self.process()
        self.out_path = os.path.join("..", "Output", data, gene, "")

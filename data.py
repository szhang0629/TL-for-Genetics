import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from dataset import DiscreteData, FunctionalData


class DiscreteData1(DiscreteData):
    def __init__(self, gene, data="ukb", race=None, target=None):
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
        if (target is not None) and (data != target):
            experience = os.path.join("..", "Data", target, gene + ".csv")
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
        super().__init__(data=[y, x, z])
        self.process()


class DiscreteData2(DiscreteData):
    def __init__(self, gene="CHRNA5", data=None):
        if data == "sage":
            x = pd.read_csv("../Data_old/" + gene + "/g_ea.csv", index_col=0)
        else:
            x = pd.read_csv("../Data_old/" + gene + "/g_ukb.csv", index_col=0)
        z = pd.read_csv("../Data_old/Phe/x_" + data + ".csv", index_col=0)
        y = pd.read_csv("../Data_old/Phe/y_" + data + ".csv", index_col=0)
        iid = np.intersect1d(z.index.values, y.index.values)
        iid = np.intersect1d(iid, x.index.values)
        z, y, x = z.loc[iid], y.loc[iid], x.loc[iid]
        super().__init__(data=[y, x, z])
        self.process()


class FunctionalData1(FunctionalData):
    def __init__(self, gene="CHRNA5", data=None, race=None, target=None):
        folder = os.path.join("..", "Data", data, "")
        x = pd.read_csv(folder + gene + ".csv", index_col=0)
        yz = pd.read_csv(folder + "Y.csv", index_col=0)
        if race is not None:
            if race//10 == 0:
                index1 = (yz.loc[:, 'eth_org'] // 1000 == race)
                index2 = (yz.loc[:, 'eth_org'] == race)
                race_index = index1 | index2
            else:
                race_index = (yz.loc[:, 'eth_org'] == race)
            yz = yz.loc[race_index, :]
        if (target is not None) and (data != target):
            experience = os.path.join("..", "..", "Data", target, gene + ".csv")
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
        iid = np.intersect1d(x.index.values, yz.index.values)
        z, y, x = yz.loc[iid, ['sex', 'age']], yz.loc[iid, ['smk']], x.loc[iid]
        pos = np.asarray(x.columns.astype('int'))
        pos = (pos - min(pos)) / (max(pos) - min(pos))
        loc = z[['age']].to_numpy().ravel()
        loc = (loc - 13) / 70
        super().__init__(data=[y, x, z, pos, loc])
        self.process(classification=False)


class FunctionalData2(FunctionalData):
    def __init__(self, gene="CHRNA5", data=None):
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
        pos = (pos - min(pos)) / (max(pos) - min(pos))
        loc = z[['age']].to_numpy().ravel()
        loc = (loc - 13) / 70
        super().__init__(data=[y, x, z, pos, loc])
        self.process(classification=False)

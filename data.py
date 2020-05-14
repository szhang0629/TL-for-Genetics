import numpy as np
import pandas as pd
import torch


class Dataset:
    def __init__(self, data):
        self.y, self.x, self.z = data

    def to(self, device):
        self.y = self.y.to(device)
        if type(self.x) is list:
            self.x = [x_.to(device) for x_ in self.x]
        else:
            self.x = self.x.to(device)
        if self.z is not None:
            self.z = self.z.to(device)
        else:
            self.z = None

    def split(self, seq):
        device = self.y.device
        res = []
        for seq_ in seq:
            if type(self.x) is list:
                x_ = [x_[seq_] for x_ in self.x]
            else:
                x_ = self.x[seq_]
            if self.z is None:
                z_ = None
            else:
                z_ = self.z[seq_]
            res_ = Dataset(data=[self.y[seq_], x_, z_])
            res_.to(device)
            res.append(res_)

        return res

    def process(self, classification=False):
        y = self.y
        if classification:
            y[y != 0] = 1
            y = y.squeeze()
            self.y = y.long()
        else:
            # y_indicator = y[:, 0].numpy()
            # smoked = np.arange(len(y_indicator))[y_indicator.astype(bool)]
            # if z is not None:
            #     z = z[smoked]
            # else:
            #     z = None
            # x, y = x[smoked], y[smoked]
            y = torch.log(y + 1)
            self.y = (y - torch.mean(y)) / torch.std(y)


class Dataset0(Dataset):
    def __init__(self, name="CHRNA5"):
        if type(name) is list:
            x = [pd.read_csv("../Data/" + name_ + "/g.csv", index_col=0) for name_ in name]
            x = [torch.from_numpy(x_.values).float() for x_ in x]
        else:
            x = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
            x = torch.from_numpy(x.values).float()
        z = pd.read_csv("../Data/Phe/x.csv", index_col=0)
        y = pd.read_csv("../Data/Phe/y.csv", index_col=0)

        z[['age_int']] = (z[['age_int']] - 13) / 70
        z = z[['race', 'sex', 'age_int']]
        z = torch.from_numpy(z.values).float()
        y = torch.from_numpy(y.values).float()
        super().__init__(data=[y, x, z])


class Dataset1(Dataset):
    def __init__(self, name="CHRNA5", name_data=None):
        if type(name) is list:
            x = [pd.read_csv("../Data/" + name_ + "/g_" + name_data + ".csv", index_col=0) for name_ in name]
        else:
            x = pd.read_csv("../Data/" + name + "/g_" + name_data + ".csv", index_col=0)
        z = pd.read_csv("../Data/Phe/x_" + name_data + ".csv", index_col=0)
        z[['age']] = (z[['age']] - 13) / 70
        y = pd.read_csv("../Data/Phe/y_" + name_data + ".csv", index_col=0)
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


# class Dataset2(Dataset):
#     def __init__(self, name="rat"):
#         if name == "rat":
#             x = pd.read_csv("../Data/g_ENSRNOG00000014187.csv", index_col=0)
#             z = pd.read_csv("../Data/x_rat.csv", index_col=0)
#             y = pd.read_csv("../Data/y_rat.csv", index_col=0)
#         if name == "mice":
#             x = pd.read_csv("../Data/g_ENSMUSG00000005533.csv", index_col=0)
#             x.index = x.index.astype(str)
#             z = pd.read_csv("../Data/x_mice.csv", index_col=0)
#             y = pd.read_csv("../Data/y_mice.csv", index_col=0)
#
#         iid = np.intersect1d(z.index.values, y.index.values)
#         iid = np.intersect1d(iid, x.index.values)
#         x, y, z = x.loc[iid].values, y.loc[iid].values, z.loc[iid].values
#         y = np.log(y)
#         reg = LinearRegression().fit(x, y)
#         y = y - reg.predict(x)
#         x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
#         super().__init__(data=[y, x, None])
#         self.y = (self.y - torch.mean(self.y)) / torch.std(self.y)

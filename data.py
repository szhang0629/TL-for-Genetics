import numpy as np
import pandas as pd
import torch


class Dataset:
    def __init__(self, name="CHRNA5", classification=False, device="cpu", name_data=None, data=None):
        self.device = device
        if data is None:
            if name_data is None:
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
            else:
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

            if classification:
                y[y != 0] = 1
                y = y.squeeze()
                y = y.long()
            else:
                # y_indicator = y[:, 0].numpy()
                # smoked = np.arange(len(y_indicator))[y_indicator.astype(bool)]
                # if z is not None:
                #     z = z[smoked]
                # else:
                #     z = None
                # x, y = x[smoked], y[smoked]
                # y = torch.log(y + 1)
                y = (y - torch.mean(y)) / torch.std(y)
        else:
            y, x, z = data

        self.y = y.to(device)
        if type(x) is list:
            self.x = [x_.to(device) for x_ in x]
        else:
            self.x = x.to(device)
        if z is not None:
            self.z = z.to(device)
        else:
            self.z = None

    def split(self, seq):
        if self.z is None:
            if type(self.x) is list:
                return [Dataset(data=[self.y[seq_], [x_[seq_] for x_ in self.x],
                                      None], device=self.device) for seq_ in seq]
            else:
                return [Dataset(data=[self.y[seq_], self.x[seq_], None], device=self.device) for seq_ in seq]
        else:
            if type(self.x) is list:
                return [Dataset(data=[self.y[seq_], [x_[seq_] for x_ in self.x],
                                      self.z[seq_]], device=self.device) for seq_ in seq]
            else:
                return [Dataset(data=[self.y[seq_], self.x[seq_], self.z[seq_]], device=self.device) for seq_ in seq]

    # def numpy(self):
    #     if self.z is None:
    #         return [self.y.cpu().numpy(), self.x.cpu().numpy(), None]
    #     else:
    #         return [self.y.cpu().numpy(), self.x.cpu().numpy(), self.z.cpu().numpy()]

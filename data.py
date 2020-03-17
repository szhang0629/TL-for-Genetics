import numpy as np
import pandas as pd
import torch


class Dataset:
    def __init__(self, name="CHRNA5", classification=False, device="cpu", name_data=None, data=None):
        if data is None:
            if name_data is None:
                g = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
                x = pd.read_csv("../Data/Phe/x.csv", index_col=0)
                y = pd.read_csv("../Data/Phe/y.csv", index_col=0)

                g = torch.from_numpy(g.values).float()
                x[['age_int']] = (x[['age_int']] - 13) / 70
                x = x[['race', 'sex', 'age_int']]
                x = torch.from_numpy(x.values).float()
                y = torch.from_numpy(y.values).float()
            else:
                g = pd.read_csv("../Data/" + name + "/g_" + name_data + ".csv", index_col=0)
                x = pd.read_csv("../Data/Phe/x_" + name_data + ".csv", index_col=0)
                y = pd.read_csv("../Data/Phe/y_" + name_data + ".csv", index_col=0)
                iid = np.intersect1d(np.intersect1d(g.index.values, y.index.values), x.index.values)
                g, x, y = g.loc[iid], x.loc[iid], y.loc[iid]
                x[['age']] = (x[['age']] - 13) / 70
                g, x, y = torch.from_numpy(g.values).float(), torch.from_numpy(x.values).float(), \
                          torch.from_numpy(y.values).float()

            if classification:
                y[y != 0] = 1
                y = y.squeeze()
                y = y.long()
            else:
                # y_indicator = y[:, 0].numpy()
                # smoked = np.arange(len(y_indicator))[y_indicator.astype(bool)]
                # if x is not None:
                #     x = x[smoked]
                # else:
                #     x = None
                # g, y = g[smoked], y[smoked]
                # y = torch.log(y + 1)
                y = (y - torch.mean(y)) / torch.std(y)
        else:
            y, g, x = data

        self.y, self.g = y.to(device), g.to(device)
        # if x is not None:
        #     self.x = x.to(device)
        # else:
        #     self.x = None
        self.x = None

        self.device = device
        self.data = [self.y, self.g, self.x]

    def split(self, seq):
        if self.x is None:
            return [Dataset(data=[self.y[seq_], self.g[seq_], None], device=self.device) for seq_ in seq]
        else:
            return [Dataset(data=[self.y[seq_], self.g[seq_], self.x[seq_]], device=self.device) for seq_ in seq]

    def numpy(self):
        if self.x is None:
            return [self.y.cpu().numpy(), self.g.cpu().numpy(), None]
        else:
            return [self.y.cpu().numpy(), self.g.cpu().numpy(), self.x.cpu().numpy()]

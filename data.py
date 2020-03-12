import numpy as np
import pandas as pd
import torch


class Dataset:
    def __init__(self, name="CHRNA5", classification=False, device="cpu", data=None):
        if data is None:
            g = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
            x = pd.read_csv("../Data/Phe/x.csv", index_col=0)
            y = pd.read_csv("../Data/Phe/y.csv", index_col=0)

            g = torch.from_numpy(g.values).float()
            x[['age_int']] = (x[['age_int']] - 13) / 70
            x = x[['race', 'sex', 'age_int']]
            x = torch.from_numpy(x.values).float()
            y = torch.from_numpy(y.values).float()

            if classification:
                y[y != 0] = 1
                y = y.squeeze()
                y = y.long()
            else:
                # y_indicator = y[:, 0].numpy()
                # smoked = np.arange(len(y_indicator))[y_indicator.astype(bool)]
                # g, x, y = g[smoked], x[smoked], y[smoked]
                y = torch.log(y + 1)
                y = (y - torch.mean(y)) / torch.std(y)
        else:
            y, g, x = data

        self.y, self.g, self.x = y.to(device), g.to(device), x.to(device)
        self.device = device
        self.data = [self.y, self.g, self.x]

    def split(self, seq):
        return [Dataset(data=[self.y[seq_], self.g[seq_], self.x[seq_]], device=self.device) for seq_ in seq]

    def numpy(self):
        if self.x is None:
            return [self.y.cpu().numpy(), self.g.cpu().numpy(), None]
        else:
            return [self.y.cpu().numpy(), self.g.cpu().numpy(), self.x.cpu().numpy()]

    # def select(self, seq, seq_test, keep=False):
    #     y_test, g_test = self.y[seq_test], self.g[seq_test]
    #     if keep:
    #         y, g = self.y[seq], self.g[seq]
    #     else:
    #         self.y, self.g = self.y[seq], self.g[seq]
    #     if self.x is not None:
    #         x_test = self.x[seq_test]
    #         if keep:
    #             x = self.x[seq]
    #         else:
    #             self.x = self.x[seq]
    #     else:
    #         x_test = None
    #     if keep:
    #         data = [y, g, x]
    #     else:
    #         self.data = [self.y, self.g, self.x]
    #     if keep:
    #         return data, [y_test, g_test, x_test]
    #     else:
    #         return [y_test, g_test, x_test]

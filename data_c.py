import pandas as pd
import torch


def data(name):
    g = pd.read_csv("../Data/" + name + "/g.csv", index_col=0)
    x = pd.read_csv("../Data/Phe/x.csv", index_col=0)
    y = pd.read_csv("../Data/Phe/y.csv", index_col=0)

    g = torch.from_numpy(g.values).float()
    x[['age_int']] = (x[['age_int']] - 13)/70
    x = torch.from_numpy(x.values).float()
    y = torch.from_numpy(y.values).float()

    y = torch.log(y + 1)
    y[y != 0] = 1
    y = y.squeeze()
    y = y.long()

    return g, x, y

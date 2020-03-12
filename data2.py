"""Select unaligned Data"""
import numpy as np
import pandas as pd
import torch


def align_xy(x, y):
    """Give Intersected Alighed Data"""
    x_iid = x.index.values
    y_iid = y.index.values
    iid = np.intersect1d(x_iid, y_iid)
    x = x.loc[iid].values
    y = y.loc[iid].values

    return x, y


def data2(name_gene, name_data, classification=False):
    g = pd.read_csv("../Data/"+name_gene+"/g_"+name_data+".csv", index_col=0)
    y = pd.read_csv("../Data/Phe/y_"+name_data+".csv", index_col=0)
    g, y = align_xy(g, y)
    g, y = torch.from_numpy(g).float(), torch.from_numpy(y).float()

    if classification:
        y[y != 0] = 1
        y = y.squeeze()
        y = y.long()
    else:
        y_indicator = y[:, 0].numpy()
        smoked = np.arange(len(y_indicator))[y_indicator.astype(bool)]
        g, y = g[smoked], y[smoked]
        # y = torch.log(y + 1)
        y = (y - torch.mean(y)) / torch.std(y)

    return g, y

import numpy as np
import pandas as pd
import torch


def data5(name_data, classification=False):
    """Train 5 Genes Together"""
    names_gene = ["CHRNA3", "CHRNA5", "CHRNA6", "CHRNB3", "CHRNB4"]
    g = [pd.read_csv("../Data/" + name_gene + "/g_" + name_data + ".csv", index_col=0)
         for name_gene in names_gene]
    x = pd.read_csv("../Data/Phe/x_" + name_data + ".csv", index_col=0)
    y = pd.read_csv("../Data/Phe/y_" + name_data + ".csv", index_col=0)
    iid = y.index.values
    for g_ in g:
        iid = np.intersect1d(iid, g_.index.values)

    g = [torch.from_numpy(g_.loc[iid].values).float() for g_ in g]
    x[['age']] = (x[['age']] - 13) / 70
    x = torch.from_numpy(x.loc[iid].values).float()
    y = torch.from_numpy(y.loc[iid].values).float()

    if classification:
        y[y != 0] = 1
        y = y.squeeze()
        y = y.long()
    else:
        y_indicator = y[:, 0].numpy()
        idx = np.arange(len(y_indicator))[y_indicator.astype(bool)]
        g = [g_.loc[idx] for g_ in g]
        x, y = x.loc[idx], y.loc[idx]
        # y = (y - torch.mean(y)) / torch.std(y)

    return g, x, y

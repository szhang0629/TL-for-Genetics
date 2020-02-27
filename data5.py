import numpy as np
import pandas as pd
import torch


def data5(name_data):
    names_gene = ["CHRNA3", "CHRNA5", "CHRNA6", "CHRNB3", "CHRNB4"]
    g = [pd.read_csv("../Data/" + name_gene + "/g_" + name_data + ".csv", index_col=0)
         for name_gene in names_gene]
    x = pd.read_csv("../Data/Phe/x_" + name_data + ".csv", index_col=0)
    y = pd.read_csv("../Data/Phe/y_" + name_data + ".csv", index_col=0)
    iid = y.index.values
    for g_ in g:
        iid = np.intersect1d(iid, g_.index.values)

    g = [g_.loc[iid] for g_ in g]
    x = x.loc[iid]
    y = y.loc[iid]

    idx = (y.smk != 0)
    g = [g_.loc[idx] for g_ in g]
    x = x.loc[idx]
    y = y.loc[idx]

    y = torch.from_numpy(y.values).float()
    x = torch.from_numpy(x.values).float()
    g = [torch.from_numpy(g_.values).float() for g_ in g]
#    y = (y - torch.mean(y)) / torch.std(y)

    return g, x, y

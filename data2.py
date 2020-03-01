'''Select unaligned Data'''
import numpy as np
import pandas as pd
import torch


def align_xy(x, y):
    '''Give Intersected Alighed Data'''
    x_iid = x.index.values
    y_iid = y.index.values
    iid = np.intersect1d(x_iid, y_iid)
    x = x.loc[iid].values
    y = y.loc[iid].value

    return x, y


def data2(name_gene, name_data):
    g = pd.read_csv("../Data/"+name_gene+"/g_"+name_data+".csv", index_col=0)
    y = pd.read_csv("../Data/Phe/y_"+name_data+".csv", index_col=0)
    x, y = align_xy(g, y)
    x[['age']] = (x[['age']] - 13)/70
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    y = (y - torch.mean(y)) / torch.std(y)

    return x, y

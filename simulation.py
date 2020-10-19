import os
import random
from skfda.representation.basis import Fourier, BSpline, Monomial

import numpy as np
import pandas as pd

from data import Data


class Simulation(Data):
    """Simulation Data"""
    def __init__(self, seed_index, target=None):
        random.seed(seed_index)
        n, p = 1000, 200
        maf_interval = [0.01, 0.5]
        # Get data for use
        ped = pd.read_csv("../Data/ped.ped", sep="\t", header=None)
        info = pd.read_csv("../Data/info.info", sep="\t")
        # ped: 1092 obs. of 12735 variables
        # info: 12735 obs. of 4 variables
        # n = 1092, p = 12735 high dimensional question

        # N samples chosen among 1092 objects for simulation
        smp_idx = random.choices(range(ped.shape[0]), k=n)  # smp as sample
        # order
        # ## maf interval SNP index
        maf_idx = (info.maf > maf_interval[0]) & (info.maf < maf_interval[1])
        geno = ped.loc[smp_idx, maf_idx]
        pos = info.pos
        pos = pos[maf_idx]

        # Delete void data
        pos = pos.loc[geno.std() > 0.1, ]
        geno = geno.loc[:, geno.std() > 0.1]  # get rid of individuals with no
        # variability
        pos = pos.to_numpy()

        # Truncated SNP index
        seg_pos = random.choice(range(len(pos) - p))
        idx_trun = range(seg_pos, seg_pos + p)
        geno = geno.iloc[:, idx_trun]
        pos = pos[idx_trun]
        x = geno.to_numpy()

        bss0 = BSpline(n_basis=10, order=6)
        t = (pos - min(pos)) / (max(pos) - min(pos))
        mat0 = bss0.evaluate(t)[:, :, 0]
        y = np.zeros([x.shape[0], 1])
        for i in range(20):
            y += np.random.normal() * \
                 (x ** np.random.uniform(1 / 4, 4)) @ \
                 mat0[range(i//2, i//2 + 1), :].T

        y = y / y.std()
        y += np.random.normal(scale=0.3, size=y.shape)
        super().__init__(data=[y, x, None, pos, None])
        self.name = str(n) + "_" + str(p)
        self.gene = str(seed_index)
        self.find_interval(target)
        self.process()
        self.out_path = os.path.join("..", "Output", "Sim", "")

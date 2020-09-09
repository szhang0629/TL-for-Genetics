from abc import ABC

import numpy as np
import torch

from layer import LayerB, LayerC, LayerD
from net import Net


class FDNN(Net, ABC):
    def __init__(self, dims):
        self.realize = False
        super(FDNN, self).__init__()
        if len(dims) == 1:
            models = [LayerD(dims[0])]
        elif dims[-1] == 1:
            models = [LayerB(dims[i], dims[i + 1])
                      for i in range(len(dims) - 2)]
            models = models + [LayerC(dims[-2])]
        else:
            models = [LayerB(dims[i], dims[i + 1])
                      for i in range(len(dims) - 1)]
        if len(dims) <= 2:
            self.str_units = "0"
        else:
            self.str_units = str("_".join(str(x) for x in dims[1:(-1)]))
        self.std_ratio = None
        self.layers = len(models)
        self.hyper_lamb = [10**x for x in range(-4, 2)]
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        if not self.realize:
            self.realization(dataset)
        res = self.model0(dataset.x * self.std_ratio)
        for i in range(1, self.layers):
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
        return res * self.std + self.mean

    def realization(self, dataset):
        device = dataset.x.device
        if self.std_ratio is None:
            self.std_ratio = dataset.scale_std()
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            if hasattr(model, 'bs0'):
                if i == 0:
                    pos = dataset.pos
                    pos0 = dataset.pos0 if hasattr(dataset, 'pos0') \
                        else min(pos)
                    pos1 = dataset.pos1 if hasattr(dataset, 'pos1') \
                        else max(pos)
                    model.bs0.length = pos1 - pos0 + 1
                    pos = (pos - pos0) / (pos1 - pos0)
                else:
                    pos = np.arange(1/model.bs0.n_basis/2, 1,
                                    1/model.bs0.n_basis)
                    model.bs0.length = len(pos)
                model.bs0.evaluate(pos)
                model.bs0.to(device)
            if hasattr(self, 'model' + str(i + 1)):
                model.index = None
                loc = np.arange(1/model.bs1.n_basis/2, 1, 1/model.bs1.n_basis)
            else:
                if hasattr(model, 'bs1'):
                    model.index = -1
                    loc = dataset.loc
                    loc0 = dataset.loc0 if hasattr(dataset, 'loc0') \
                        else min(loc)
                    loc1 = dataset.loc1 if hasattr(dataset, 'loc1') \
                        else max(loc)
                    loc = (loc - loc0) / (loc1 - loc0)
                else:
                    break
            model.bs1.evaluate(loc)
            model.bs1.to(device)

    def fit_init(self, dataset):
        self.mean, self.std = dataset.y.mean(), dataset.y.std()
        self.size = dataset.x.shape[0]
        self.realization(dataset)
        self.realize = True

    def fit_end(self):
        self.realize = False
        print(self.epoch, self.loss, self.penalty().tolist())

    def penalty(self):
        # return getattr(self, 'model' + str(self.layers-1)).pen() * self.lamb
        penalty = 0
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            penalty += model.pen()
        return penalty * self.lamb_sca

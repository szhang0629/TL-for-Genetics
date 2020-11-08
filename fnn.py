from abc import ABC

import numpy as np

from layer import LayerB, LayerC, LayerD
from net import Net
from haar import Haar as Basis
# from basis import Basis
from torch.nn import ReLU


class FNN(Net, ABC):
    def __init__(self, dims, lamb=None):
        self.realize = False
        super(FNN, self).__init__(lamb)
        if len(dims) == 1:
            models = [LayerD(Basis(dims[0]))]
            self.method_name = "FBase"
        if len(dims) == 2:
            if dims[-1] == 1:
                models = [LayerC(dims[-2])]
            else:
                models = [LayerB(Basis(dims[0]), Basis(dims[1]))]
        mid = False
        if len(dims) > 2:
            models = [LayerB(Basis(dims[0]), Basis(dims[1], mid), bias=False)]
            models += [LayerB(Basis(dims[i], mid), Basis(dims[i+1], mid))
                       for i in range(1, len(dims) - 2)]
            if dims[-1] == 1:
                models += [LayerC(Basis(dims[-2], mid))]
            else:
                models += [LayerB(Basis(dims[-2], mid), Basis(dims[-1]))]
        if len(dims) <= 2:
            self.str_units = "0"
        else:
            self.str_units = str("_".join(str(x) for x in dims[1:(-1)]))
        self.std_ratio = None
        self.layers = len(models)
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])
        # self.lamb_ = 1e4

    def forward(self, dataset):
        if not self.realize:
            self.realization(dataset)
        # res = self.model0(dataset.x * self.std_ratio)
        res = self.model0(dataset.x)
        for i in range(1, self.layers):
            # res = ReLU()(res)
            res = res.sigmoid()
            res = getattr(self, 'model' + str(i))(res)
        return res * self.std + self.mean

    def to(self, device):
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            model = model.to(device)

    def realization(self, dataset):
        device = dataset.y.device
        # if self.std_ratio is None:
        #     self.std_ratio = dataset.scale_std()
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            if hasattr(model, 'bs0'):
                if i == 0:
                    if not hasattr(dataset, 'pos0'):
                        dataset.pos0 = min(dataset.pos) - 1/model.bs0.n_basis/2
                    if not hasattr(dataset, 'pos1'):
                        dataset.pos1 = max(dataset.pos) + 1/model.bs1.n_basis/2
                    model.bs0.length = dataset.pos1 - dataset.pos0 + 1
                    pos = (dataset.pos - dataset.pos0) / (model.bs0.length-1)
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
                    if not hasattr(dataset, 'loc0'):
                        dataset.loc0 = min(dataset.loc) - 1/model.bs1.n_basis/2
                    if not hasattr(dataset, 'loc1'):
                        dataset.loc1 = max(dataset.loc) + 1/model.bs1.n_basis/2
                    loc = (dataset.loc - dataset.loc0) / (dataset.loc1 -
                                                          dataset.loc0)
                else:
                    break
            model.bs1.evaluate(loc)
            model.bs1.to(device)

    def fit_init(self, dataset):
        self.mean, self.std = dataset.y.mean(), dataset.y.std()
        self.size = dataset.y.shape[0]
        self.realization(dataset)
        self.realize = True

    def fit_end(self):
        self.realize = False
        print(self.epoch, self.loss)

    def penalty(self):
        if self.layers == 1:
            return self.model0.pen(lamb0=self.lamb, lamb1=self.lamb)
        penalty = 0
        for i in range(self.layers-1):
            model = getattr(self, 'model' + str(i))
            penalty += model.pen(lamb0=1e-9, lamb1=1e-3)
        model = getattr(self, 'model' + str(self.layers-1))
        penalty += model.pen(lamb0=1e0, lamb1=self.lamb)
        return penalty / self.size

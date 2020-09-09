from abc import ABC

import torch

from layer import LayerA
from net import Net


class DNN(Net, ABC):
    """
    A class to represent deep neural network combined by layers defined above
    """
    def __init__(self, dims):
        super(DNN, self).__init__()
        if len(dims) == 2:
            self.str_units = "0"
        else:
            self.str_units = str("_".join(str(x) for x in dims[1:(-1)]))
        models = [LayerA(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.layers = len(models)
        # self.mean_ratio, self.std_ratio = None, None
        self.hyper_lamb = [10 ** x for x in range(-4, 2)]
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        # res = dataset.x  # (dataset.x - self.mean_ratio)/self.std_ratio
        res = self.model0(dataset.x, dataset.z)
        for i in range(1, self.layers):
            # res = torch.nn.LeakyReLU()(res)
            res = torch.sigmoid(res)
            res = getattr(self, 'model' + str(i))(res)
        return res*self.std + self.mean

    def fit_init(self, dataset):
        # if self.mean_ratio is None:
        #     self.mean_ratio = dataset.x.mean(0)
        # if self.std_ratio is None:
        #     self.std_ratio = dataset.x.std(0)*(dataset.x.shape[1]**0.5)
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.size = dataset.x.shape[0]

    def fit_end(self):
        print(self.epoch, self.loss, self.penalty().tolist())

    def penalty(self):
        penalty = 0
        for i in range(self.layers-1):
            model = getattr(self, 'model' + str(i))
            penalty += model.pen() * 10e-3 / (self.size ** 0.5)
        model = getattr(self, 'model' + str(self.layers-1))
        penalty += model.pen() * self.lamb_sca
        return penalty


class DNNDo(Net, ABC):
    """
    A class to represent deep neural network using Dropout
    """
    def __init__(self, dims):
        super(DNNDo, self).__init__()
        if len(dims) == 2:
            self.str_units = "0"
        else:
            self.str_units = str("_".join(str(x) for x in dims[1:(-1)]))
        models = [LayerA(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.layers = len(models)
        self.hyper_lamb = [10 ** x for x in range(-4, 2)]
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])
            setattr(self, "dropout" + str(i), torch.nn.Dropout())
        for i in range(1, self.layers):
            setattr(self, "activation" + str(i), torch.nn.LeakyReLU())

    def forward(self, dataset):
        res = self.dropout0(dataset.x)
        res = self.model0(res, dataset.z)
        for i in range(1, self.layers):
            res = getattr(self, 'activation' + str(i))(res)
            res = getattr(self, 'dropout' + str(i))(res)
            res = getattr(self, 'model' + str(i))(res)
        return res*self.std + self.mean

    def fit_init(self, dataset):
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.size = dataset.x.shape[0]

    def fit_end(self):
        print(self.epoch, self.loss)

    def penalty(self):
        return 0

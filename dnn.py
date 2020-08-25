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
        self.hyper_lamb = [10 ** x for x in range(-4, 2)]
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        res = self.model0(dataset.x, dataset.z)
        for i in range(1, self.layers):
            res = torch.nn.LeakyReLU()(res)
            # res = torch.nn.Tanh()(res)
            res = getattr(self, 'model' + str(i))(res)
        return res*self.std + self.mean

    def fit_init(self, dataset):
        self.mean, self.std = torch.mean(dataset.y), torch.std(dataset.y)
        self.size = dataset.x.shape[0]

    def fit_end(self):
        print(self.epoch, self.loss, self.penalty().tolist())

    def penalty(self):
        penalty = 0
        for name, param in self.named_parameters():
            if param.requires_grad and "fc" in name:
                # and not name.endswith(".bias"):
                penalty += torch.sum(param ** 2)
        return penalty * self.lamb


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

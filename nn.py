from abc import ABC
from torch import nn as nn

from layer import LayerA
from net import Net

import torch


class NN(Net, ABC):
    """
    A class to represent deep neural network combined by layers defined above
    """

    def __init__(self, dims, lamb=None):
        super(NN, self).__init__(lamb)
        torch.set_default_tensor_type(torch.DoubleTensor)
        if len(dims) == 2:
            self.str_units = "0"
        else:
            self.str_units = str("_".join(str(x) for x in dims[1:(-1)]))
        self.layers = len(dims) - 1
        firstlayer = [LayerA(dims[0], dims[1])]
        models = firstlayer + [LayerA(dims[i], dims[i + 1])
                               for i in range(1, len(dims) - 2)]
        lastlayer = LayerA(dims[-2], dims[-1], bias=False)
        models.append(lastlayer)
        self.mean_ratio, self.std_ratio = None, None
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        # res = (dataset.x - self.mean_ratio)/self.std_ratio
        res = dataset.x
        res = self.model0(res, dataset.z)
        for i in range(1, self.layers):
            # res = getattr(self, 'model' + str(i)).do(res)
            res = res.sigmoid()
            # res = ReLU()(res)
            res = getattr(self, 'model' + str(i))(res)
        return res * self.std + self.mean
        # return res

    def to(self, device):
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            model = model.to(device)

    def fit_init(self, dataset):
        # if self.mean_ratio is None:
        #     self.mean_ratio = dataset.x.mean(0)
        # if self.std_ratio is None:
        #     self.std_ratio = dataset.x.std(0)  # *(dataset.x.shape[1]**0.5)
        self.to(dataset.y.device)
        self.epoch, self.loss = 0, float('inf')
        self.mean = dataset.y.mean()
        self.std = ((dataset.y - self.mean) ** 2).mean() ** 0.5
        self.size = dataset.y.shape[0]
        torch.random.manual_seed(0)
        # self.init.normal_(.1 / self.lamb)
        # firstlayer = self.model0
        # firstlayer.fc.weight.data.normal_(0., 1e-3)
        # for i in range(1, self.layers - 1):
        #     getattr(self, 'model' + str(i)).fc.weight.data.normal_(0, 1e-4)
        #     getattr(self, 'model' + str(i)).fc.bias.data.normal_(0, 1e-4)
        lastlayer = getattr(self, 'model' + str(self.layers - 1))
        lastlayer.fc.weight.data.normal_(0., .1 / self.lamb)
        # lastlayer.fc.weight.data.fill_(0.)
        if lastlayer.fc.bias is not None:
            lastlayer.fc.bias.data.normal_(0., .1 / self.lamb)

    def fit_end(self):
        print(self.epoch, self.loss, self.penalty().tolist())

    def penalty(self):
        # penalty = self.model0.pen2() * 1e-8
        # penalty += sum([getattr(self, 'model' + str(i)).pen2()
        #                for i in range(1, self.layers - 1)]) * 1e-8
        # for i in range(self.layers-1):
        #     model = getattr(self, 'model' + str(i))
        #     penalty += model.pen() * 1e-9
        model = getattr(self, 'model' + str(self.layers - 1))
        # penalty += model.pen() * self.lamb
        penalty = model.pen2() * self.lamb
        return penalty

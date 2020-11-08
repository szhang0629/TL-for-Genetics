from abc import ABC

from layer import LayerA
from net import Net


class NN(Net, ABC):
    """
    A class to represent deep neural network combined by layers defined above
    """
    def __init__(self, dims, lamb=None):
        super(NN, self).__init__(lamb)
        if len(dims) == 2:
            self.str_units = "0"
        else:
            self.str_units = str("_".join(str(x) for x in dims[1:(-1)]))
        models = [LayerA(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.layers = len(models)
        # self.mean_ratio, self.std_ratio = None, None
        for i in range(self.layers):
            setattr(self, "model" + str(i), models[i])

    def forward(self, dataset):
        # res = (dataset.x - self.mean_ratio)/self.std_ratio
        res = self.model0(dataset.x, dataset.z)
        for i in range(1, self.layers):
            res = res.sigmoid()
            res = getattr(self, 'model' + str(i))(res)
        return res*self.std + self.mean

    def to(self, device):
        for i in range(self.layers):
            model = getattr(self, 'model' + str(i))
            model = model.to(device)

    def fit_init(self, dataset):
        # if self.mean_ratio is None:
        #     self.mean_ratio = dataset.x.mean(0)
        # if self.std_ratio is None:
        #     self.std_ratio = dataset.x.std(0)  # *(dataset.x.shape[1]**0.5)
        self.mean, self.std = dataset.y.mean(), dataset.y.std()
        self.size = dataset.y.shape[0]

    def fit_end(self):
        print(self.epoch, self.loss, self.penalty().tolist())
        return

    def penalty(self):
        penalty = 0
        for i in range(self.layers-1):
            model = getattr(self, 'model' + str(i))
            penalty += model.pen() * 1e-7
        model = getattr(self, 'model' + str(self.layers-1))
        penalty += model.pen()
        return penalty * self.lamb

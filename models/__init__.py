# __init__.py
import torch
import torch.nn as nn


class MyEnsemble(nn.Module):
    def __init__(self, *args):
        super(MyEnsemble, self).__init__()
        i = 0
        for arg in args:
            setattr(self, "model" + str(i), arg)
            i += 1

    def forward(self, dataset):
        if hasattr(self, 'model1'):
            x1 = self.model0(dataset.x)
            return self.model1(x1, dataset.z)
        else:
            return self.model0(dataset.x, dataset.z)

# class MyEnsemble(nn.Module):
#     def __init__(self, model_a, model_b, model_c):
#         super(MyEnsemble, self).__init__()
#         self.ModelA = model_a
#         self.ModelB = model_b
#         self.ModelC = model_c
#
#     def forward(self, x, z):
#         x1 = self.ModelA(x)
#         # x2 = self.ModelB(x1)
#         y = self.ModelC(x1, z)
#         return y

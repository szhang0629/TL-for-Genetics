# __init__.py
import torch
import torch.nn as nn


class MyEnsemble(nn.Module):
    def __init__(self, model_a, model_c):
        super(MyEnsemble, self).__init__()
        self.ModelA = model_a
        self.ModelC = model_c

    def forward(self, x, z):
        x1 = self.ModelA(x)
        y = self.ModelC(x1, z)
        return y

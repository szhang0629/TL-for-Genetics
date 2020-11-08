import numpy as np
import torch
from skfda.representation.basis import Fourier, BSpline, Monomial
import matplotlib.pyplot as plt


class Basis:
    def __init__(self, n_basis, linear=False):
        self.n_basis = n_basis - linear
        self.linear = linear
        # self.bss = Fourier((0, 1), n_basis=self.n_basis, period=1)
        self.bss = BSpline(n_basis=self.n_basis, order=4)
        bss2 = self.bss.derivative(order=1)
        coefficients = torch.from_numpy(bss2.coefficients)
        pen2 = torch.from_numpy(bss2.basis.gram_matrix())
        self.pen0 = torch.from_numpy(self.bss.gram_matrix())
        self.pen2 = coefficients @ pen2 @ coefficients.t()
        if self.linear:
            bs_mon = Monomial((0, 1), n_basis=2)
            mat1 = self.bss.inner_product_matrix(bs_mon)[:, 1]
            mat1 = torch.from_numpy(mat1).reshape(1, self.n_basis)
            self.pen0 = torch.cat([mat1, self.pen0], 0)
            mat1 = mat1.reshape(self.n_basis, 1)
            mat1 = torch.cat([torch.ones([1, 1])*1/3, mat1], 0)
            self.pen0 = torch.cat([mat1, self.pen0], 1)
            self.pen2 = torch.cat([torch.zeros(1, self.n_basis), self.pen2], 0)
            self.pen2 = torch.cat([torch.zeros(self.n_basis + 1, 1),
                                   self.pen2], 1)
        self.mat = None
        self.length = None

    def evaluate(self, point):
        mat = self.bss.evaluate(point)[:, :, 0]  # len(point) = col(mat)
        self.mat = torch.from_numpy(mat.T)
        if self.linear:
            linear_func = torch.from_numpy(point).reshape(len(point), 1)
            self.mat = torch.cat([linear_func, self.mat], 1)

    def to(self, device):
        self.mat = self.mat.to(device)
        self.pen0 = self.pen0.to(device)
        self.pen2 = self.pen2.to(device)

    def pen_1d(self, param, lamb1=1.):
        # pen = param[-self.n_basis:] @ self.pen2 @ param[-self.n_basis:]
        pen = param @ self.pen2 @ param * lamb1
        # if pen > 0:
        #     var = torch.sum(param[1-self.n_basis:] ** 2)
        #     pen = pen/var
        return pen

    def pen_2d(self, param, basis, lamb0=1., lamb1=1.):
        # param = param[-basis.n_basis:, -self.n_basis:]
        pen = torch.trace(basis.pen0 @ ((param.t() @ self.pen2) @ param)) \
              * lamb1 + torch.trace(self.pen0 @ ((param @ basis.pen2)
                                                 @ param.t())) * lamb0
        # pen = torch.trace(((param @ self.pen2) @ param.t()) +
        #                   ((param.t() @ basis.pen2) @ param)) * 0.5
        # if pen > 0:
        #     var = torch.sum(param[1 - basis.n_basis:, 1 - self.n_basis:] ** 2)
        #     pen = pen / var
        return pen

    def plot(self, param):
        param = np.asarray(torch.reshape(param, (-1,)).tolist())
        param = param.reshape([len(param), 1])
        x = np.linspace(0, 1, 100)
        y = sum(param * self.bss.evaluate(x)[:, :, 0])
        plt.plot(x, y, 'r')
        plt.show()

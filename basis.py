import torch
from skfda.representation.basis import BSpline, Fourier


class Basis:
    def __init__(self, n_basis):
        self.n_basis = n_basis
        self.bss = Fourier((0, 1), n_basis=n_basis, period=2)
        # self.bss = BSpline(n_basis=n_basis, order=4)
        # bss2 = Fourier((0, 1), n_basis=n_basis, period=1).derivative(order=2)
        bss2 = self.bss.derivative(order=2)
        coefficients = torch.from_numpy(bss2.coefficients)
        pen2 = torch.from_numpy(bss2.basis.gram_matrix())
        self.pen0 = torch.from_numpy(self.bss.gram_matrix())
        # bss = Fourier((0, 1), n_basis=n_basis, period=1)
        # self.pen0 = torch.from_numpy(bss.gram_matrix())
        self.pen2 = coefficients @ pen2 @ coefficients.t()
        self.mat = None
        self.length = None
        bs0 = Fourier((0, 1), n_basis=1, period=1)
        self.integral = torch.from_numpy(self.bss.inner_product_matrix(bs0))

    def evaluate(self, point):
        mat = self.bss.evaluate(point)[:, :, 0]
        self.mat = torch.from_numpy(mat.T)

    def to(self, device):
        self.mat = self.mat.to(device)
        self.pen0 = self.pen0.to(device)
        self.pen2 = self.pen2.to(device)
        self.integral = self.integral.to(device)

    def pen_1d(self, param):
        pen = param @ self.pen2 @ param
        if pen > 0:
            mean = param @ self.integral
            var = param @ self.pen0 @ param - mean @ mean
            pen = pen / var
        else:
            pen = 0
        return pen

    def pen_2d(self, param, basis):
        pen = torch.trace(basis.pen0 @ ((param @ self.pen2) @ param.t()) +
                          self.pen0 @ ((param.t() @ basis.pen2) @ param)) * 0.5
        if pen > 0:
            mean = torch.sum(param * (basis.integral.t() @ self.integral))
            var = torch.trace((basis.pen0 @ ((param @ self.pen0) @
                                             param.t()))) - (mean ** 2)
            pen = pen / var
        else:
            pen = 0
        return pen

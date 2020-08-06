import numpy as np
import torch
import math


class Haar:
    def __init__(self, n_basis):
        self.level = int(math.log2(n_basis))
        self.n_basis = n_basis
        self.linear = False
        self.phi = lambda x: ((x >= 0) & (x < 1))*1
        self.psi = lambda x: ((x >= 0) & (x < 0.5))*1 - ((x >= 0.5) & (x < 1))*1
        self.phi_j_k = lambda x, j, k: 2**(j/2) * self.phi(2**j * x - k)
        self.psi_j_k = lambda x, j, k: 2**(j/2) * self.psi(2**j * x - k)
        self.mat = None
        self.length = None
        self.pen2 = np.zeros([self.n_basis, self.n_basis])
        self.pen2[2, 2] = 4
        for level1 in range(1, self.level):
            base1 = 2**level1
            for k in range(1, base1):
                self.pen2[base1 + k - 1, base1 + k] = base1
                self.pen2[base1 + k, base1 + k] = 6 * base1
            self.pen2[base1, base1] = 5 * base1
            self.pen2[2 * base1 - 1, 2 * base1 - 1] = 5 * base1
            # self.pen2[base1//2, base1] = -2 ** (level1+0.5)
            # self.pen2[base1 - 1, base1*2-1] = -2 ** (level1+0.5)
            for l in range(level1):
                base2 = 2 ** l
                interval = 2 ** (level1-l)
                value = 2 ** (l/2 + level1/2)
                for m in range(1, base2):
                    value_m = value * (((-1)**m) * 1.5 - 0.5)
                    self.pen2[base2 + m, base1 + m * interval - 1] = value_m
                    self.pen2[base2 + m, base1 + m * interval] = value_m

        self.pen2 = self.pen2 + self.pen2.T - np.diag(np.diag(self.pen2))
        self.pen2 *= 2**self.level
        self.pen2 = torch.from_numpy(self.pen2)

    def evaluate(self, point):
        self.mat = np.zeros([self.n_basis, len(point)])
        self.mat[0, :] = self.phi_j_k(point, 0, 0)
        i = 1
        for j in range(self.level):
            for k in range(2**j):
                self.mat[i, :] = self.psi_j_k(point, j, k)
                i += 1
        self.mat = torch.from_numpy(self.mat.T)

    def to(self, device):
        self.mat = self.mat.to(device)
        self.pen2 = self.pen2.to(device)

    def pen_1d(self, param):
        pen = param @ self.pen2 @ param
        # if pen > 0:
        #     var = torch.sum(param[1-self.n_basis:] ** 2)
        #     pen = pen/var
        return pen

    def pen_2d(self, param, basis):
        pen = torch.trace(((param @ self.pen2) @ param.t()) +
                          ((param.t() @ basis.pen2) @ param)) * 0.5
        # if pen > 0:
        #     var = torch.sum(param[1 - basis.n_basis:, 1 - self.n_basis:] ** 2)
        #     pen = pen / var
        return pen


# def haar(f, interval, level):
#     c0 = quadgl(lambda t: f(t) * phi_j_k(t, 0, 0), interval)
#     coef = []
#     for j in range(0, level):
#         for k in range(0, 2**j):
#             djk = quadgl(lambda t: f(t) * psi_j_k(t, j, k), interval)
#             coef.append((j, k, djk))
#
#     return c0, coef
#
# def haarval(haar_coef, x):
#     c0, coef = haar_coef
#     s = c0 * phi_j_k(x, 0, 0)
#     for j, k, djk in coef:
#         s += djk * psi_j_k(x, j, k)
#     return s
#
# # --------- to plot an Haar wave
# interval = [0, 1]
# plot([lambda x: psi_j_k(x, 5, 4)], interval)
#
# nb_coeff = 5
# interval = [0, 1]
#
# fct = lambda x: x
#
# haar_coef = haar(fct, interval, nb_coeff)
# haar_series_apx = lambda x: haarval(haar_coef, x)
